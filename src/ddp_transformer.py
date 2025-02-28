import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
import math
import numpy as np
from window_dataset import time_series_dataset, collate_func, MAX_SEQ_LEN, FEATURE_DIM
from tqdm.auto import tqdm
import warnings
import os
from lossfuncion.focal_loss import focal_loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random

warnings.filterwarnings("ignore")

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class BinaryTransformer(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, d_model=128, nhead=8, 
                 num_layers=4, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.embedding.out_features)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_encoder(positions)
        x = self.layer_norm(x)
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = x * mask.unsqueeze(-1)
        pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return self.classifier(pooled).squeeze(-1)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

def get_distributed_loader(dataset, batch_size, num_workers, collate_func, world_size, shuffle, rank):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False,  # Shuffle由sampler控制
        collate_fn=collate_func,
        pin_memory=True
    )

def train_epoch(rank, model, loader, criterion, optimizer, epoch):
    model.train()
    sampler = loader.sampler
    sampler.set_epoch(epoch)
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    world_size = dist.get_world_size()
    
    progress_bar = tqdm(
        total=len(loader),
        disable=(rank != 0),
        desc=f"Training epoch {epoch}",
        bar_format="{l_bar}{bar:20}{r_bar}",
    )
    
    for batch in loader:
        inputs = batch['features'].to(rank, non_blocking=True)
        labels = batch['label'].float().to(rank, non_blocking=True)
        masks = batch['mask'].to(rank, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs, masks)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 收集每个batch的预测和标签
        probs = torch.sigmoid(outputs).detach()
        all_preds.append(probs)
        all_labels.append(labels)
        
        # 累加loss（假设loss是均值）
        total_loss += loss.detach() * len(labels)  # 使用样本数加权
        
        # 更新进度条
        if rank == 0:
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
    
    # 收集所有进程的预测和标签
    all_preds_tensor = torch.cat(all_preds).to(rank)
    all_labels_tensor = torch.cat(all_labels).to(rank)
    
    # 分配存储空间
    gathered_preds = [torch.zeros_like(all_preds_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(all_labels_tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered_preds, all_preds_tensor)
    dist.all_gather(gathered_labels, all_labels_tensor)
    
    # 合并数据并计算指标（仅在rank 0执行）
    if rank == 0:
        all_preds_combined = torch.cat(gathered_preds).cpu().numpy()
        all_labels_combined = torch.cat(gathered_labels).cpu().numpy()
        predictions = (all_preds_combined > 0.5).astype(int)
        
        # 计算总loss
        total_loss_tensor = torch.tensor(total_loss).to(rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        total_samples = len(all_labels_combined)
        avg_loss = total_loss_tensor.item() / total_samples
        
        acc = accuracy_score(all_labels_combined, predictions)
        f1 = f1_score(all_labels_combined, predictions)
        auc = roc_auc_score(all_labels_combined, all_preds_combined)
        tn, fp, fn, tp = confusion_matrix(all_labels_combined, predictions).ravel()
        return avg_loss, acc, f1, auc, [int(tn), int(fp), int(fn), int(tp)]
    return None, None, None, None, None

def evaluate(rank, model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    world_size = dist.get_world_size()
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['features'].to(rank, non_blocking=True)
            labels = batch['label'].float().to(rank, non_blocking=True)
            masks = batch['mask'].to(rank, non_blocking=True)
            
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            
            # 累加loss（假设loss是均值）
            total_loss += loss.detach() * len(labels)  # 使用样本数加权
            
            probs = torch.sigmoid(outputs)
            all_preds.append(probs)
            all_labels.append(labels)
    
    # 收集所有进程的预测和标签
    all_preds_tensor = torch.cat(all_preds).to(rank)
    all_labels_tensor = torch.cat(all_labels).to(rank)
    
    gathered_preds = [torch.zeros_like(all_preds_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(all_labels_tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered_preds, all_preds_tensor)
    dist.all_gather(gathered_labels, all_labels_tensor)
    
    if rank == 0:
        all_preds_combined = torch.cat(gathered_preds).cpu().numpy()
        all_labels_combined = torch.cat(gathered_labels).cpu().numpy()
        predictions = (all_preds_combined > 0.5).astype(int)
        
        # 计算总loss
        total_loss_tensor = torch.tensor(total_loss).to(rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        total_samples = len(all_labels_combined)
        avg_loss = total_loss_tensor.item() / total_samples
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels_combined, predictions),
            'f1': f1_score(all_labels_combined, predictions),
            'auc': roc_auc_score(all_labels_combined, all_preds_combined)
        }
        tn, fp, fn, tp = confusion_matrix(all_labels_combined, predictions).ravel()
        return metrics, [int(tn), int(fp), int(fn), int(tp)]
    return None, None

def main(rank, world_size):
    set_random_seed(223)
    setup(rank, world_size)
    model = BinaryTransformer()
    model_path = '/mnt/zhangrengang/model/best_model_train_start_022619.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    
    criterion = focal_loss(reduction="mean")
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4 * world_size, weight_decay=0.01)
    
    dataset_path = "/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc/"
    ds = time_series_dataset(dataset_path, is_train=True)
    train_size = int(0.9 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
    
    train_loader = get_distributed_loader(train_ds, 16, 32, collate_func, world_size, True, rank)
    val_loader = get_distributed_loader(test_ds, 8, 32, collate_func, world_size, False, rank)
    
    best_score = 0
    for epoch in range(1, 51):
        # 训练
        train_loss, acc, f1, auc, train_confusion = train_epoch(rank, ddp_model, train_loader, criterion, optimizer, epoch)
        # 验证
        val_metrics, val_confusion = evaluate(rank, ddp_model, val_loader, criterion)
        
        if rank == 0:
            print(f"Epoch {epoch}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            print(f"    Train Confusion: [tn, fp, fn, tp] {train_confusion}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} "
                  f"| F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
            print(f"    Val Confusion: [tn, fp, fn, tp] {val_confusion}")
            
            # 保存最佳模型
            if val_metrics['auc'] > best_score:
                best_score = val_metrics['auc']
                torch.save(ddp_model.module.state_dict(), model_path)
    
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
# from torch.utils.data import DataLoader
# import math
# import numpy as np
# from window_dataset import time_series_dataset, collate_func, MAX_SEQ_LEN, FEATURE_DIM
# from tqdm.auto import tqdm
# import warnings
# import os
# from lossfuncion.focal_loss import focal_loss
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
# import random

# # from torchvision.ops.focal_loss import sigmoid_focal_loss
# warnings.filterwarnings("ignore")

# # 固定随机种子
# def set_random_seed(seed: int) -> None:
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True

# # 改进的Transformer分类模型
# class BinaryTransformer(nn.Module):
#     def __init__(self, input_dim=FEATURE_DIM, d_model=128, nhead=8, 
#                  num_layers=4, max_seq_len=MAX_SEQ_LEN):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(d_model)
        
#         self.embedding = nn.Linear(input_dim, d_model)
#         self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
#             batch_first=True, dropout=0.0
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
#         # 改进的分类头
#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, 64),
#             nn.ReLU(),
#             nn.Dropout(0.0),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x, mask):
#         # x: (B, L, F), mask: (B, L)
#         seq_len = x.size(1)
        
#         # 嵌入层
#         x = self.embedding(x) * math.sqrt(self.embedding.out_features)
        
#         # 位置编码
#         positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
#         x = x + self.pos_encoder(positions)
                
#         # 正则化 
#         x = self.layer_norm(x)
        
#         # Transformer处理
#         x = self.transformer(x, src_key_padding_mask=~mask)

#         # 池化层（带mask的平均池化）
#         x = x * mask.unsqueeze(-1)
        
#         pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        
#         # 分类输出
#         return self.classifier(pooled).squeeze(-1)

# # 并行训练设置
# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
    
#     # 指定使用cuda:1-7中的设备
#     torch.cuda.set_device(rank)  # rank0对应cuda:0，以此类推
    
#     # 初始化进程组
#     dist.init_process_group(
#         backend='nccl',
#         rank=rank,
#         world_size=world_size
#     )

# # 清空并行训练设置
# def cleanup():
#     dist.destroy_process_group()

# # 并行训练Dataloader
# def get_distributed_loader(dataset, batch_size, num_workers, collate_func, world_size, shuffle, rank):
#     sampler = DistributedSampler(
#         dataset,
#         num_replicas=world_size,
#         rank=rank,
#         shuffle=shuffle
#     )
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         num_workers=num_workers,
#         shuffle=False,
#         collate_fn=collate_func,
#         pin_memory=True
#     )

# # 完整的训练流程
# def train_epoch(rank, model, loader, criterion, optimizer, epoch):
#     model.train()
#     sampler = loader.sampler
#     sampler.set_epoch(epoch)
    
#     total_loss = torch.tensor(0.0).to(rank)
#     loss_list = []
#     all_preds = []
#     all_labels = []
    
#     # 创建进度条
#     progress_bar = tqdm(
#         total=len(loader),
#         disable=(rank!=0),
#         desc=f"Training epoch {epoch}",
#         bar_format="{l_bar}{bar:20}{r_bar}",
#     )
    
#     for batch_idx, batch in enumerate(loader):
#         inputs = batch['features'].to(rank, non_blocking=True)
#         labels = batch['label'].float().to(rank, non_blocking=True)
#         masks = batch['mask'].to(rank, non_blocking=True)
        
#         optimizer.zero_grad()
#         outputs = model(inputs, masks)
        
#         loss = criterion(outputs, labels)
#         dist.all_reduce(loss, op=dist.ReduceOp.SUM)
#         total_loss += loss.item() / dist.get_world_size()
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
        
#         probs = torch.sigmoid(outputs).detach()
#         all_preds.extend(probs.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
        
#         # 实时更新进度条描述
#         if rank == 0:
#             avg_loss = total_loss / (batch_idx + 1)
#             progress_bar.set_postfix({
#                 "avg_loss": f"{avg_loss:.4f}",
#                 "loss": f"{loss_list[-1]:.4f}",
#                 "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
#             })
#             progress_bar.update(1)
    
#     all_preds = torch.cat(all_preds).numpy()
#     all_labels = torch.cat(all_labels).numpy()
    
#     if rank==0:
#         progress_bar.close()
#         # 计算指标
#         avg_loss = total_loss / len(loader)
#         predictions = (np.array(all_preds) > 0.5).astype(int)
#         acc = accuracy_score(all_labels, predictions)
#         f1 = f1_score(all_labels, predictions)
#         auc = roc_auc_score(all_labels, all_preds)
#         tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
#         return avg_loss, acc, f1, auc, [int(tn), int(fp), int(fn), int(tp)]
#     return None, None, None, None, None

# # 评估函数
# def evaluate(rank, model, loader, criterion):
#     model.eval()
#     total_loss = torch.tensor(0.0).to(rank)
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for batch in loader:
#             inputs = batch['features'].to(rank, non_blocking=True)
#             labels = batch['label'].float().to(rank, non_blocking=True)
#             masks = batch['mask'].to(rank, non_blocking=True)
            
#             outputs = model(inputs, masks)
#             loss = criterion(outputs, labels)
            
#             dist.all_reduce(loss, op=dist.ReduceOp.SUM)
#             total_loss += loss.item() / dist.get_world_size()
            
#             probs = torch.sigmoid(outputs)
#             all_preds.extend(probs.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    
#     all_preds = torch.cat(all_preds).numpy()
#     all_labels = torch.cat(all_labels).numpy()   
       
#     avg_loss = total_loss / len(loader)
#     predictions = (np.array(all_preds) > 0.5).astype(int)
    
#     if rank == 0:
#         metrics = {
#             'loss': avg_loss,
#             'accuracy': accuracy_score(all_labels, predictions),
#             'f1': f1_score(all_labels, predictions),
#             'auc': roc_auc_score(all_labels, all_preds)
#         }
#         tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
#         return metrics, [int(tn), int(fp), int(fn), int(tp)]
#     return None, None 

# # 推理函数
# def predict(model, samples, device):
#     """
#     支持批量推理
#     Args:
#         samples: list of dicts {'features': tensor(seq_len, 16), 'mask': tensor(seq_len)}
#     Returns:
#         probs: numpy array of probabilities
#     """
#     model.eval()
#     processed = collate_func(samples)  # 使用相同的collate函数
#     inputs = processed['features'].to(device)
#     masks = processed['mask'].to(device)
    
#     with torch.no_grad():
#         logits = model(inputs, masks)
#         probs = torch.sigmoid(logits).cpu().numpy()
#     return probs

# def main(rank, world_size):
#     set_random_seed(223)
#     setup(rank, world_size)
#     model = BinaryTransformer()
#     model_path = '/mnt/zhangrengang/model/best_model_train_start_022513.pth'
#     # 加载已有模型
#     if os.path.exists(model_path):
#         model.load_state_dict(torch.load(model_path))
#     model = model.to(rank)
    
#     ddp_model = DDP(
#         model,
#         device_ids=[rank],
#         output_device=rank
#     )
    
#     criterion = focal_loss(reduction = "mean")
#     optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4 * world_size, weight_decay=0.01)
#     dataset_path = "/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc/"
#     ds = time_series_dataset(dataset_path, is_train=True)
#     train_size = int(0.9 * len(ds))
#     test_size = len(ds) - train_size
#     train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

#     train_loader = get_distributed_loader(train_ds, batch_size=16, num_workers=32, collate_func=collate_func, world_size=world_size,shuffle=True,rank=rank)
#     val_loader = get_distributed_loader(test_ds, batch_size=8, num_workers=32, collate_func=collate_func, world_size=world_size,shuffle=True,rank=rank)

#     # 训练循环
#     best_score = 0
    
#     for epoch in range(1, 51):
#         train_loss, acc, f1, auc, train_confusion_mesg = train_epoch(rank, ddp_model, train_loader, criterion, optimizer, epoch)
#         val_metrics, val_confusion_mesg = evaluate(rank, ddp_model, val_loader, criterion)
        
#         print(f"Epoch {epoch}")
        
#         print(f"Train Loss: {train_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
#         print(f"    Train Confusion: [tn, fp, fn, tp] {train_confusion_mesg}")
        
#         print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} "
#               f"| F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
#         print(f"    Val Confusion: [tn, fp, fn, tp] {val_confusion_mesg}")
        
#         # 保存最佳模型
#         if val_metrics['auc'] > best_score:
#             best_score = val_metrics['auc']
#             torch.save(ddp_model.module.state_dict(), model_path)
            
#     cleanup()
    
    
# # 使用示例
# if __name__ == '__main__':
#     world_size = torch.cuda.device_count()
#     torch.multiprocessing.spawn(
#         main,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True
#     )