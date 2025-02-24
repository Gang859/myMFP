import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
import math
import numpy as np
from window_dataset import time_series_dataset, collate_func, MAX_SEQ_LEN
from tqdm.auto import tqdm
import warnings
import os
from lossfuncion.focal_loss import focal_loss
# from torchvision.ops.focal_loss import sigmoid_focal_loss
warnings.filterwarnings("ignore")

# 改进的Transformer分类模型
class BinaryTransformer(nn.Module):
    def __init__(self, input_dim=16, d_model=128, nhead=8, 
                 num_layers=4, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, 
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 改进的分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        # x: (B, L, F), mask: (B, L)
        seq_len = x.size(1)
        
        # 嵌入层
        x = self.embedding(x) * math.sqrt(self.embedding.out_features)
        
        # 位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_encoder(positions)
                
        # 正则化 
        x = self.layer_norm(x)
        
        # Transformer处理
        x = self.transformer(x, src_key_padding_mask=~mask)

        # 池化层（带mask的平均池化）
        x = x * mask.unsqueeze(-1)
        
        pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        
        # 分类输出
        return self.classifier(pooled).squeeze(-1)

# 完整的训练流程
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    loss_list = []
    all_preds = []
    all_labels = []
    
    # 创建进度条
    progress_bar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Training",
        bar_format="{l_bar}{bar:20}{r_bar}",
        dynamic_ncols=True
    )
    
    for batch_idx, batch in progress_bar:
        inputs = batch['features'].to(device)
        labels = batch['label'].float().to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, masks)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        loss_list.append(loss.item())
        probs = torch.sigmoid(outputs).detach()
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 实时更新进度条描述
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            "avg_loss": f"{avg_loss:.4f}",
            "loss": f"{loss_list[-1]:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    # 关闭进度条
    progress_bar.close()
    
    # 计算指标
    avg_loss = total_loss / len(loader)
    predictions = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    
    return avg_loss, acc, f1, auc, [tn, fp, fn, tp]

# 评估函数
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['features'].to(device)
            labels = batch['label'].float().to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    predictions = (np.array(all_preds) > 0.5).astype(int)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, predictions),
        'f1': f1_score(all_labels, predictions),
        'auc': roc_auc_score(all_labels, all_preds)
    }
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    return metrics, [tn, fp, fn, tp] 

# 推理函数
def predict(model, samples, device):
    """
    支持批量推理
    Args:
        samples: list of dicts {'features': tensor(seq_len, 16), 'mask': tensor(seq_len)}
    Returns:
        probs: numpy array of probabilities
    """
    model.eval()
    processed = collate_func(samples)  # 使用相同的collate函数
    inputs = processed['features'].to(device)
    masks = processed['mask'].to(device)
    
    with torch.no_grad():
        logits = model(inputs, masks)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

# 使用示例
if __name__ == '__main__':
    # 初始化组件
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    model = BinaryTransformer()
    
    model_path = '/mnt/zhangrengang/model/best_model.pth'
    # 加载已有模型
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    model = model.to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30])).to(device)
    criterion = focal_loss(reduction = "mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    dataset_path = "/mnt/zhangrengang/data/win30m_ds/"
    ds = time_series_dataset(dataset_path, is_train=True)
    # train_ds = time_series_dataset(dataset_path, is_train=True)
    # test_ds = time_series_dataset(dataset_path, is_train=False)
    train_size = int(0.9 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size], generator=torch.Generator().manual_seed(223))
    
    # 假设已经准备好数据集
    train_loader = DataLoader(train_ds, batch_size=16, num_workers=8,
                             shuffle=True, collate_fn=collate_func)
    val_loader = DataLoader(test_ds, batch_size=4, num_workers=8,
                            shuffle=True,collate_fn=collate_func)
    
    # 训练循环
    best_score = 0
    for epoch in range(1, 51):
        train_loss, acc, f1, auc, train_confusion_mesg = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_confusion_mesg = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}")
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print(f"    Train Confusion: [tn, fp, fn, tp] {train_confusion_mesg}")
        
        print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} "
              f"| F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
        print(f"    Val Confusion: [tn, fp, fn, tp] {val_confusion_mesg}")
        
        # 保存最佳模型
        if val_metrics['auc'] > best_score:
            best_score = val_metrics['auc']
            torch.save(model.state_dict(), model_path)
    
    # 推理示例
    # test_samples = [
    #     {'features': torch.randn(10, 16), 'mask': torch.ones(10).bool(), 'id': 'test1'},
    #     {'features': torch.randn(15, 16), 'mask': torch.ones(15).bool(), 'id': 'test2'}
    # ]
    # probabilities = predict(model, test_samples, device)
    # print(f"Prediction probabilities: {probabilities}")