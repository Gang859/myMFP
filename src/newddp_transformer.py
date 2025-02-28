import datetime
import time
from progress.bar import Bar

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
from utils.logger import Logger
from lossfuncion.focal_loss import focal_loss
# from torchvision.ops.focal_loss import sigmoid_focal_loss
warnings.filterwarnings("ignore")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


# 改进的Transformer分类模型
class BinaryTransformer(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, d_model=128, nhead=8,
                 num_layers=4, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

        self.win_level_features_dim = 99
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        
        self.win_features_processor = nn.Sequential(
            nn.Linear(self.win_level_features_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.0
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 改进的分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask, win_level_features):
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

        win_macro_embedding = self.win_features_processor(win_level_features)

        combined = torch.cat([pooled, win_macro_embedding], dim=1)

        # 分类输出
        return self.classifier(combined).squeeze(-1)

class BinaryTransformer2(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, d_model=256, nhead=8,
                 num_layers=6, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

        self.win_level_features_dim = 99
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        
        self.win_features_processor = nn.Sequential(
            nn.Linear(self.win_level_features_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.2
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 改进的分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask, win_level_features):
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

        win_macro_embedding = self.win_features_processor(win_level_features)

        combined = torch.cat([pooled, win_macro_embedding], dim=1)

        # 分类输出
        return self.classifier(combined).squeeze(-1)


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

    return avg_loss, acc, f1, auc, [int(tn), int(fp), int(fn), int(tp)]

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
    return metrics, [int(tn), int(fp), int(fn), int(tp)]

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


def get_parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Pytorch training script of Ts.')
    parser.add_argument('task', default='test', choices=['zrg', "test"],
                        help='test')
    parser.add_argument('--exp_id', default='default')
    parser.add_argument('--gpus', default='-1',
                        help='-1 for CPU, use comma for multiple gpus')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='dataloader threads. 0 for single-thread.')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='num_epochs.')
    parser.add_argument('--save_intervals', type=int, default=5,
                        help='save_intervals.')
    parser.add_argument('--gpus_offset', type=int, default=0,
                        help='gpus_offset.')
    parser.add_argument('--weight_decay', type=float, default=0.00,
                        help='weight_decay.')
    parser.add_argument('--load_model', default='',
                        help='path to pretrained model')
    parser.add_argument('--num_iters', type=int, default=-1,
                        help='default: #samples / batch_size.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--model_type', type=int, default=1,
                        help='1, 2')
    parser.add_argument('--sn_type', default='A', choices=['A', 'B'],
                        help='sn_type')
    parser.add_argument('--metric', default='loss',
                        help='metric')
    parser.add_argument('--find_opt_threadhold', action = 'store_true',
                        help='find_opt_threadhold')
    args = parser.parse_args()
    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.gpus = args.gpus if args.gpus[0] >= 0 else [-1]
    args.root_dir = os.path.join(os.path.dirname(__file__), '..')
    args.exp_dir = os.path.join(args.root_dir, 'exp', args.task)
    args.save_dir = os.path.join(args.exp_dir, args.exp_id)
    args.debug_dir = os.path.join(args.save_dir, 'debug')
    args.random_seed = 3407
    args.local_rank = 0
    return args


class Trainer(object):
    def __init__(self, args, model, optimizer):
        self.args = args
        self.optimizer = optimizer
        print(args.device)
        self.model_with_loss = ModelWithLoss(model, args.gpus, args.device)

    def set_device(self, device, local_rank, gpus):
        if len(gpus) > 1:
            self.model_with_loss = self.model_with_loss.to(device)
            self.model_with_loss = nn.parallel.DistributedDataParallel(self.model_with_loss,
                                                                       device_ids=[
                                                                           local_rank],
                                                                       find_unused_parameters=False)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, dataset, local_rank):
        model_with_loss = self.model_with_loss
        # --------------------------- phase setting
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.args.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()
        # --------------------------- init
        data_time, batch_time = AverageMeter(), AverageMeter()
        loss_stats = {'loss': 0}
        avg_loss_stats = {"loss": AverageMeter()}
        num_iters = len(dataset) if args.num_iters < 0 else args.num_iters
        all_preds = []
        all_labels = []
        tn_list = []
        fp_list = []
        fn_list = []
        tp_list = []
        auc_list = []
        f1_list = []
        acc_list = []
        if local_rank == 0:
            bar = Bar('\033[1;31;42m{}/{}'.format(args.task,
                      args.exp_id), max=num_iters)
        end = time.time()
        # --------------------------- iter
        for iter_id, batch in enumerate(dataset):
            if iter_id >= num_iters:
                break
            if len(self.args.gpus) == 1:
                batch = batch.to(self.args.device)
            data_time.update(time.time() - end)
            # --------------------------- loss calculate
            outputs, labels, loss_stats = model_with_loss(batch)
            loss = loss_stats["loss"]
            probs = torch.sigmoid(outputs).detach()
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # --------------------------- gradient calculate
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model_with_loss.parameters(), 1.0)
                self.optimizer.step()
            # --------------------------- log print
            batch_time.update(time.time() - end)
            end = time.time()
            if local_rank == 0:
                Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                    epoch, iter_id, num_iters, phase=phase,
                    total=bar.elapsed_td, eta=bar.eta_td)
                for l in avg_loss_stats:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['features'].shape[0])
                    Bar.suffix = Bar.suffix + \
                        '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
                predictions = (np.array(all_preds) > 0.5).astype(int)
                acc = accuracy_score(all_labels, predictions)
                f1 = f1_score(all_labels, predictions)
                auc = roc_auc_score(all_labels, all_preds)
                tn, fp, fn, tp = confusion_matrix(all_labels, predictions, labels=[0, 1]).ravel()
                tn_list.append(tn)
                fp_list.append(fp)
                fn_list.append(fn)
                tp_list.append(tp)
                auc_list.append(auc)
                acc_list.append(acc)
                f1_list.append(f1)
                Bar.suffix = Bar.suffix + \
                    '\033[1;31;42m |acc {:}\033[0m'.format(round(acc,4))
                Bar.suffix = Bar.suffix + \
                    '\033[1;31;42m |f1 {:}\033[0m'.format(round(f1,4))
                Bar.suffix = Bar.suffix + \
                    '\033[1;31;42m |auc {:}\033[0m'.format(round(auc,4))
                Bar.suffix = Bar.suffix + \
                    '\033[1;31;42m |tn, fp, fn, tp {tn} {fp} {fn} {tp}\033[0m'.format(
                        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
                bar.next()
            del outputs, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        if local_rank == 0:
            bar.finish()
            ret['time'] = bar.elapsed_td.total_seconds() / 60.
            ret["tn"] = int(tn_list[-1])
            ret["fp"] = int(fp_list[-1])
            ret["fn"] = int(fn_list[-1])
            ret["tp"] = int(tp_list[-1])
            if np.isnan(auc_list[-1]):
                ret["auc"] = -1.00
            else:
                ret["auc"] = round(auc_list[-1],4)
            ret["acc"] = round(acc_list[-1],4)
            ret["f1"] = round(f1_list[-1],4)

        return ret

    def val(self, epoch, data_loader, local_rank):
        return self.run_epoch('  val', epoch, data_loader, local_rank)

    def train(self, epoch, data_loader, local_rank):
        return self.run_epoch('train', epoch, data_loader, local_rank)

    def val_opt_threadhold(self, epoch, data_loader, local_rank):
        return self.val_for_find_optimal_threadhold_epoch(epoch, data_loader, local_rank)
    
    def val_for_find_optimal_threadhold_epoch(self, epoch, dataset, local_rank):
        model_with_loss = self.model_with_loss
        phase = 'val_opt_threadhold'
        # --------------------------- phase setting
        if len(self.args.gpus) > 1:
            model_with_loss = self.model_with_loss.module
        model_with_loss.eval()
        torch.cuda.empty_cache()
        # --------------------------- init
        data_time, batch_time = AverageMeter(), AverageMeter()
        loss_stats = {'loss': 0}
        avg_loss_stats = {"loss": AverageMeter()}
        num_iters = len(dataset) if args.num_iters < 0 else args.num_iters
        all_probs = []
        all_labels = []
        if local_rank == 0:
            bar = Bar('\033[1;31;42m{}/{}'.format(args.task,
                      args.exp_id), max=num_iters)
        end = time.time()
        # --------------------------- iter
        for iter_id, batch in enumerate(dataset):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            # --------------------------- loss calculate
            outputs, labels, loss_stats = model_with_loss(batch)
            loss = loss_stats["loss"]
            probs = torch.sigmoid(outputs).detach()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # --------------------------- gradient calculate
            # --------------------------- log print
            batch_time.update(time.time() - end)
            end = time.time()
            if local_rank == 0:
                Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                    epoch, iter_id, num_iters, phase=phase,
                    total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()
            del outputs, loss, loss_stats
        ret = {"best_optimal_threshold" : self.find_optimal_threshold(all_labels, all_probs)}
        return ret
            
    def find_optimal_threshold(self, y_true, y_pred):
        thresholds = np.linspace(0.1, 0.9, 50)
        best_f1 = 0
        best_thresh = 0.5
        for thresh in thresholds:
            preds = (y_pred > thresh).astype(int)
            f1 = f1_score(y_true, preds)
            acc = accuracy_score(y_true, preds)
            auc = roc_auc_score(y_true, preds)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            print(f"thresh : {thresh}, F1:{f1}, acc: {acc}, auc: {auc}, [tn, fp, fn, tp] : {tn} {fp} {fn} {tp}")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        print(f"best_thresh : {best_thresh}, F1 : {best_f1}")
        return best_thresh


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, gpus, device):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.gpus = gpus
        self.device = device
        self.criterion = focal_loss(reduction="mean", gamma = 3, alpha = 0.8)

    def forward(self, batch):
        inputs = batch['features'].to(self.device)
        labels = batch['label'].float().to(self.device)
        masks = batch['mask'].to(self.device)
        win_level_features = batch['win_level_features'].to(self.device)

        outputs = self.model(inputs, masks, win_level_features)
        loss = self.criterion(outputs, labels)
        loss_stats = {'loss': loss}
        return outputs, labels, loss_stats

def set_random_seed(seed: int) -> None:
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

def load_model(model, model_path, optimizer=None, local_rank = 0, device='cuda'):
    start_epoch = 0

    if device == 'cuda':
        map  = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(
            model_path, map_location=map)
    else:
        checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    
    if local_rank == 0:
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                if local_rank == 0:
                 print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                          k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            if local_rank == 0:
                print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            if local_rank == 0:
                print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]      
    model.load_state_dict(state_dict, strict=False)
    
   # resume optimizer parameters
    if optimizer is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        else:
            if local_rank == 0:
                print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model

# 使用示例
def main(args):
    #################
    # Device
    #################
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cuda:%d' % (
        args.gpus_offset) if args.gpus[0] >= 0 else 'cpu')
    args.world_size = 1
    args.rank = 0  # global rank
    if args.device != 'cpu' and len(args.gpus) > 1:
        args.distributed = len(args.gpus)
    else:
        args.distributed = False

    if args.distributed:
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.getenv('LOCAL_RANK'))
        args.device = 'cuda:%d' % (args.local_rank+args.gpus_offset)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=21600))
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('\033[32m====> Training in distributed mode. Device {}, Process {:}, total {:}.\033[0m'.format(
            args.device, args.rank, args.world_size))
    else:
        print('\033[32m====> Training in single device: \033[0m', args.device)

    logger = Logger(args, args.local_rank)

    #################
    # Dataset
    #################
    dataset_path = f"/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc_type{args.sn_type}/"
    if args.local_rank == 0:
        print('\033[32m====> Using settings {}\033[0m'.format(args))
        print('==> Loading dataset from: ', dataset_path)
    ds = time_series_dataset(dataset_path, is_train=True)
    train_size = int((0.9 * len(ds)))
    val_size = int((len(ds) - train_size))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size], generator=torch.Generator().manual_seed(223))
    if args.local_rank == 0:
        print("Size: ", len(ds))
        print('Splitting the dataset into training and validation sets..')
        print('# training circuits: ', train_size)
        print('# validation circuits: ', val_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds,
                                                                    num_replicas=args.world_size,
                                                                    rank=args.rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds,
                                                                  num_replicas=args.world_size,
                                                                  rank=args.rank)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=False, collate_fn=collate_func, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size//2, num_workers=args.num_workers,
                            shuffle=False, collate_fn=collate_func, sampler=val_sampler)

    #################
    # Model
    #################
    if args.model_type == 1:
        model = BinaryTransformer()
    elif args.model_type == 2:
        model = BinaryTransformer2()

    if args.local_rank == 0:
        print('==> Creating model...')
        print(model)
        
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    
    start_epoch = 0
    if args.load_model != '':
        model, optimizer, start_epoch = load_model(model, args.load_model, optimizer, args.local_rank, args.device)
        mesg = "\033[32m====> Load model from : "+args.load_model+" ...\033[0m"
        print(mesg)

    trainer = Trainer(args, model, optimizer)
    trainer.set_device(args.device, args.local_rank, args.gpus)

    if args.local_rank == 0:
        print('==> Starting training...')
    
    if args.find_opt_threadhold:
        with torch.no_grad():
            val_loader.sampler.set_epoch(0)
            log_dict_val = trainer.val_opt_threadhold(0, val_loader, args.local_rank)
    else:
        best = 1e10
        for epoch in range(start_epoch + 1, args.num_epochs + 1):
            mark = 'last'
            train_loader.sampler.set_epoch(epoch)
            log_dict_train = trainer.train(epoch, train_loader, args.local_rank)
            if args.local_rank == 0:
                int_name_list = ["tn", "fp", "fn", "tp"]
                logger.write('epoch: {} |'.format(epoch), args.local_rank)
                logger.write("\n", args.local_rank)
                logger.write("Train:|", args.local_rank)
                for k, v in log_dict_train.items():
                    logger.scalar_summary('train_{}'.format(k), v, epoch, args.local_rank)
                    if k not in int_name_list:
                        logger.write('{} {:4f} | '.format(k, round(v,4)), args.local_rank)
                    else:
                        logger.write('{} {:} | '.format(k, v), args.local_rank)
                if args.save_intervals > 0 and epoch % args.save_intervals == 0:
                    save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)
            with torch.no_grad():
                val_loader.sampler.set_epoch(0)
                log_dict_val = trainer.val(epoch, val_loader, args.local_rank)

            if args.local_rank == 0:
                logger.write("\n", args.local_rank)
                logger.write("Eval: |", args.local_rank)
                for k, v in log_dict_val.items():
                    logger.scalar_summary('val_{}'.format(k), v, epoch, args.local_rank)
                    if k not in int_name_list:
                        logger.write('{} {:4f} | '.format(k, round(v,4)), args.local_rank)
                    else:
                        logger.write('{} {:} | '.format(k, v), args.local_rank)
                if log_dict_val[args.metric] < best:
                    best = log_dict_val[args.metric]
                    save_model(os.path.join(args.save_dir, 'model_best.pth'), epoch, model, optimizer)
                else:
                    save_model(os.path.join(args.save_dir, 'model_last.pth'), epoch, model, optimizer)
                logger.write('\n', args.local_rank)
        if args.local_rank == 0:
            logger.close()
    destroy_process_group()


def set_seed(args):
    import random
    import torch.backends.cudnn as cudnn
    # fix randomseed for reproducing the results
    print('Setting random seed for reproductivity..')
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = get_parse_args()
    set_seed(args)
    main(args)
