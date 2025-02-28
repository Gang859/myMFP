import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 最大的seq长度，对应一个窗口内保留的最大CE数量
MAX_SEQ_LEN = 512
# 特征维度，每个CE的特征数量，（对应token的长度）
FEATURE_DIM = 48

# 对窗口进行pad，截断/补齐到MAX_SEQ_LEN


def pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='front', value=0):

    mask = np.full((len(sequences), maxlen), 0)
    padded_sequences = np.full((len(sequences), maxlen, FEATURE_DIM), value)

    for i, seq in enumerate(sequences):
        # post padding
        if padding == 'post':
            mask[i, :len(seq)] = 1
            padded_sequences[i, :len(seq), :] = np.array(seq)[-maxlen:, :]
        # front padding
        else:
            mask[i, -len(seq):] = 1
            padded_sequences[i, -len(seq):, :] = np.array(seq)[-maxlen:, :]
    # 返回padding之后的seq，以及mask（有效位为1，padding位为0）
    return padded_sequences, np.array(mask,  dtype=np.bool_)

# a =[ [[0,1],[1,1],[2,2]],
#      [[0,1],[1,1]],
#      [[0,1],[1,1],[0,1],[1,1]],
#      [[0,1],[1,1],[0,1],[1,1],[0,1],[1,1],[0,1],[1,1]]]

# print(pad_sequences(a, 4))
# exit(0)

# 自定义数据集


class time_series_dataset(Dataset):
    def __init__(self, data_root: str, is_train: bool = True, is_aug_pos: bool = False):
        """
        :param data_root:   数据集路径
        :param is_train:    True-加载训练集， False-加载测试集
        :param is_aug_pos:  True-加载增强后的正样本，False-加载原始正样本 
        """
        self.data_root = data_root
        file_prefix = []
        self.neg_counter = 2000000000000
        self.pos_counter = 1000000000000
        self.test_counter = 6400000000
        
        if is_train:
            # 加载负样本窗口
            neg_file_list = os.listdir(data_root+"/"+"neg_windows_feature")
            neg_file_list.sort(key=lambda x: (int(x.split(".")[0].split("_")[1])))
            for file in neg_file_list:
                if '.json' in file:
                    file_prefix.append(
                        "/".join(file.split("/")[:-1])+"neg_windows_feature/" + file.split("/")[-1].split('.')[0])
                # TODO
                self.neg_counter -= 1
                if self.neg_counter <= 0:
                    break

            # fixed by wly
            pos_file_path = "pos_windows_feature"
            if is_aug_pos:
                pos_file_path = "aug_pos_windows_feature"
            pos_file_list = os.listdir(data_root + "/" + pos_file_path)
            pos_file_list.sort(key=lambda x: (int(x.split(".")[0].split("_")[1])))
            for file in pos_file_list:
                if '.json' in file:
                    file_prefix.append(
                        "/".join(file.split("/")[:-1])+pos_file_path+"/" + file.split("/")[-1].split('.')[0])
                # TODO
                self.pos_counter -= 1
                if self.pos_counter <= 0:
                    break
        else:
            test_file_list = os.listdir(data_root+"/"+"test_windows_feature")
            test_file_list.sort(key=lambda x: (int(x.split(".")[0].split("_")[1])))
            for file in test_file_list:
                if '.json' in file:
                    file_prefix.append(
                        "/".join(file.split("/")[:-1])+"test_windows_feature/" + file.split("/")[-1].split('.')[0])
                # TODO
                self.test_counter -= 1
                if self.test_counter <= 0:
                    break
        file_prefix = list(set(file_prefix))
        self.data = file_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prefix = self.data[index]
        import json
        with open(self.data_root+prefix+'.json', 'r', encoding='utf-8') as f:
            data_dict = json.load(f)

        feature, mask = pad_sequences(data_dict['features'])
        feature = torch.from_numpy(feature).float()
        label = torch.tensor(data_dict['label'], dtype=torch.long)
        mask = torch.from_numpy(mask).bool()
        ids = data_dict["sn_name"]
        log_times = data_dict["Window_LogTime"]
        win_level_features = np.array(data_dict["window_features"])
        sample = {'features': feature, 'label': label, 'id': ids, "mask": mask,
                  "logtime": log_times, "win_level_features": win_level_features}
        return sample


def collate_func(batch_dic):
    mask_batch = []
    fea_batch = []
    label_batch = []
    id_batch = []
    logtime_batch = []
    win_level_feature_batch = []
    for i in range(len(batch_dic)):
        dic = batch_dic[i]
        fea_batch.append(dic['features'])
        label_batch.append(dic['label'])
        id_batch += dic['id']
        logtime_batch += dic['logtime']
        mask_batch.append(dic['mask'])
        win_level_feature_batch.append(dic['win_level_features'])

    res = {}
    res['features'] = torch.tensor(np.concatenate(fea_batch, axis=0))
    res['label'] = torch.tensor(np.concatenate(
        label_batch, axis=0), dtype=torch.long)
    res['id'] = id_batch
    res['logtime'] = logtime_batch
    res['mask'] = torch.tensor(np.concatenate(
        mask_batch, axis=0), dtype=torch.bool)
    res['win_level_features'] = torch.tensor(np.concatenate(
        win_level_feature_batch, axis=0), dtype=torch.float32)
    return res


if __name__ == "__main__":
    dataset = time_series_dataset(
        "/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc/", is_aug_pos=True, is_train = True)
    train_loader = DataLoader(
        dataset, batch_size=2, num_workers=16, shuffle=True, collate_fn=collate_func)
    for batch_idx, batch in enumerate(train_loader):
        inputs, labels, masks, ids, logtime, win_level_features = batch['features'], batch[
            'label'], batch['mask'], batch['id'], batch['logtime'], batch['win_level_features']
        print(inputs.shape, labels.shape, masks.shape, win_level_features.shape)
        print(win_level_features[0])
        exit(0)
