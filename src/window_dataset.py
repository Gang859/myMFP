import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
MAX_SEQ_LEN = 512
FEATURE_DIM = 48

def pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='front', value=0):
    
    mask = np.full((len(sequences), maxlen), 0)
    padded_sequences = np.full((len(sequences), maxlen, FEATURE_DIM), value)
    
    for i, seq in enumerate(sequences):
        if padding == 'post':
            mask[i, :len(seq)] = 1
            padded_sequences[i, :len(seq), :] = np.array(seq)[-maxlen:, :]
        else:
            mask[i, -len(seq):] = 1
            padded_sequences[i, -len(seq):, :] = np.array(seq)[-maxlen:, :]
        
    return padded_sequences, np.array(mask,  dtype=np.bool_)

# a =[ [[0,1],[1,1],[2,2]], 
#      [[0,1],[1,1]],
#      [[0,1],[1,1],[0,1],[1,1]],
#      [[0,1],[1,1],[0,1],[1,1],[0,1],[1,1],[0,1],[1,1]]]

# print(pad_sequences(a, 4))
# exit(0)


class time_series_dataset(Dataset):
    def __init__(self, data_root:str, is_train:bool=True):
        """
        :param data_root:   数据集路径
        """
        self.data_root = data_root
        file_prefix = []
        
        if is_train:
            file_list = os.listdir(data_root+"/"+"neg_windows_feature")
            for file in file_list:
                if '.json' in file:
                    file_prefix.append("/".join(file.split("/")[:-1])+"neg_windows_feature/" + file.split("/")[-1].split('.')[0])
                    
                    
            file_list = os.listdir(data_root+"/"+"pos_windows_feature")
            for file in file_list:
                if '.json' in file:
                    file_prefix.append("/".join(file.split("/")[:-1])+"pos_windows_feature/" + file.split("/")[-1].split('.')[0])
        else:
            file_list = os.listdir(data_root+"/"+"test_windows_feature")
            for file in file_list:
                if '.json' in file:
                    file_prefix.append("/".join(file.split("/")[:-1])+"test_windows_feature/" + file.split("/")[-1].split('.')[0])
                    
               
        file_prefix = list(set(file_prefix))
        self.data = file_prefix
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        prefix = self.data[index]
        import json
        with open(self.data_root+prefix+'.json','r',encoding='utf-8') as f:
            data_dict=json.load(f)
            
        feature, mask= pad_sequences(data_dict['features'])
        feature = torch.from_numpy(feature).float()
        label = torch.tensor(data_dict['label'], dtype=torch.long)
        mask = torch.from_numpy(mask).bool()
        sample = {'features': feature, 'label': label, 'id': prefix, "mask":mask}
        return sample
    
def collate_func(batch_dic):
    mask_batch = []
    fea_batch=[]
    label_batch=[]
    id_batch=[]
    for i in range(len(batch_dic)):
        dic = batch_dic[i]
        fea_batch.append(dic['features'])
        label_batch.append(dic['label'])
        id_batch.append(dic['id'])
        mask_batch.append(dic['mask'])
        
    res={}
    res['features'] = torch.tensor(np.concatenate(fea_batch,axis = 0))
    res['label'] = torch.tensor(np.concatenate(label_batch,axis = 0), dtype=torch.long)
    res['id'] = id_batch
    res['mask'] = torch.tensor(np.concatenate(mask_batch,axis = 0), dtype=torch.bool)
    return res

if __name__ == "__main__":
    counter = 0
    dataset = time_series_dataset("/mnt/zhangrengang/data/dump/")
    batch_size=1
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=64, shuffle=True,collate_fn=collate_func)
    for batch_idx, batch in enumerate(train_loader):
        inputs,labels,masks,ids=batch['features'],batch['label'],batch['mask'],batch['id']
        # (512, 128, 31) 
        # print(inputs.shape,labels.shape,masks.shape)