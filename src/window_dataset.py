import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
def pad_sequences(sequences, maxlen=128, padding='front', value=0):
    
    mask = np.full((len(sequences), maxlen), 0)
    padded_sequences = np.full((len(sequences), maxlen, 15), value)
    
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
    def __init__(self, data_root):
        """
        :param data_root:   数据集路径
        """
        self.data_root = data_root
        file_prefix = []
        
        file_list = os.listdir(data_root+"/"+"negtive_feature_with_label")
        for file in file_list:
            if '.json' in file:
                file_prefix.append("/".join(file.split("/")[:-1])+"negtive_feature_with_label/" + file.split("/")[-1].split('.')[0])
                
                
        file_list = os.listdir(data_root+"/"+"positive_feature_with_label")
        for file in file_list:
            if '.json' in file:
                file_prefix.append("/".join(file.split("/")[:-1])+"positive_feature_with_label/" + file.split("/")[-1].split('.')[0])
                
        file_prefix = list(set(file_prefix))
        self.data = file_prefix
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        prefix = self.data[index]
        import json
        with open(self.data_root+prefix+'.json','r',encoding='utf-8') as f:
            data_dic=json.load(f)
            
        feature, mask= pad_sequences(data_dic['features'])
        length=len(data_dic['features'])
        feature = torch.from_numpy(feature)
        label = np.array(data_dic['lable'])
        label = torch.from_numpy(label)
        sample = {'features': feature, 'lable': label, 'id': prefix,'length':length, "mask":mask}
        return sample
    
def collate_func(batch_dic):
    batch_len=len(batch_dic)
    max_seq_length=128
    mask_batch = []
    fea_batch=[]
    label_batch=[]
    id_batch=[]
    for i in range(len(batch_dic)):
        dic = batch_dic[i]
        fea_batch.append(dic['features'])
        label_batch.append(dic['lable'])
        id_batch.append(dic['id'])
        mask_batch.append(dic['mask'])
        
    res={}
    res['features'] = np.concatenate(fea_batch,axis = 0)
    res['lable'] = np.concatenate(label_batch,axis = 0)
    res['id'] =  id_batch
    res['mask'] = np.concatenate(mask_batch,axis = 0)
    return res

if __name__ == "__main__":
    counter = 0
    dataset = time_series_dataset("/mnt/zhangrengang/workspace/myMFP/dump/")
    batch_size=5
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=True,collate_fn=collate_func)
    for batch_idx, batch in enumerate(train_loader):
        inputs,labels,masks,ids=batch['features'],batch['lable'],batch['mask'],batch['id']
        print(inputs.shape,labels.shape,masks.shape)
        counter+=1
    print(counter)