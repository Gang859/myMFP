import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 最大的seq长度，对应一个窗口内保留的最大sub window数量
MAX_SEQ_LEN = 73
# 特征维度，每个sub window的特征数量，（对应token的长度）
FEATURE_DIM = 66

# 对窗口进行pad，截断/补齐到MAX_SEQ_LEN


def pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='front', value=0):

    mask = np.full((len(sequences), maxlen), 0)
    padded_sequences = np.full((len(sequences), maxlen, FEATURE_DIM+1), value)

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
    def __init__(self, data_root: str, is_train: bool = True, is_aug_pos: bool = False, is_aug_test:bool = False):
        """
        :param data_root:   数据集路径
        :param is_train:    True-加载训练集， False-加载测试集
        :param is_aug_pos:  True-加载增强后的正样本，False-加载原始正样本 
        """
        self.data_root = data_root
        file_prefix = []
        self.neg_counter = 4000000000000
        self.pos_counter = 100000000000
        self.test_counter = 6400000000000
        
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
            test_file_path = "test_windows_feature"
            if is_aug_test:
                test_file_path = "aug_test_windows_feature"
            test_file_list = os.listdir(data_root+"/"+test_file_path)
            test_file_list.sort(key=lambda x: (int(x.split(".")[0].split("_")[1])))
            for file in test_file_list:
                if '.json' in file:
                    file_prefix.append(
                        "/".join(file.split("/")[:-1]) + test_file_path + "/" + file.split("/")[-1].split('.')[0])
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
        lens = data_dict["lens"]
        ids = data_dict["sn_name"]
        label = torch.tensor(data_dict['label'], dtype=torch.long)
        
        # ------------------------------- feature process begin----------------------------
        last_ce_logtime = [win_tokens[-1][0] for idx, win_tokens in enumerate(data_dict['features'])]
        win_end_time = data_dict["win_end_time"]
        feature, mask = pad_sequences(data_dict['features'])
        sub_win_relative_end_time = np.array([win_end_time[idx] - feature[idx,:,-1] for idx, win_tokens in enumerate(feature)])
        for idx, arr in enumerate(sub_win_relative_end_time):
            sub_win_relative_end_time[idx][:len(arr) - lens[idx]] = 0
        feature[:,:,0] = sub_win_relative_end_time
        # ------------------------------- feature process endin----------------------------
        
        feature = torch.from_numpy(feature).float()
        mask = torch.from_numpy(mask).bool()
        sample = {'features': feature, 'label': label, 'id': ids, "mask": mask,
                  "last_ce_logtime": last_ce_logtime, "win_end_time": win_end_time, "lens":lens}
        return sample

def collate_func(batch_dic):
    mask_batch = []
    fea_batch = []
    label_batch = []
    id_batch = []
    sn_idx_batch = []
    last_ce_logtime_batch = []
    win_end_time_batch = []
    lens_batch = []
    for i in range(len(batch_dic)):
        dic = batch_dic[i]
        fea_batch.append(dic['features'])
        label_batch.append(dic['label'])
        id_batch += dic['id']
        last_ce_logtime_batch += dic['last_ce_logtime']
        win_end_time_batch += dic['win_end_time']
        mask_batch.append(dic['mask'])
        lens_batch += dic['lens']
        for j in range(len(dic['id'])):
            sn_idx_batch.append([int(dic['id'][j].split("_")[-1])])

    res = {}
    res['features'] = torch.tensor(np.concatenate(fea_batch, axis=0))[:,:,:-1]
    res['label'] = torch.tensor(np.concatenate(label_batch, axis=0), dtype=torch.long)
    res['id'] = id_batch
    res['sn_idx'] = torch.tensor(sn_idx_batch, dtype=torch.float32)
    res['mask'] = torch.tensor(np.concatenate(mask_batch, axis=0), dtype=torch.bool)
    res['last_ce_logtime'] = last_ce_logtime_batch
    res['win_end_time'] = win_end_time_batch
    res['lens'] = lens_batch
    return res

class time_series_dataset2(Dataset):
    def __init__(self, data_root: str, is_train: bool = True, is_aug_pos: bool = False ,is_aug_test = False) :
        """
        :param data_root:   数据集路径
        :param is_train:    True-加载训练集， False-加载测试集
        :param is_aug_pos:  True-加载增强后的正样本，False-加载原始正样本 
        """
        self.data_root = data_root
        file_prefix = []
        self.neg_counter = 4000000000000
        self.pos_counter = 100000000000
        self.test_counter = 6400000000000
        
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
            test_file_path = "test_windows_feature"
            if is_aug_test:
                test_file_path = "aug_test_windows_feature"
            test_file_list = os.listdir(data_root+"/"+test_file_path)
            test_file_list.sort(key=lambda x: (int(x.split(".")[0].split("_")[1])))
            for file in test_file_list:
                if '.json' in file:
                    file_prefix.append(
                        "/".join(file.split("/")[:-1])+test_file_path + "/" + file.split("/")[-1].split('.')[0])
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
        lens = data_dict["lens"]
        ids = data_dict["sn_name"]
        label = torch.tensor(data_dict['label'], dtype=torch.long)
        last_15m_win_feature =  torch.tensor(data_dict['last_15m_win_feature'], dtype=torch.float32)
        last_30m_win_feature =  torch.tensor(data_dict['last_30m_win_feature'], dtype=torch.float32)

        # ------------------------------- feature process begin----------------------------
        last_ce_logtime = [win_tokens[-1][0] for idx, win_tokens in enumerate(data_dict['features'])]
        win_end_time = data_dict["win_end_time"]
        feature, mask = pad_sequences(data_dict['features'])
        sub_win_relative_end_time = np.array([win_end_time[idx] - feature[idx,:,-1] for idx, win_tokens in enumerate(feature)])
        for idx, arr in enumerate(sub_win_relative_end_time):
            sub_win_relative_end_time[idx][:len(arr) - lens[idx]] = 0
        feature[:,:,0] = sub_win_relative_end_time
        # ------------------------------- feature process endin----------------------------
        
        feature = torch.from_numpy(feature).float()
        mask = torch.from_numpy(mask).bool()
        sample = {'features': feature, 'label': label, 'id': ids, "mask": mask,
                  "last_ce_logtime": last_ce_logtime, "win_end_time": win_end_time, "lens":lens, "last_30m_win_feature" :last_30m_win_feature, "last_15m_win_feature":last_15m_win_feature}
        return sample

def collate_func2(batch_dic):
    mask_batch = []
    fea_batch = []
    label_batch = []
    id_batch = []
    last_ce_logtime_batch = []
    win_end_time_batch = []
    lens_batch = []
    window_feature_batch = []

    for i in range(len(batch_dic)):
        dic = batch_dic[i]
        id_batch += dic['id']
        last_ce_logtime_batch += dic['last_ce_logtime']
        win_end_time_batch += dic['win_end_time']
        lens_batch += dic['lens']
        fea_batch.append(dic['features'])
        label_batch.append(dic['label'])
        mask_batch.append(dic['mask'])
        window_feature_batch.append(np.concatenate((dic['last_15m_win_feature'], dic['last_30m_win_feature']), axis = 1))
    res = {}
    res['features'] = torch.tensor(np.concatenate(fea_batch, axis=0))[:,:,:-1]
    res["window_features"] = torch.tensor(np.concatenate(window_feature_batch, axis=0), dtype=torch.float32)
    res['label'] = torch.tensor(np.concatenate(label_batch, axis=0), dtype=torch.long)
    res['id'] = id_batch
    res['mask'] = torch.tensor(np.concatenate(mask_batch, axis=0), dtype=torch.bool)
    res['last_ce_logtime'] = last_ce_logtime_batch
    res['win_end_time'] = win_end_time_batch
    res['lens'] = lens_batch
    return res
            
def find_optimal_threshold(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
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

def find_optimal_threshold_paralle(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    import numpy as np
    from joblib import Parallel, delayed

    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1 = 0
    best_thresh = 0.5

    # 将每个阈值的计算封装成函数，便于并行
    def compute_metrics(thresh, y_true, y_pred):
        preds = (y_pred > thresh).astype(int)
        f1 = f1_score(y_true, preds)
        acc = accuracy_score(y_true, preds)
        auc = roc_auc_score(y_true, preds)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        print(f"thresh : {thresh}, F1:{f1}, acc: {acc}, auc: {auc}, [tn, fp, fn, tp] : {tn} {fp} {fn} {tp}")
        return (thresh, f1, acc, auc, tn, fp, fn, tp)

    # 并行计算所有阈值
    results = Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(compute_metrics)(thresh, y_true, y_pred) 
        for thresh in thresholds
    )

    # 遍历结果并找到最佳阈值
    for result in results:
        thresh, f1, acc, auc, tn, fp, fn, tp = result
        print(f"thresh: {thresh:.2f}, F1: {f1:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}, [TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}]")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nbest_thresh: {best_thresh:.4f}, F1: {best_f1:.4f}")
    return best_thresh

def LGBM_train():
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    import joblib

    ds = time_series_dataset("/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc_typeA/", is_train=True)
    train_size = int((0.9 * len(ds)))
    val_size = int((len(ds) - train_size))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size], generator=torch.Generator().manual_seed(223))

    train_loader = DataLoader(train_ds, batch_size=32, num_workers=72, shuffle=True, collate_fn=collate_func)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=16, shuffle=False, collate_fn=collate_func)

    all_train_features = []
    all_train_label = []

    all_eval_features = []
    all_eval_label = []

    for batch in tqdm(train_loader, desc="Processing train_datset"):
        all_train_features.extend(batch['win_level_features'])
        all_train_label.extend(batch['label'])

    for batch in tqdm(val_loader, desc="Processing eval_dataset"):
        all_eval_features.extend(batch['win_level_features'])
        all_eval_label.extend(batch['label'])

    model_params = {
            "learning_rate": 0.02,
            "n_estimators": 500,
            "max_depth": 8,
            "num_leaves": 20,
            "min_child_samples": 20,
            "verbose": 1,
        }
    model = LGBMClassifier(**model_params)
    model.fit(all_train_features, all_train_label)

    all_eval_pred_prob = model.predict_proba(all_eval_features)
    all_eval_pred_result = [1 if probs[0] < probs[1] else 0 for probs in all_eval_pred_prob]
    all_eval_pred_prob = [probs[1] for probs in all_eval_pred_prob]

    find_optimal_threshold_paralle(all_eval_label,all_eval_pred_prob)
    # acc = accuracy_score(all_eval_label, all_eval_pred_result)
    # f1 = f1_score(all_eval_label, all_eval_pred_result)
    # auc = roc_auc_score(all_eval_label, all_eval_pred_prob[:,1])
    # tn, fp, fn, tp = confusion_matrix(all_eval_label, all_eval_pred_result, labels=[0, 1]).ravel()
    # print(f"F1:{f1}, acc: {acc}, auc: {auc}, [tn, fp, fn, tp] : {tn} {fp} {fn} {tp}")
    joblib.dump(model, '/mnt/zhangrengang/workspace/myMFP/model_pth/LGBMClassifier_model_opt4threshold.pkl')
    
def LGBM_infer(model_path):
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    import joblib
    import pandas as pd
    model = joblib.load('/mnt/zhangrengang/workspace/myMFP/model_pth/LGBMClassifier_model.pkl')
    dataset_path = "/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc_typeA/"
    ds = time_series_dataset(dataset_path, is_train=False)
    test_loader = DataLoader(ds, batch_size=64, num_workers=72, shuffle=False, collate_fn=collate_func)
    all_test_sn_name = []
    all_test_logtime = []
    for batch in tqdm(test_loader, desc="Processing test_datset"):
        all_test_pred_prob = model.predict_proba(batch['win_level_features'])
        for i, prob_pair in enumerate(all_test_pred_prob):
            if prob_pair[0] < prob_pair[1]:
                all_test_sn_name.append(batch['id'][i])
                all_test_logtime.append(batch['logtime'][i])

    pred_df = pd.DataFrame(
        {'sn_name': all_test_sn_name, 'prediction_timestamp': all_test_logtime})
    pred_df = pred_df.assign(serial_number_type="A")
    pred_df.to_csv("/mnt/zhangrengang/workspace/myMFP/model_pth/submission.csv", index=None)

def Xgboost_train():
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    import joblib, pickle
    ds = time_series_dataset("/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc_typeA/", is_train=True)
    train_size = int((0.9 * len(ds)))
    val_size = int((len(ds) - train_size))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size], generator=torch.Generator().manual_seed(223))

    train_loader = DataLoader(train_ds, batch_size=32, num_workers=72, shuffle=True, collate_fn=collate_func)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=16, shuffle=False, collate_fn=collate_func)

    all_train_features = []
    all_train_label = []

    all_eval_features = []
    all_eval_label = []

    for batch in tqdm(train_loader, desc="Processing train_datset"):
        all_train_features.extend(batch['win_level_features'])
        all_train_label.extend(batch['label'])

    for batch in tqdm(val_loader, desc="Processing eval_dataset"):
        all_eval_features.extend(batch['win_level_features'])
        all_eval_label.extend(batch['label'])

    model_params = {
            "learning_rate": 0.02,
            "silent": 1,
            "nthread": -1,
            "scale_pos_weight": 10.0,
            "n_estimators": 500,
            "max_depth": 8,
            "num_class " : 2,
        }
    model = xgb.XGBClassifier()
    model.fit(all_train_features, all_train_label)

    all_eval_pred_prob = model.predict_proba(all_eval_features)
    all_eval_pred_prob = np.array(all_eval_pred_prob)
    all_eval_pred_result = [1 if probs[0] < probs[1] else 0 for probs in all_eval_pred_prob]

    # find_optimal_threshold_paralle(all_eval_label,all_eval_pred_prob)
    acc = accuracy_score(all_eval_label, all_eval_pred_result)
    f1 = f1_score(all_eval_label, all_eval_pred_result)
    auc = roc_auc_score(all_eval_label, all_eval_pred_prob[:,1])
    tn, fp, fn, tp = confusion_matrix(all_eval_label, all_eval_pred_result, labels=[0, 1]).ravel()
    print(f"F1:{f1}, acc: {acc}, auc: {auc}, [tn, fp, fn, tp] : {tn} {fp} {fn} {tp}")
    with open (f"/mnt/zhangrengang/workspace/myMFP/model_pth/XBGClassifier_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
def Xgboost_infer(model_path):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    import joblib, pickle
    import pandas as pd

    with open (model_path, "rb") as f:
        model = pickle.load(f)
    dataset_path = "/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc_typeA/"
    ds = time_series_dataset(dataset_path, is_train=False)
    test_loader = DataLoader(ds, batch_size=64, num_workers=72, shuffle=False, collate_fn=collate_func)
    all_test_sn_name = []
    all_test_logtime = []
    for batch in tqdm(test_loader, desc="Processing test_datset"):
        all_test_pred_prob = model.predict_proba(batch['win_level_features'])
        for i, prob_pair in enumerate(all_test_pred_prob):
            if prob_pair[0] < prob_pair[1]:
                all_test_sn_name.append(batch['id'][i])
                all_test_logtime.append(batch['logtime'][i])

    pred_df = pd.DataFrame(
        {'sn_name': all_test_sn_name, 'prediction_timestamp': all_test_logtime})
    pred_df = pred_df.assign(serial_number_type="A")
    pred_df.to_csv("/mnt/zhangrengang/workspace/myMFP/model_pth/submission_xgb.csv", index=None)

def test():
    ds = time_series_dataset("/backup/home/zhangrengang/workspace/STIM_Data1/STIM_win_feature_A/", is_train=True)
    train_size = int((0.9 * len(ds)))
    val_size = int((len(ds) - train_size))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size], generator=torch.Generator().manual_seed(223))

    train_loader = DataLoader(train_ds, batch_size=16, num_workers=72, shuffle=True, collate_fn=collate_func)
    val_loader = DataLoader(val_ds, batch_size=16, num_workers=16, shuffle=False, collate_fn=collate_func)
    counter = 0
    pos_counter = 0
    neg_counter = 0
    
    for batch in tqdm(train_loader, desc="Processing train_datset"):
        if counter == 0:
            print(batch['features'][-1][0])
            print(batch['mask'][0])
            print(batch['features'][0][:][:,0]//3600)
            print(batch['label'][0])
            print(batch['id'][0])
            print(batch['last_ce_logtime'][0])
            print(batch['win_end_time'][0])
            print(batch['lens'][0])
            exit(0)
        counter+=batch['features'].shape[0]
        pos_counter += batch['label'].sum()
    for batch in tqdm(val_loader, desc="Processing val_datset"):
        # print(batch['features'].shape)
        counter+=batch['features'].shape[0]
        # print(batch['features'][0][71 - batch['lens'][0] :][:,0])
        # print(batch['label'][0])
        # print(batch['mask'][0])
        # print(~batch['mask'][0])
        # print(batch['id'][0])
        # print(batch['last_ce_logtime'][0])
        # print(batch['win_end_time'][0])
        # print(batch['lens'][0])
        # break
        pos_counter += batch['label'].sum()
        
    print(counter, pos_counter, counter - pos_counter)

def test_test():
    test_ds = time_series_dataset("/backup/home/zhangrengang/workspace/STIM_Data1/STIM_win_feature_A/", is_train=False)
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=16, shuffle=True, collate_fn=collate_func)
    counter = 0
    pos_counter = 0
    neg_counter = 0
    for batch in tqdm(test_loader, desc="Processing test_datset"):
        if counter == 0:
            print(batch['features'][0][-2:])
            print(batch['mask'][0])
            print(batch['features'][0][:][:,0]//3600)
            print(batch['label'][0])
            print(batch['id'][0])
            print(batch['last_ce_logtime'][0])
            print(batch['win_end_time'][0])
            print(batch['lens'][0])
            exit(0)
        counter+=batch['features'].shape[0]
        pos_counter += batch['label'].sum()
    for batch in tqdm(val_loader, desc="Processing val_datset"):
        # print(batch['features'].shape)
        counter+=batch['features'].shape[0]
        # print(batch['features'][0][71 - batch['lens'][0] :][:,0])
        # print(batch['label'][0])
        # print(batch['mask'][0])
        # print(~batch['mask'][0])
        # print(batch['id'][0])
        # print(batch['last_ce_logtime'][0])
        # print(batch['win_end_time'][0])
        # print(batch['lens'][0])
        # break
        pos_counter += batch['label'].sum()
        
    print(counter, pos_counter, counter - pos_counter)
def set_seed():
    import random
    import torch.backends.cudnn as cudnn
    # fix randomseed for reproducing the results
    print('Setting random seed for reproductivity..')
    random_seed = 3407
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed()
    test_test()
    # Xgboost_train()
    # Xgboost_infer("/mnt/zhangrengang/workspace/myMFP/model_pth/XBGClassifier_model.pkl")
