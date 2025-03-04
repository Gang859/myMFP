import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import time
import sys
import pandas as pd
import json
if __name__ == "__main__":
    # pred_data = []
    # with open("/mnt/zhangrengang/workspace/myMFP/exp/zrg/model4_infer/result.csv","r") as file:
    #     for idx, line in enumerate(file):
    #         if idx > 0:
    #             pred_data.append(int(line.split(",")[1]))
    # for data_by_second in pred_data:
    #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data_by_second)))
    #     if int(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data_by_second)).split("-")[0]) < 6:
    #         assert(False)
    
    
    # with open("/backup/home/zhangrengang/workspace/STIM_Data_train05_test_06/STIM_win_feature_A/aug_pos_windows/sn_35644.json", 'r') as json_f:
    #     win_dict = json.load(json_f)
    #     json_f.close()
    # win_end_time_list = win_dict["win_end_time"]
    # for i in range(len(win_end_time_list)):
    #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(win_end_time_list[i]))))
    
    # file_list = os.listdir("/backup/home/zhangrengang/workspace/STIM_Data1/STIM_win_feature_A/aug_pos_windows")
    # file2_list = os.listdir("/backup/home/zhangrengang/workspace/STIM_Data_train05_test_06/STIM_win_feature_A/aug_pos_windows")
    # print("file num diff(neg56 - neg45)", len(file_list) - len(file2_list))
    # file_list.sort()
    # sumnum = 0
    # file_total_window_num = 0
    # file2_total_window_num = 0
    # for sn_name in file_list:
    #     with open("/backup/home/zhangrengang/workspace/STIM_Data1/STIM_win_feature_A/aug_pos_windows/" + sn_name, 'r') as json_f1:
    #         win_dict = json.load(json_f1)
    #         json_f1.close()
    #     len56trian = len(win_dict["labels"])
    #     file_total_window_num+=len56trian

    # for sn_name in file2_list:
    #     with open("/backup/home/zhangrengang/workspace/STIM_Data_train05_test_06/STIM_win_feature_A/aug_pos_windows/" + sn_name, 'r') as json_f1:
    #         win_dict = json.load(json_f1)
    #         json_f1.close()
    #     len45trian = len(win_dict["labels"])
    #     file2_total_window_num+=len45trian
    # print(file_total_window_num, file2_total_window_num)
'''
STIM_Data1: 
    pos win num:            chunknum:106
    aug pos win num:20479   chunknum:640
    neg win num:1866281     chunknum:13642
    test win num: 4456821   chunknum:32823
    
    
STIM_Data_train05_test_06:
    pos win num:            chunknum:
    aug pos win num:15841   chunknum:496
    neg win num:1809298     chunknum:56541
    test win num: 1967043   chunknum:32823

'''
        