import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import time
import sys
import pandas as pd

sys.path.append("/mnt/zhangrengang/workspace/myMFP/src")

from window_dataset_STIM import time_series_dataset, collate_func, time_series_dataset2, collate_func2

if __name__ == "__main__":
    ds = time_series_dataset2("/mnt/zhangrengang/data/STIM_Data_train_01-04/STIM_win_feature_obv30d_neg60d_A/", is_train=True)
    train_size = int((0.9 * len(ds)))
    val_size = int((len(ds) - train_size))
    train_ds, val_ds = torch.utils.data.random_split(
    ds, [train_size, val_size], generator=torch.Generator().manual_seed(223))
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=8, shuffle=True, collate_fn=collate_func2)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=8, shuffle=False, collate_fn=collate_func2)
    
    ticket = pd.read_csv("/backup/home/zhangrengang/workspace/Doc/competition_data/stage1_feather/ticket.csv")
    ticket = ticket[ticket["alarm_time"] <= int(datetime.strptime("2024-06-01", "%Y-%m-%d").timestamp())]
    ticket_sn_map = {
            sn: sn_t
            for sn, sn_t in zip(list(ticket["sn_name"]), list(ticket["alarm_time"]))
        }
    
    counter = 1500000
    for batch in tqdm(train_loader, desc="Processing train_datset"):
        for i in range(len( batch["label"])):
            # if batch['id'][i] in ticket_sn_map:
            #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(batch['win_end_time'][i])), 
            #         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ticket_sn_map[batch['id'][i]])))
            month = int(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(batch['win_end_time'][i])).split("-")[1])
            # assert(month >=4 and month <=5)
            if not (month <= 3) :
                assert(False)
            counter -= 1
            if counter == 0:
                exit(0)
        
    