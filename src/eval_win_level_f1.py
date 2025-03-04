import os
import pandas as pd
from datetime import datetime
import json

submission_file = "/mnt/zhangrengang/workspace/myMFP/exp/zrg/model4_val4result_infer/result.csv"
test_window_dir = "/mnt/zhangrengang/data/STIM_Data_test_05-06/STIM_win_feature_obv30d_neg60d_A/test_windows"
ticket_path = "/backup/home/zhangrengang/workspace/Doc/competition_data/stage1_feather/ticket.csv" 
output_file = "/mnt/zhangrengang/workspace/myMFP/output/test_label.json"

test_start_date = "2024-05-01"
test_end_date = "2024-06-01"

ONE_MINUTE = 60
ONE_HOUR = 60 * ONE_MINUTE
ONE_DAY = 24 * ONE_HOUR

def gen_test_label():
    # read ticket file
    ticket = pd.read_csv(ticket_path)
    ticket_sn_map = {
        sn: sn_t
        for sn, sn_t in zip(list(ticket["sn_name"]), list(ticket["alarm_time"]))
    }
    # cal timestamp
    # load_test_window
    sn_file_list = os.listdir(test_window_dir)
    test_lable_map = {}
    test_lable_map["pos_win_7d"] = []
    test_lable_map["pos_win_30d"] = []
    for sn_file in sn_file_list:
        sn_name = sn_file.split(".")[0]
        with open(os.path.join(test_window_dir, sn_file), 'r') as fin:
            windows_dict = json.load(fin)
            fin.close()
        for last_ce_log_time in windows_dict["win_last_ce_logtime"]:
            if sn_name in ticket_sn_map:
                ue_time = ticket_sn_map[sn_name]
                if last_ce_log_time > ue_time - 7 * ONE_DAY and last_ce_log_time < ue_time - 15 * ONE_MINUTE:
                    test_lable_map["pos_win_7d"].append(sn_name+"_"+str(last_ce_log_time))
                else:
                    pass
                
                if last_ce_log_time > ue_time - 30 * ONE_DAY and last_ce_log_time < ue_time:
                    test_lable_map["pos_win_30d"].append(sn_name+"_"+str(last_ce_log_time))
                else:
                    pass
            else:
                pass
            # test_lable_map[dict_key] = [label_7d, label_30d]
        
    with open(output_file, 'w') as fout:
        json_str = json.dumps(test_lable_map, indent=4)
        fout.write(json_str)
        fout.close()
        
        test_start_timestamp = int(datetime.strptime(test_start_date, "%Y-%m-%d").timestamp())
        test_end_timestamp = int(datetime.strptime(test_end_date, "%Y-%m-%d").timestamp())
        
    
    return test_lable_map

def eval_test_f1_score(test_lable_map):
    ticket = pd.read_csv(ticket_path)
    ticket = ticket
    ticket_sn_map = {
        sn: sn_t
        for sn, sn_t in zip(list(ticket["sn_name"]), list(ticket["alarm_time"]))
    }
    submissin_df = pd.read_csv(submission_file)
    tp_7d = 0
    tp_30d = 0
    
    for index, sb_row in submissin_df.iterrows():
        pred_sn_name = sb_row["sn_name"]
        pred_time= sb_row["prediction_timestamp"]
        key_str = pred_sn_name + "_" + str(int(pred_time))
        
        # ------- 7d score -------
        if key_str in test_lable_map["pos_win_7d"]:
            tp_7d += 1
        
        # ------- 30d score -------
        if key_str in test_lable_map["pos_win_30d"]:
            tp_30d += 1

    total_pred_pos = submissin_df.shape[0]
    precision_7d = float(tp_7d) / total_pred_pos
    precision_30d = float(tp_30d) / total_pred_pos
    
    total_true_pos_7d = len(test_lable_map["pos_win_7d"])
    recall_7d = float(tp_7d) / total_true_pos_7d
    total_true_pos_30d = len(test_lable_map["pos_win_30d"])
    recall_30d = float(tp_30d) / total_true_pos_30d
    
    print("========= win level f1 score =========")
    print("-------7d------")
    print(f"tp_7d: {tp_7d} \t precision_7d: {precision_7d:.4f} \t recall_7d: {recall_7d:.4f} \t f1: {2 * precision_7d * recall_7d / (precision_7d + recall_7d)}")
    print(f"fp: {total_pred_pos - tp_7d} \t fn: {total_true_pos_7d - tp_7d}")
    print("-------30d------")
    print(f"tp_30d: {tp_30d} \t precision_30d: {precision_30d:.4f} \t recall_30d: {recall_30d:.4f} \t f1: {2 * precision_30d * recall_30d / (precision_30d + recall_30d)}")
    print(f"fp: {total_pred_pos - tp_30d} \t fn: {total_true_pos_30d - tp_30d}")
    
        
    
    

if __name__ == "__main__":
    test_lable_map = gen_test_label()
    eval_test_f1_score(test_lable_map)