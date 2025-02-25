import os
import json
import pandas as pd


window_feature_dir = "/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc/"
windows_json_files_dir = window_feature_dir + "windows"
pos_windows_json_files_dir = window_feature_dir + "pos_windows"
neg_windows_json_files_dir = window_feature_dir + "neg_windows"
test_windows_json_files_dir = window_feature_dir + "test_windows"

GROUP_SIZE = 32

if __name__ == "__main__":
    length_list = []
    
    file_list = os.listdir(pos_windows_json_files_dir)
    for file in file_list:
        if '.json' in file:
            with open(os.path.join(pos_windows_json_files_dir, file), 'r') as in_file:
                win_dict = json.load(in_file)
                length_list += win_dict["lens"]
                in_file.close()
                
    # file_list = os.listdir(neg_windows_json_files_dir)
    # for file in file_list:
    #     if '.json' in file:
    #         with open(os.path.join(neg_windows_json_files_dir, file), 'r') as in_file:
    #             win_dict = json.load(in_file)
    #             length_list += win_dict["lens"]
    #             in_file.close()
    
    length_list.sort()
    group_id_list = [x // GROUP_SIZE for x in length_list]
    df = pd.DataFrame(length_list, columns=["lens"])
    df = df.assign(group_id = group_id_list)
    
    df = df.assign(
        count=df.groupby("group_id")[
            "group_id"
        ].transform("count")
    )
    
    df = df.drop_duplicates(
        subset="group_id", keep="first"
    )
    
    df.to_csv("/mnt/zhangrengang/workspace/myMFP/output/window_length.csv")
    