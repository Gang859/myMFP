import os
from multiprocessing import Pool
from tqdm import tqdm
import json
import pandas as pd
import numpy as np


# ----------路径配置
ori_dir = "/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc/"
new_dir = "/backup/home/zhangrengang/workspace/Doc/win30m_feature_with_ecc_smoothened/"
feature_name = "neg_windows_feature"
os.makedirs(new_dir + feature_name, exist_ok=True)

# ------------参数设置
USE_MULTI_PROCESS = True
WORKER_NUM = 40
SMOOTH_WINDOW_LENGTH = 85  # 85 (6 * 3600s / 256)


def process_single_chunk_file(chunkfile: str):
    chunkfile_path = os.path.join(ori_dir+feature_name, chunkfile)
    with open(chunkfile_path, 'r') as file_in:
        chunk_dict = json.load(file_in)
        file_in.close()

    new_features = []
    for win_idx in range(len(chunk_dict["features"])):
        feature_df = pd.DataFrame(
            np.array(chunk_dict["features"][win_idx]),
            columns=[
                "LogTime",               # 0
                "deviceID",
                "BankId",
                "RowId",
                "ColumnId",
                "MciAddr",               # 5
                "RetryRdErrLogParity",
                "error_type_is_READ_CE",
                "error_type_is_SCRUB_CE",
                "bit_count",
                "dq_count",              # 10
                "burst_count",
                "max_dq_interval",
                "max_burst_interval",
                "ecc_bit_0",
                "ecc_bit_1",  # 15
                "ecc_bit_2",
                "ecc_bit_3",
                "ecc_bit_4",
                "ecc_bit_5",
                "ecc_bit_6",  # 20
                "ecc_bit_7",
                "ecc_bit_8",
                "ecc_bit_9",
                "ecc_bit_10",
                "ecc_bit_11",  # 25
                "ecc_bit_12",
                "ecc_bit_13",
                "ecc_bit_14",
                "ecc_bit_15",
                "ecc_bit_16",  # 30
                "ecc_bit_17",
                "ecc_bit_18",
                "ecc_bit_19",
                "ecc_bit_20",
                "ecc_bit_21",  # 35
                "ecc_bit_22",
                "ecc_bit_23",
                "ecc_bit_24",
                "ecc_bit_25",
                "ecc_bit_26",  # 40
                "ecc_bit_27",
                "ecc_bit_28",
                "ecc_bit_29",
                "ecc_bit_30",
                "ecc_bit_31",  # 45
                "Lens",
                "Count",
            ]
        )
        feature_df["time_index"] = feature_df["LogTime"] // SMOOTH_WINDOW_LENGTH
        log_times = feature_df["LogTime"].values
        smooth_win_end_time = feature_df.groupby(
            "time_index")["LogTime"].max().values
        feature_df = feature_df.drop('time_index', axis=1)
        smooth_win_start_time = smooth_win_end_time - SMOOTH_WINDOW_LENGTH
        start_indices = np.searchsorted(
            log_times, smooth_win_start_time, side="left")
        end_indices = np.searchsorted(
            log_times, smooth_win_end_time, side="right")
        smooth_win_list = []
        for start_idx, end_idx in zip(
            start_indices, end_indices
        ):
            smooth_win_dict = {}
            smooth_win_df = feature_df.iloc[start_idx:end_idx]
            smooth_win_dict["LogTime"] = smooth_win_df["LogTime"].values.max()
            # --------------unused features
            # smooth_win_dict["deviceID"] = -1
            # smooth_win_dict["BankId"] = -1
            # smooth_win_dict["RowId"] = -1
            # smooth_win_dict["ColumnId"] = -1
            # smooth_win_dict["MciAddr"] = -1
            # smooth_win_dict["RetryRdErrLogParity"] = -1
            # smooth_win_dict["error_type_is_READ_CE"] = -1
            # smooth_win_dict["error_type_is_SCRUB_CE"] = -1
            # --------------smooth count features
            smooth_win_dict["bit_count"] = smooth_win_df["Count"].sum()
            smooth_win_dict["dq_count"] = smooth_win_df["dq_count"].sum()
            smooth_win_dict["burst_count"] = smooth_win_df["burst_count"].sum()
            smooth_win_dict["max_dq_interval"] = smooth_win_df["max_dq_interval"].values.max()
            smooth_win_dict["max_burst_interval"] = smooth_win_df["max_burst_interval"].values.max()
            # --------------smooth the error checking bits
            for i in range(32):
                smooth_win_dict[f"ecc_bit_{i}"] = smooth_win_df.apply(
                    lambda x:
                        x[f"ecc_bit_{i}"] * x["Count"],
                    axis=1
                ).sum()
            # -------------copy other features
            smooth_win_dict["Lens"] = smooth_win_df["Lens"].values[0]
            smooth_win_dict["Count"] = smooth_win_df["Count"].values.sum()
            smooth_win_list.append(smooth_win_dict)
        smooth_win_feature_list = pd.DataFrame(smooth_win_list).values.tolist()
        new_features.append(smooth_win_feature_list)
    # -------------check feature dim
    # assert len(chunk_dict["features"][0][0]) == len(new_features[0][0])
    chunk_dict["features"] = new_features
    # --------------write new chunk
    new_chunkfile_path = os.path.join(new_dir+feature_name, chunkfile)
    with open(new_chunkfile_path, 'w') as file_out:
        json_str = json.dumps(chunk_dict, indent=4)
        file_out.write(json_str)
        file_out.close()


def process_all_chunk_file():
    chunk_file_list = os.listdir(ori_dir + feature_name)
    exist_new_chunk_file_list = os.listdir(new_dir + feature_name)
    
    chunk_file_list = [fname for fname in chunk_file_list if (
        fname not in exist_new_chunk_file_list and # 不重复生成
        '.json' in fname)]
    
    chunk_file_list.sort()
    if USE_MULTI_PROCESS:
        with Pool(WORKER_NUM) as pool:
            list(
                tqdm(
                    pool.imap(process_single_chunk_file, chunk_file_list),
                    total=len(chunk_file_list),
                    desc="processing chunk files",
                )
            )
    else:
        for chunkfile in tqdm(chunk_file_list, desc="processing chunk files"):
            process_single_chunk_file(chunkfile)


if __name__ == "__main__":
    process_all_chunk_file()
