# -*- coding: utf-8 -*-
import abc
import os
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, Tuple, NoReturn, Union, List
import json
import argparse
import feather
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from tqdm import tqdm
import queue
import threading

# 定义时间常量（单位：秒）
ONE_MINUTE = 60  # 一分钟的秒数
ONE_HOUR = 3600  # 一小时的秒数（60秒 * 60分钟）
ONE_DAY = 86400  # 一天的秒数（60秒 * 60分钟 * 24小时）

OBS_WIN_LENGHT = 30 * ONE_DAY
WIN_MAX_LENGTH =  3 * ONE_DAY 
FEATURE_INTERVAL = 1 * ONE_HOUR
# TODO
# NEG_GEN_INTERVAL = 60 * ONE_DAY
NEG_GEN_INTERVAL = 30 * ONE_DAY

@dataclass
class Config(object):
    """
    配置类, 用于存储和管理程序的配置信息
    包括时间窗口大小、路径设置、日期范围、特征提取间隔等
    """

    # 重要! 如果使用 csv 文件则设置 DATA_SUFFIX 为 csv, 如果使用 feather 文件则设置为 feather
    DATA_SUFFIX: str = field(default="feather", init=False)

    # 时间窗口大小映射表, 键为时间长度(秒), 值为对应的字符串表示
    TIME_WINDOW_SIZE_MAP: dict = field(
        default_factory=lambda: {
            15 * ONE_MINUTE: "15m",
            1 * ONE_HOUR: "1h",
            6 * ONE_HOUR: "6h"
        },
        init=False,
    )

    # 与时间相关的列表, 存储常用的时间间隔(秒)
    TIME_RELATED_LIST: List[int] = field(
        default_factory=lambda: [15 * ONE_MINUTE, ONE_HOUR, 6 * ONE_HOUR],
        init=False,
    )

    # 缺失值填充的默认值
    IMPUTE_VALUE: int = field(default=-1, init=False)

    USE_MULTI_PROCESS: bool = field(default=True, init=False)

    # 如果使用多进程, 并行时 worker 的数量
    WORKER_NUM: int = field(default=64, init=False)

    # 数据路径配置, 分别是原始数据集路径、生成的特征路径、处理后训练集特征路径、处理后测试集特征路径、维修单路径
    data_path: str = "To be filled"
    feature_path: str = "To be filled"
    train_data_path: str = "To be filled"
    test_data_path: str = "To be filled"
    ticket_path: str = "To be filled"

    # 日期范围配置
    train_date_range: tuple = ("2024-01-01", "2024-06-01")
    test_data_range: tuple = ("2024-06-01", "2024-08-01")

    # 特征提取的时间间隔(秒), 为了更高的性能, 可以修改为 15 * ONE_MINUTE 或 30 * ONE_MINUTE
    # 生成训练数据：interval = 30 min
    # 生成测试数据：interval = 90 min
    feature_interval: int = 30 * ONE_MINUTE

    # dump json path


class FeatureFactory(object):
    """
    特征工厂类, 用于生成特征
    """

    # 考虑 DDR4 内存, 其 DQ_COUNT 和 BURST_COUNT 分别为 4 和 8
    DQ_COUNT = 4
    BURST_COUNT = 8

    def __init__(self, config: Config, processed_win1h_df_files_dir:str="to be filled", processed_ce_df_files_dir:str="to be filled", windows_json_files_dir:str="to be filled"):
        """
        初始化特征工厂类

        :param config: 配置类实例, 包含路径等信息
        """

        self.config = config
        self.processed_win1h_df_files_dir = processed_win1h_df_files_dir
        self.processed_ce_df_files_dir = processed_ce_df_files_dir
        self.windows_json_files_dir = windows_json_files_dir
        

    def _unique_num_filtered(self, input_array: np.ndarray) -> int:
        """
        对输入的列表进行去重, 再去除值为 IMPUTE_VALUE 的元素后, 统计元素个数

        :param input_array: 输入的列表
        :return: 返回经过过滤后的列表元素个数
        """

        unique_array = np.unique(input_array)
        return len(unique_array) - int(self.config.IMPUTE_VALUE in unique_array)

    @staticmethod
    def _calculate_ce_storm_count(
        log_times: np.ndarray,
        ce_storm_interval_seconds: int = 60,
        ce_storm_count_threshold: int = 10,
    ) -> int:
        """
        计算 CE 风暴的数量

        CE 风暴定义:
        - 首先定义相邻 CE 日志: 若两个 CE 日志 LogTime 时间间隔 < 60s, 则为相邻日志;
        - 如果相邻日志的个数 >10, 则为发生 1 次 CE 风暴(注意: 如果相邻日志数量持续增长, 超过了 10, 则也只是记作 1 次 CE 风暴)

        :param log_times: 日志 LogTime 列表
        :param ce_storm_interval_seconds: CE 风暴的时间间隔阈值
        :param ce_storm_count_threshold: CE 风暴的数量阈值
        :return: CE风暴的数量
        """

        log_times = sorted(log_times)
        ce_storm_count = 0
        consecutive_count = 0

        for i in range(1, len(log_times)):
            if log_times[i] - log_times[i - 1] <= ce_storm_interval_seconds:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count > ce_storm_count_threshold:
                ce_storm_count += 1
                consecutive_count = 0

        return ce_storm_count

    def _get_temporal_features(
        self, window_df: pd.DataFrame, time_window_size: int
    ) -> Dict[str, int]:
        """
        获取时间特征, 包括 CE 数量、日志数量、CE 风暴数量、日志发生频率等

        :param window_df: 时间窗口内的数据
        :param time_window_size: 时间窗口大小
        :return: 时间特征

        - read_ce_log_num, read_ce_count: 时间窗口内, 读 CE 的 count 总数, 日志总数
        - scrub_ce_log_num, scrub_ce_count: 时间窗口内, 巡检 CE 的 count 总数, 日志总数
        - all_ce_log_num, all_ce_count: 时间窗口内, 所有 CE 的 count 总数, 日志总数
        - log_happen_frequency: 日志发生频率
        - ce_storm_count: CE 风暴数量
        """

        error_type_is_READ_CE = window_df["error_type_is_READ_CE"].values
        error_type_is_SCRUB_CE = window_df["error_type_is_SCRUB_CE"].values
        ce_count = window_df["Count"].values

        temporal_features = dict()
        temporal_features["read_ce_log_num"] = error_type_is_READ_CE.sum()
        temporal_features["scrub_ce_log_num"] = error_type_is_SCRUB_CE.sum()
        temporal_features["all_ce_log_num"] = len(window_df)

        temporal_features["read_ce_count"] = (
            error_type_is_READ_CE * ce_count).sum()
        temporal_features["scrub_ce_count"] = (
            error_type_is_SCRUB_CE * ce_count).sum()
        temporal_features["all_ce_count"] = ce_count.sum()

        temporal_features["log_happen_frequency"] = (
            time_window_size / len(window_df) if not window_df.empty else 0
        )
        temporal_features["ce_storm_count"] = self._calculate_ce_storm_count(
            window_df["LogTime"].values
        )
        return temporal_features

    def _get_spatio_features(self, window_df: pd.DataFrame) -> Dict[str, int]:
        """
        获取空间特征, 包括故障模式, 同时发生行列故障的数量

        :param window_df: 时间窗口内的数据
        :return: 空间特征

        - fault_mode_others: 其他故障, 即多个 device 发生故障
        - fault_mode_device: device 故障, 相同 id 的 device 发生多个 bank 故障
        - fault_mode_bank: bank 故障, 相同 id 的 bank 发生多个 row故障
        - fault_mode_row: row 故障, 相同 id 的 row 发生多个不同 column 的 cell 故障
        - fault_mode_column: column 故障, 相同 id 的 column 发生多个不同 row 的 cell 故障
        - fault_mode_cell: cell 故障, 发生多个相同 id 的 cell 故障
        - fault_row_num: 同时发生 row 故障的行个数
        - fault_column_num: 同时发生 column 故障的列个数
        """

        spatio_features = {
            "fault_mode_others": 0,
            "fault_mode_device": 0,
            "fault_mode_bank": 0,
            "fault_mode_row": 0,
            "fault_mode_column": 0,
            "fault_mode_cell": 0,
            "fault_row_num": 0,
            "fault_column_num": 0,
        }

        # 根据故障设备、Bank、行、列和单元的数量判断故障模式
        if self._unique_num_filtered(window_df["deviceID"].values) > 1:
            spatio_features["fault_mode_others"] = 1
        elif self._unique_num_filtered(window_df["BankId"].values) > 1:
            spatio_features["fault_mode_device"] = 1
        elif (
            self._unique_num_filtered(window_df["ColumnId"].values) > 1
            and self._unique_num_filtered(window_df["RowId"].values) > 1
        ):
            spatio_features["fault_mode_bank"] = 1
        elif self._unique_num_filtered(window_df["ColumnId"].values) > 1:
            spatio_features["fault_mode_row"] = 1
        elif self._unique_num_filtered(window_df["RowId"].values) > 1:
            spatio_features["fault_mode_column"] = 1
        elif self._unique_num_filtered(window_df["CellId"].values) == 1:
            spatio_features["fault_mode_cell"] = 1

        # 记录相同行对应的列地址信息
        row_pos_dict = {}
        # 记录相同列对应的行地址信息
        col_pos_dict = {}

        for device_id, bank_id, row_id, column_id in zip(
            window_df["deviceID"].values,
            window_df["BankId"].values,
            window_df["RowId"].values,
            window_df["ColumnId"].values,
        ):
            current_row = "_".join([str(pos)
                                   for pos in [device_id, bank_id, row_id]])
            current_col = "_".join(
                [str(pos) for pos in [device_id, bank_id, column_id]]
            )
            row_pos_dict.setdefault(current_row, [])
            col_pos_dict.setdefault(current_col, [])
            row_pos_dict[current_row].append(column_id)
            col_pos_dict[current_col].append(row_id)

        for row in row_pos_dict:
            if self._unique_num_filtered(np.array(row_pos_dict[row])) > 1:
                spatio_features["fault_row_num"] += 1
        for col in col_pos_dict:
            if self._unique_num_filtered(np.array(col_pos_dict[col])) > 1:
                spatio_features["fault_column_num"] += 1

        return spatio_features

    @staticmethod
    def _get_err_parity_features(window_df: pd.DataFrame) -> Dict[str, int]:
        """
        获取奇偶校验特征

        :param window_df: 时间窗口内的数据
        :return: 奇偶校验特征

        - error_bit_count: 时间窗口内, 总错误 bit 数
        - error_dq_count: 时间窗口内, 总 dq 错误数
        - error_burst_count: 时间窗口内, 总 burst 错误数
        - max_dq_interval: 时间窗口内, 每个 parity 最大错误 dq 距离的最大值
        - max_burst_interval: 时间窗口内, 每个 parity 最大错误 burst 距离的最大值
        - dq_count=n: dq 错误数等于 n 的总数量, n 取值范围为 [1, 2, 3, 4], 默认值为 0
        - burst_count=n: burst 错误数等于 n 的总数量, n 取值范围为 [1, 2, 3, 4, 5, 6, 7, 8], 默认值为 0
        """

        err_parity_features = dict()

        # 计算总错误 bit 数、DQ 错误数和 Burst 错误数
        err_parity_features["error_bit_count"] = window_df["bit_count"].values.sum(
        )
        err_parity_features["error_dq_count"] = window_df["dq_count"].values.sum()
        err_parity_features["error_burst_count"] = window_df["burst_count"].values.sum(
        )

        # 计算最大 DQ 间隔和最大 Burst 间隔
        err_parity_features["max_dq_interval"] = window_df[
            "max_dq_interval"
        ].values.max()
        err_parity_features["max_burst_interval"] = window_df[
            "max_burst_interval"
        ].values.max()

        # 统计 DQ 错误数和 Burst 错误数的分布
        dq_counts = dict()
        burst_counts = dict()
        for dq, burst in zip(
            window_df["dq_count"].values, window_df["burst_count"].values
        ):
            dq_counts[dq] = dq_counts.get(dq, 0) + 1
            burst_counts[burst] = burst_counts.get(burst, 0) + 1

        # 计算 'dq错误数=n' 的总数量, DDR4 内存的 DQ_COUNT 为 4, 因此 n 取值 [1,2,3,4]
        for dq in range(1, FeatureFactory.DQ_COUNT + 1):
            err_parity_features[f"dq_count={dq}"] = dq_counts.get(dq, 0)

        # 计算 'burst错误数=n' 的总数量, DDR4 内存的 BURST_COUNT 为 8, 因此 n 取值 [1,2,3,4,5,6,7,8]
        for burst in [1, 2, 3, 4, 5, 6, 7, 8]:
            err_parity_features[f"burst_count={burst}"] = burst_counts.get(
                burst, 0)

        return err_parity_features

    @staticmethod
    def _get_bit_dq_burst_info(bin_parity) -> Tuple[int, int, int, int, int]:
        """
        获取特定 parity 的奇偶校验信息

        :param parity: 奇偶校验值
        :return: parity 的奇偶校验信息

        - bit_count: parity 错误 bit 数量
        - dq_count: parity 错误 dq 数量
        - burst_count: parity 错误 burst 数量
        - max_dq_interval: parity 错误 dq 的最大间隔
        - max_burst_interval: parity 错误 burst 的最大间隔
        """

        # 将 Parity 转换为 32 位二进制字符串
        # bin_parity = bin(parity)[2:].zfill(32)

        # 计算错误 bit 数量
        bit_count = bin_parity.count("1")

        # 计算 burst 相关特征
        binary_row_array = [
            bin_parity[i: i + 4].count("1") for i in range(0, 32, 4)]
        binary_row_array_indices = [
            idx for idx, value in enumerate(binary_row_array) if value > 0
        ]
        burst_count = len(binary_row_array_indices)
        max_burst_interval = (
            binary_row_array_indices[-1] - binary_row_array_indices[0]
            if binary_row_array_indices
            else 0
        )

        # 计算 dq 相关特征
        binary_column_array = [bin_parity[i::4].count("1") for i in range(4)]
        binary_column_array_indices = [
            idx for idx, value in enumerate(binary_column_array) if value > 0
        ]
        dq_count = len(binary_column_array_indices)
        max_dq_interval = (
            binary_column_array_indices[-1] - binary_column_array_indices[0]
            if binary_column_array_indices
            else 0
        )

        return bit_count, dq_count, burst_count, max_dq_interval, max_burst_interval

    def _get_processed_ce_df(self, sn_file:str) -> pd.DataFrame:
        # 检查对应路径下是否已经存在已经生成好的DataFrame
        if os.path.exists(os.path.join(self.processed_ce_df_files_dir, sn_file.split('.')[0] + '.feather')):
            processed_ce_df = feather.read_dataframe(os.path.join(
                self.processed_ce_df_files_dir, sn_file.split('.')[0] + '.feather'))
            return processed_ce_df
        
        # =============================== ce features begin ===============================
        # 读取原始数据(CE)并按 LogTime 排序
        if self.config.DATA_SUFFIX == "csv":
            raw_df = pd.read_csv(os.path.join(self.config.data_path, sn_file))
        else:
            raw_df = feather.read_dataframe(
                os.path.join(self.config.data_path, sn_file))

        raw_df = raw_df.sort_values(by="LogTime").reset_index(drop=True)

        # 提取需要的列并初始化 processed_df
        processed_ce_df = raw_df[
            [
                "LogTime",
                "deviceID",
                "BankId",
                "RowId",
                "ColumnId",
                "MciAddr",
                "RetryRdErrLogParity",
            ]
        ].copy()

        # deviceID 可能存在缺失值, 填充缺失值
        processed_ce_df["deviceID"] = (
            processed_ce_df["deviceID"].fillna(
                self.config.IMPUTE_VALUE).astype(int)
        )

        # 将 error_type 转换为独热编码
        processed_ce_df["error_type_is_READ_CE"] = (
            raw_df["error_type_full_name"] == "CE.READ"
        ).astype(int)
        processed_ce_df["error_type_is_SCRUB_CE"] = (
            raw_df["error_type_full_name"] == "CE.SCRUB"
        ).astype(int)

        # 记录位置信息，便于之后去重
        processed_ce_df["CellId"] = (
            processed_ce_df["RowId"].astype(str)
            + "_"
            + processed_ce_df["ColumnId"].astype(str)
        )
        processed_ce_df["position_and_parity"] = (
            processed_ce_df["deviceID"].astype(str)
            + "_"
            + processed_ce_df["BankId"].astype(str)
            + "_"
            + processed_ce_df["RowId"].astype(str)
            + "_"
            + processed_ce_df["ColumnId"].astype(str)
            + "_"
            + processed_ce_df["RetryRdErrLogParity"].astype(str)
        )

        # 获取校验码信息
        err_log_parity_array = (
            processed_ce_df["RetryRdErrLogParity"]
            .fillna(0)
            .replace("", 0)
            .astype(np.int64)  # 转换为 np.int64, 此处如果为 int 会溢出
            .values
        )

        # 计算每个 parity 的 bit_count、dq_count、burst_count、max_dq_interval 和 max_burst_interval
        parity_dict = dict()
        bit_dq_burst_count = list()
        # 32位checkingbits
        checkingbits = list()
        for idx, err_log_parity in enumerate(err_log_parity_array):
            bin_parity = bin(err_log_parity)[2:].zfill(32)
            checkingbits.append(
                [bool(True) if x == '1' else bool(False) for x in list(bin_parity)])
            if err_log_parity not in parity_dict:
                parity_dict[err_log_parity] = self._get_bit_dq_burst_info(
                    bin_parity
                )
            bit_dq_burst_count.append(parity_dict[err_log_parity])
        # 校验妈的分布特征
        processed_ce_df = processed_ce_df.join(
            pd.DataFrame(
                bit_dq_burst_count,
                columns=[
                    "bit_count",
                    "dq_count",
                    "burst_count",
                    "max_dq_interval",
                    "max_burst_interval",
                ],
            )
        )
        # 将十进制的校验码转换成二进制代码
        processed_ce_df = processed_ce_df.join(
            pd.DataFrame(
                checkingbits,
                columns=[
                    "ecc_bit_0",
                    "ecc_bit_1",
                    "ecc_bit_2",
                    "ecc_bit_3",
                    "ecc_bit_4",
                    "ecc_bit_5",
                    "ecc_bit_6",
                    "ecc_bit_7",
                    "ecc_bit_8",
                    "ecc_bit_9",
                    "ecc_bit_10",
                    "ecc_bit_11",
                    "ecc_bit_12",
                    "ecc_bit_13",
                    "ecc_bit_14",
                    "ecc_bit_15",
                    "ecc_bit_16",
                    "ecc_bit_17",
                    "ecc_bit_18",
                    "ecc_bit_19",
                    "ecc_bit_20",
                    "ecc_bit_21",
                    "ecc_bit_22",
                    "ecc_bit_23",
                    "ecc_bit_24",
                    "ecc_bit_25",
                    "ecc_bit_26",
                    "ecc_bit_27",
                    "ecc_bit_28",
                    "ecc_bit_29",
                    "ecc_bit_30",
                    "ecc_bit_31",
                ],
            )
        )
        feather.write_dataframe(
            processed_ce_df,
            os.path.join(os.path.join(self.processed_ce_df_files_dir,
                         sn_file.split('.')[0] + '.feather')),
        )
        return processed_ce_df
        
    def _get_processed_1h_win_df(self, sn_file: str) -> pd.DataFrame:
        """
        获取处理后的 DataFram
        :param sn_file: SN 文件名
        :return: 处理后的 DataFrame
            处理后的DataFrame中  每一行表示一个子窗口的特征  子窗口的时常为1HOUR
        """
        # 检查对应路径下是否已经存在已经生成好的DataFrame
        # 这里的DF是1h window的特征df
        if os.path.exists(os.path.join(self.processed_win1h_df_files_dir, sn_file.split('.')[0] + '.feather')):
            processed_win1h_df = feather.read_dataframe(os.path.join(
                self.processed_win1h_df_files_dir, sn_file.split('.')[0] + '.feather'))
            return processed_win1h_df
        # =============================== one hour sub window features begin ===============================
        processed_win1h_df = self._get_processed_ce_df(sn_file)
        # 将sn的CE按照 步长 (ONE HOUR) 分组，得到分组编号
        processed_win1h_df["time_index"] = processed_win1h_df["LogTime"] // ONE_HOUR
        log_times = processed_win1h_df["LogTime"].values
        # 同一组的CE作为一个子窗口
        # 计算每个子时间窗口的结束时间和开始时间
        window_start_times = processed_win1h_df.groupby("time_index")["LogTime"].min().values
        window_end_times = processed_win1h_df.groupby("time_index")["LogTime"].max().values
        # 根据时间窗口的起始和结束时间, 找到对应的数据索引
        start_indices = np.searchsorted(
            log_times, window_start_times, side="left")
        end_indices = np.searchsorted(
            log_times, window_end_times, side="right")
        sub_win_feature_dict_list = []
        counter = 0
        # ------------------------- sub_win features generation 
        for start_idx, end_idx in zip(start_indices, end_indices):
            # 生成子窗口的feature
            sub_win_feature_dict = {}
            # 根据子窗口的index提取对应的CE数据
            sub_win_df = processed_win1h_df.iloc[start_idx:end_idx]
            sub_win_feature_dict["sub_win_last_ce_logtime"] = sub_win_df["LogTime"].values.max()
            # 提取时间特征、空间特征和奇偶校验特征
            # 复制窗口，计数并去重
            sub_win_df_copy = sub_win_df.copy()
            sub_win_df_copy = sub_win_df_copy.assign(
                Count=sub_win_df_copy.groupby("position_and_parity")[
                    "position_and_parity"
                ].transform("count")
            )
            sub_win_df_copy = sub_win_df_copy.drop_duplicates(
                subset="position_and_parity", keep="first"
            )
            temporal_features = self._get_temporal_features(
                sub_win_df_copy, time_window_size=ONE_HOUR
            )
            # NOTE: 使用去重的数据计算空间和奇偶校验特征
            spatio_features = self._get_spatio_features(sub_win_df_copy)
            err_parity_features = self._get_err_parity_features(sub_win_df_copy)
            # 将特征合并到 combined_dict 中, 并添加时间窗口大小的后缀
            sub_win_feature_dict.update(
                {
                    f"{key}_{self.config.TIME_WINDOW_SIZE_MAP[ONE_HOUR]}": value
                    for d in [
                        temporal_features,
                        spatio_features,
                        err_parity_features,
                    ]
                    for key, value in d.items()
                }
            )
            # sum up bit map
            for i in range(32):
                sub_win_feature_dict[f"ecc_bit_{i}"] = sub_win_df[f"ecc_bit_{i}"].sum(
                )
            # add sub_window end time
            sub_win_feature_dict["sub_win_end_time"] = (window_end_times[counter]//ONE_HOUR + 1)*ONE_HOUR
            counter+=1
            sub_win_feature_dict_list.append(sub_win_feature_dict)
        sub_win_feature_df = pd.DataFrame(sub_win_feature_dict_list)

        # NOTE: TEST
        # print(sub_win_feature_df)
        # print(sub_win_feature_df.columns)
        # assert(False)

        # write sub_win_feature_df
        feather.write_dataframe(
            sub_win_feature_df,
            os.path.join(os.path.join(self.processed_win1h_df_files_dir,
                         sn_file.split('.')[0] + '.feather')),
        )
        # =============================== one hour sub window features endin ===============================
        return sub_win_feature_df

    def process_all_sn(self, aug_pos: bool = False) -> NoReturn:
        """
        处理所有 sn 文件, 并保存特征, 支持多进程处理以提高效率
        """
        sn_files = os.listdir(self.config.data_path)
        exist_sn_file_list = os.listdir(windows_json_files_dir)
        sn_files = [
            x for x in sn_files if ((x.replace('.feather', '.json') not in exist_sn_file_list))
            and x.endswith(self.config.DATA_SUFFIX)
        ]
        sn_files.sort()

        if self.config.USE_MULTI_PROCESS:
            worker_num = self.config.WORKER_NUM
            with Pool(worker_num) as pool:
                list(
                    tqdm(
                        pool.imap(
                            self.process_single_sn_for_transfomer_aug_pos if aug_pos else self.process_single_sn_for_transfomer,
                            sn_files
                        ),
                        total=len(sn_files),
                        desc="Generating window index",
                    )
                )
        else:
            for sn_file in tqdm(sn_files, desc="Generating window index"):
                if aug_pos:
                    self.process_single_sn_for_transfomer_aug_pos(sn_file)
                else:
                    self.process_single_sn_for_transfomer(sn_file)

    # 生成大窗口的角标
    def process_single_sn_for_transfomer(self, sn_file: str) -> NoReturn:
        zrg_dict = {"start_sub_win_indices": [],
                    "end_sub_win_indices": [],
                    "win_end_time": [],
                    "win_last_ce_logtime": [],
                    "labels": [],
                    }

        # 获取处理后的 DataFrame
        new_df = self._get_processed_1h_win_df(sn_file)
        # 根据生成特征的间隔, 计算时间索引
        new_df["time_index"] = new_df["sub_win_end_time"] // FEATURE_INTERVAL
        sub_win_end_times = new_df["sub_win_end_time"].values

        # 计算每个时间窗口的结束时间和开始时间, 每次生成特征最多用3DAY的历史数据
        window_end_time = new_df.groupby("time_index")["sub_win_end_time"].max().values
        window_start_times = window_end_time - WIN_MAX_LENGTH
        window_last_ce_logtime = new_df.groupby("time_index")["sub_win_last_ce_logtime"].max().values

        # 根据时间窗口的起始和结束时间, 找到对应的数据索引
        start_indices = np.searchsorted(
            sub_win_end_times, window_start_times, side="left")
        end_indices = np.searchsorted(
            sub_win_end_times, window_end_time, side="right")

        zrg_dict["start_sub_win_indices"] = start_indices.tolist()
        zrg_dict["end_sub_win_indices"] = end_indices.tolist()
        zrg_dict["win_end_time"] = window_end_time.tolist()
        zrg_dict["win_last_ce_logtime"] = window_last_ce_logtime.tolist()
        with open(os.path.join(self.windows_json_files_dir, sn_file.split(".")[0] + '.json'), "w") as zrg_file:
            json_str = json.dumps(zrg_dict, indent=4)
            zrg_file.write(json_str)
            zrg_file.close()


def zrg_unique_num_filtered(input_array: np.ndarray) -> int:
    IMPUTE_VALUE: int = field(default=-1, init=False)
    """
    对输入的列表进行去重, 再去除值为 IMPUTE_VALUE 的元素后, 统计元素个数

    :param input_array: 输入的列表
    :return: 返回经过过滤后的列表元素个数
    """

    unique_array = np.unique(input_array)
    return len(unique_array) - int(IMPUTE_VALUE in unique_array)

def zrg_calculate_ce_storm_count(
    log_times: np.ndarray,
    ce_storm_interval_seconds: int = 60,
    ce_storm_count_threshold: int = 10,
) -> int:
    """
    计算 CE 风暴的数量

    CE 风暴定义:
    - 首先定义相邻 CE 日志: 若两个 CE 日志 LogTime 时间间隔 < 60s, 则为相邻日志;
    - 如果相邻日志的个数 >10, 则为发生 1 次 CE 风暴(注意: 如果相邻日志数量持续增长, 超过了 10, 则也只是记作 1 次 CE 风暴)

    :param log_times: 日志 LogTime 列表
    :param ce_storm_interval_seconds: CE 风暴的时间间隔阈值
    :param ce_storm_count_threshold: CE 风暴的数量阈值
    :return: CE风暴的数量
    """

    log_times = sorted(log_times)
    ce_storm_count = 0
    consecutive_count = 0

    for i in range(1, len(log_times)):
        if log_times[i] - log_times[i - 1] <= ce_storm_interval_seconds:
            consecutive_count += 1
        else:
            consecutive_count = 0
        if consecutive_count > ce_storm_count_threshold:
            ce_storm_count += 1
            consecutive_count = 0

    return ce_storm_count

def zrg_get_temporal_features(window_df: pd.DataFrame, time_window_size: int) -> Dict[str, int]:
    """
    获取时间特征, 包括 CE 数量、日志数量、CE 风暴数量、日志发生频率等

    :param window_df: 时间窗口内的数据
    :param time_window_size: 时间窗口大小
    :return: 时间特征

    - read_ce_log_num, read_ce_count: 时间窗口内, 读 CE 的 count 总数, 日志总数
    - scrub_ce_log_num, scrub_ce_count: 时间窗口内, 巡检 CE 的 count 总数, 日志总数
    - all_ce_log_num, all_ce_count: 时间窗口内, 所有 CE 的 count 总数, 日志总数
    - log_happen_frequency: 日志发生频率
    - ce_storm_count: CE 风暴数量
    """

    error_type_is_READ_CE = window_df["error_type_is_READ_CE"].values
    error_type_is_SCRUB_CE = window_df["error_type_is_SCRUB_CE"].values
    ce_count = window_df["Count"].values

    temporal_features = dict()
    temporal_features["read_ce_log_num"] = error_type_is_READ_CE.sum()
    temporal_features["scrub_ce_log_num"] = error_type_is_SCRUB_CE.sum()
    temporal_features["all_ce_log_num"] = len(window_df)

    temporal_features["read_ce_count"] = (
        error_type_is_READ_CE * ce_count).sum()
    temporal_features["scrub_ce_count"] = (
        error_type_is_SCRUB_CE * ce_count).sum()
    temporal_features["all_ce_count"] = ce_count.sum()

    temporal_features["log_happen_frequency"] = (
        time_window_size / len(window_df) if not window_df.empty else 0
    )
    temporal_features["ce_storm_count"] = zrg_calculate_ce_storm_count(
        window_df["LogTime"].values
    )
    return temporal_features

def zrg_get_spatio_features(window_df: pd.DataFrame) -> Dict[str, int]:
    """
    获取空间特征, 包括故障模式, 同时发生行列故障的数量

    :param window_df: 时间窗口内的数据
    :return: 空间特征

    - fault_mode_others: 其他故障, 即多个 device 发生故障
    - fault_mode_device: device 故障, 相同 id 的 device 发生多个 bank 故障
    - fault_mode_bank: bank 故障, 相同 id 的 bank 发生多个 row故障
    - fault_mode_row: row 故障, 相同 id 的 row 发生多个不同 column 的 cell 故障
    - fault_mode_column: column 故障, 相同 id 的 column 发生多个不同 row 的 cell 故障
    - fault_mode_cell: cell 故障, 发生多个相同 id 的 cell 故障
    - fault_row_num: 同时发生 row 故障的行个数
    - fault_column_num: 同时发生 column 故障的列个数
    """

    spatio_features = {
        "fault_mode_others": 0,
        "fault_mode_device": 0,
        "fault_mode_bank": 0,
        "fault_mode_row": 0,
        "fault_mode_column": 0,
        "fault_mode_cell": 0,
        "fault_row_num": 0,
        "fault_column_num": 0,
    }

    # 根据故障设备、Bank、行、列和单元的数量判断故障模式
    if zrg_unique_num_filtered(window_df["deviceID"].values) > 1:
        spatio_features["fault_mode_others"] = 1
    elif zrg_unique_num_filtered(window_df["BankId"].values) > 1:
        spatio_features["fault_mode_device"] = 1
    elif (
        zrg_unique_num_filtered(window_df["ColumnId"].values) > 1
        and zrg_unique_num_filtered(window_df["RowId"].values) > 1
    ):
        spatio_features["fault_mode_bank"] = 1
    elif zrg_unique_num_filtered(window_df["ColumnId"].values) > 1:
        spatio_features["fault_mode_row"] = 1
    elif zrg_unique_num_filtered(window_df["RowId"].values) > 1:
        spatio_features["fault_mode_column"] = 1
    elif zrg_unique_num_filtered(window_df["CellId"].values) == 1:
        spatio_features["fault_mode_cell"] = 1

    # 记录相同行对应的列地址信息
    row_pos_dict = {}
    # 记录相同列对应的行地址信息
    col_pos_dict = {}

    for device_id, bank_id, row_id, column_id in zip(
        window_df["deviceID"].values,
        window_df["BankId"].values,
        window_df["RowId"].values,
        window_df["ColumnId"].values,
    ):
        current_row = "_".join([str(pos)
                               for pos in [device_id, bank_id, row_id]])
        current_col = "_".join(
            [str(pos) for pos in [device_id, bank_id, column_id]]
        )
        row_pos_dict.setdefault(current_row, [])
        col_pos_dict.setdefault(current_col, [])
        row_pos_dict[current_row].append(column_id)
        col_pos_dict[current_col].append(row_id)

    for row in row_pos_dict:
        if zrg_unique_num_filtered(np.array(row_pos_dict[row])) > 1:
            spatio_features["fault_row_num"] += 1
    for col in col_pos_dict:
        if zrg_unique_num_filtered(np.array(col_pos_dict[col])) > 1:
            spatio_features["fault_column_num"] += 1

    return spatio_features

def zrg_get_err_parity_features(window_df: pd.DataFrame) -> Dict[str, int]:
    """
    获取奇偶校验特征

    :param window_df: 时间窗口内的数据
    :return: 奇偶校验特征

    - error_bit_count: 时间窗口内, 总错误 bit 数
    - error_dq_count: 时间窗口内, 总 dq 错误数
    - error_burst_count: 时间窗口内, 总 burst 错误数
    - max_dq_interval: 时间窗口内, 每个 parity 最大错误 dq 距离的最大值
    - max_burst_interval: 时间窗口内, 每个 parity 最大错误 burst 距离的最大值
    - dq_count=n: dq 错误数等于 n 的总数量, n 取值范围为 [1, 2, 3, 4], 默认值为 0
    - burst_count=n: burst 错误数等于 n 的总数量, n 取值范围为 [1, 2, 3, 4, 5, 6, 7, 8], 默认值为 0
    """

    err_parity_features = dict()

    # 计算总错误 bit 数、DQ 错误数和 Burst 错误数
    err_parity_features["error_bit_count"] = window_df["bit_count"].values.sum()
    err_parity_features["error_dq_count"] = window_df["dq_count"].values.sum()
    err_parity_features["error_burst_count"] = window_df["burst_count"].values.sum(
    )

    # 计算最大 DQ 间隔和最大 Burst 间隔
    err_parity_features["max_dq_interval"] = window_df[
        "max_dq_interval"
    ].values.max()
    err_parity_features["max_burst_interval"] = window_df[
        "max_burst_interval"
    ].values.max()

    # 统计 DQ 错误数和 Burst 错误数的分布
    dq_counts = dict()
    burst_counts = dict()
    for dq, burst in zip(
        window_df["dq_count"].values, window_df["burst_count"].values
    ):
        dq_counts[dq] = dq_counts.get(dq, 0) + 1
        burst_counts[burst] = burst_counts.get(burst, 0) + 1

    # 计算 'dq错误数=n' 的总数量, DDR4 内存的 DQ_COUNT 为 4, 因此 n 取值 [1,2,3,4]
    for dq in range(1, FeatureFactory.DQ_COUNT + 1):
        err_parity_features[f"dq_count={dq}"] = dq_counts.get(dq, 0)

    # 计算 'burst错误数=n' 的总数量, DDR4 内存的 BURST_COUNT 为 8, 因此 n 取值 [1,2,3,4,5,6,7,8]
    for burst in [1, 2, 3, 4, 5, 6, 7, 8]:
        err_parity_features[f"burst_count={burst}"] = burst_counts.get(
            burst, 0)

    return err_parity_features

def process_windows(windows_dir: str, processed_df_dir: str, output_dir: str, chunk_size: int = 1):
    os.makedirs(output_dir, exist_ok=True)
    buffer_dict = {"features": [],
                   "label": [],
                   "win_end_time": [],
                   "sn_name": [],
                   "lens": [],}
    chunk_number = 1
    file_list = sorted(os.listdir(windows_dir))
    for filename in tqdm(file_list, desc=f"processing {windows_dir.split('/')[-1].split('_')[0]} windows"):
        sn_name = filename.split('.')[0]
        # open json file
        with open(os.path.join(windows_dir, filename)) as win_file:
            windows_dict = json.load(win_file)
            win_file.close()
        processed_df = feather.read_dataframe(os.path.join(processed_df_dir, sn_name + '.feather'))
        
        for i in range(len(windows_dict["start_sub_win_indices"])):
            # ----------------------------- data frame process begin ----------------------------------
            win_df = processed_df.iloc[windows_dict["start_sub_win_indices"][i]:windows_dict["end_sub_win_indices"][i]]
            # ----------------------------- data list process begin ----------------------------------
            window_list = win_df.values.tolist()
            buffer_dict["features"].append(window_list)
            buffer_dict["label"].append(windows_dict["labels"][i])
            buffer_dict["win_end_time"].append(windows_dict["win_end_time"][i])
            buffer_dict["sn_name"].append(sn_name)
            buffer_dict["lens"].append(len(window_list))
            # ----------------------------- data list process endin ----------------------------------

            # ----------------------------- data list write begin ----------------------------------
            if len(buffer_dict["features"]) == chunk_size:
                output_file = os.path.join(
                    output_dir, f'chunk_{chunk_number}.json')
                with open(output_file, "w") as out_f:
                    json_str = json.dumps(buffer_dict, indent=3)
                    out_f.write(json_str)
                    out_f.close()
                buffer_dict = {
                    "features": [],
                    "label": [],
                    "win_end_time": [],
                    "sn_name": [],
                    "lens": [],
                }
            # ----------------------------- data list write endin ----------------------------------

                chunk_number += 1
    # 写入剩余数据
    if len(buffer_dict["features"]) > 0:
        output_file = os.path.join(output_dir, f'chunk_{chunk_number}.json')
        with open(output_file, "w") as out_f:
            json_str = json.dumps(buffer_dict, indent=3)
            out_f.write(json_str)
            out_f.close()
            
def process_windows_with_last_subwin_feature(windows_dir: str, processed_win1h_df_dir: str, processed_ce_df_dir: str, output_dir: str, chunk_size: int = 1):
    os.makedirs(output_dir, exist_ok=True)
    buffer_dict = {"features": [],
                   "label": [],
                   "win_end_time": [],
                   "sn_name": [],
                   "lens": [],
                   "last_15m_win_feature": [],
                   "last_30m_win_feature": [],
                   }
    chunk_number = 1
    file_list = sorted(os.listdir(windows_dir))
    for filename in tqdm(file_list, desc=f"processing {windows_dir.split('/')[-1].split('_')[0]} windows"):
        sn_name = filename.split('.')[0]
        # open json file
        with open(os.path.join(windows_dir, filename)) as win_file:
            windows_dict = json.load(win_file)
            win_file.close()
        processed_win1h_df = feather.read_dataframe(os.path.join(processed_win1h_df_dir, sn_name + '.feather'))
        # get ori ce data
        ce_df = feather.read_dataframe(os.path.join(processed_ce_df_dir, sn_name + '.feather'))
        # print(ce_df)
        ce_logtimes = ce_df["LogTime"].values
        for i in range(len(windows_dict["start_sub_win_indices"])):
            # ----------------------------- data frame process begin ----------------------------------
            win_df = processed_win1h_df.iloc[windows_dict["start_sub_win_indices"][i]:windows_dict["end_sub_win_indices"][i]]
            # ----------------------------- data list process begin ----------------------------------
            window_list = win_df.values.tolist()
            buffer_dict["features"].append(window_list)
            buffer_dict["label"].append(windows_dict["labels"][i])
            buffer_dict["win_end_time"].append(windows_dict["win_end_time"][i])
            buffer_dict["sn_name"].append(sn_name)
            buffer_dict["lens"].append(len(window_list))
            # ----------------------------- gen last win feature -----------------------------
            # 确定是从win_end_time往前取还是从last_ce_log_time往前取
            end_time = win_df["sub_win_last_ce_logtime"].values.max()
            end_index = np.searchsorted(
                        ce_logtimes,
                        end_time,
                        side="right",
                    )
            for last_win_size in [15 * ONE_MINUTE, 30 * ONE_MINUTE]:
                last_window_feature = []
                start_time = end_time - last_win_size
                start_index = np.searchsorted(
                        ce_logtimes,
                        start_time,
                        side="left",
                    )
                ce_last_win_df = ce_df.iloc[start_index:end_index]
                # -----------------ce计数
                ce_last_win_df = ce_last_win_df.assign(
                    Count=ce_last_win_df.groupby("position_and_parity")[
                        "position_and_parity"
                    ].transform("count")
                )
                # -----------------ce去重
                ce_last_win_df = ce_last_win_df.drop_duplicates(
                    subset="position_and_parity", keep="last"
                )
                if not ce_last_win_df.shape[0] > 0:
                    print(win_df)
                    print(ce_df)
                    print(start_time)
                    print(end_time)
                    print(start_index, end_index)
                    
                assert(ce_last_win_df.shape[0] > 0)
                # print(start_index, end_index)
                # print(ce_last_win_df)
                # 时间特征
                temporal_features = zrg_get_temporal_features(ce_last_win_df, last_win_size).values()
                temporal_features_list = [float(value) for value in temporal_features]
                # 空间特征
                spatio_features = zrg_get_spatio_features(ce_last_win_df).values()
                spatio_features_list = [float(value) for value in spatio_features]
                # 奇偶特征
                err_parity_features = zrg_get_err_parity_features(ce_last_win_df).values()
                err_parity_features_list = [float(value) for value in err_parity_features]
                # 总特征
                last_window_feature = temporal_features_list + spatio_features_list + err_parity_features_list
                
                if last_win_size == 15 * ONE_MINUTE:
                    buffer_dict["last_15m_win_feature"].append(last_window_feature)
                elif last_win_size == 30 * ONE_MINUTE:
                    buffer_dict["last_30m_win_feature"].append(last_window_feature)
                else:
                    assert False
            # ----------------------------- data list process endin ----------------------------------

            # ----------------------------- data list write begin ----------------------------------
            if len(buffer_dict["features"]) == chunk_size:
                output_file = os.path.join(
                    output_dir, f'chunk_{chunk_number}.json')
                with open(output_file, "w") as out_f:
                    json_str = json.dumps(buffer_dict, indent=3)
                    out_f.write(json_str)
                    out_f.close()
                buffer_dict = {
                    "features": [],
                    "label": [],
                    "win_end_time": [],
                    "sn_name": [],
                    "lens": [],
                    "last_15m_win_feature": [],
                    "last_30m_win_feature": [],
                }
                chunk_number += 1
            # ----------------------------- data list write endin ----------------------------------

    # 写入剩余数据
    if len(buffer_dict["features"]) > 0:
        output_file = os.path.join(output_dir, f'chunk_{chunk_number}.json')
        with open(output_file, "w") as out_f:
            json_str = json.dumps(buffer_dict, indent=3)
            out_f.write(json_str)
            out_f.close()


class DataGenerator(object):
    """
    数据生成器基类, 用于生成训练和测试数据
    """

    def __init__(self,
                 config: Config,
                 windows_json_files_dir: str,
                 pos_windows_json_files_dir: str,
                 neg_windows_json_files_dir: str,
                 test_windows_json_files_dir: str,
                 gen_interval_1H_for_pos_and_test: bool,):
        """
        初始化数据生成器

        :param config: 配置类实例, 包含路径、日期范围等信息
        """

        self.config = config
        self.ticket_path = self.config.ticket_path
        self.windows_json_files_dir = windows_json_files_dir
        self.pos_windows_json_files_dir = pos_windows_json_files_dir
        self.neg_windows_json_files_dir = neg_windows_json_files_dir
        self.test_windows_json_files_dir = test_windows_json_files_dir
        self.gen_interval_1H_for_pos_and_test = gen_interval_1H_for_pos_and_test

        # 将日期范围转换为时间戳
        self.train_start_date = self._datetime_to_timestamp(
            self.config.train_date_range[0]
        )
        self.train_end_date = self._datetime_to_timestamp(
            self.config.train_date_range[1]
        )
        self.test_start_date = self._datetime_to_timestamp(
            self.config.test_data_range[0]
        )
        self.test_end_date = self._datetime_to_timestamp(
            self.config.test_data_range[1])

        ticket = pd.read_csv(self.ticket_path)
        ticket = ticket[ticket["alarm_time"] <= self.train_end_date]
        ticket = ticket[ticket["alarm_time"] >= self.train_start_date]
        self.ticket = ticket
        self.ticket_sn_map = {
            sn: sn_t
            for sn, sn_t in zip(list(ticket["sn_name"]), list(ticket["alarm_time"]))
        }

    @staticmethod
    def concat_in_chunks(chunks: List) -> Union[pd.DataFrame, None]:
        """
        将 chunks 中的 DataFrame 进行拼接

        :param chunks: DataFrame 列表
        :return: 拼接后的 DataFrame, 如果 chunks 为空则返回 None
        """

        chunks = [chunk for chunk in chunks if chunk is not None]
        if chunks:
            return pd.concat(chunks)
        return None

    @staticmethod
    def _datetime_to_timestamp(date: str) -> int:
        """
        将 %Y-%m-%d 格式的日期转换为时间戳

        :param date: 日期字符串
        :return: 时间戳
        """

        return int(datetime.strptime(date, "%Y-%m-%d").timestamp())

    def get_data(self) -> pd.DataFrame:
        """
        获取 feature_path 下的所有数据, 并进行处理

        :return: 处理后的数据
        """

        file_list = os.listdir(windows_json_files_dir)
        file_list = [x for x in file_list if x.endswith(".json")]
        file_list.sort()

        if self.config.USE_MULTI_PROCESS:
            worker_num = self.config.WORKER_NUM
            with Pool(worker_num) as pool:
                list(
                    tqdm(
                        pool.imap(self._process_file, file_list),
                        total=len(file_list),
                        desc="Processing files",
                    )
                )
        else:
            for i in tqdm(range(len(file_list)), desc="Processing files"):
                self._process_file(file_list[i])

    # 给大窗口打label
    def _process_file(self, sn_file):
        """
        处理单个文件, 子类需要实现该方法
        :param sn_file: 文件名
        """
        if self.gen_interval_1H_for_pos_and_test:
            windows_dict = self._get_windows(sn_file)                    
            sn_name = os.path.splitext(sn_file)[0]
            # -------gen pos windows
            if self.ticket_sn_map.get(sn_name):
                pos_windows_dict = {"start_sub_win_indices": [],
                                    "end_sub_win_indices": [],
                                    "win_last_ce_logtime": [],
                                    "win_end_time": [],
                                    "labels": [],
                                    }
                end_time = self.ticket_sn_map.get(sn_name)
                start_time = end_time - OBS_WIN_LENGHT
                for i in range(len(windows_dict["win_last_ce_logtime"])):
                    if (windows_dict["win_last_ce_logtime"][i] <= end_time) & (windows_dict["win_last_ce_logtime"][i] >= start_time):
                        pos_windows_dict["start_sub_win_indices"].append(
                            windows_dict["start_sub_win_indices"][i])
                        pos_windows_dict["end_sub_win_indices"].append(
                            windows_dict["end_sub_win_indices"][i])
                        pos_windows_dict["win_last_ce_logtime"].append(
                            windows_dict["win_last_ce_logtime"][i])
                        pos_windows_dict["win_end_time"].append(
                            windows_dict["win_end_time"][i])
                        pos_windows_dict["labels"].append(1)
                        

                if len(pos_windows_dict["start_sub_win_indices"]) > 0:
                    with open(os.path.join(self.pos_windows_json_files_dir, sn_file.split(".")[0] + '.json'), "w") as zrg_file:
                        json_str = json.dumps(pos_windows_dict, indent=3)
                        zrg_file.write(json_str)
                        zrg_file.close()
            # -------gen test windows
            test_windows_dict =  {"start_sub_win_indices": [],
                                    "end_sub_win_indices": [],
                                    "win_last_ce_logtime": [],
                                    "win_end_time": [],
                                    "labels": [],
                                    }

            for i in range(len(windows_dict["win_last_ce_logtime"])):
                if (windows_dict["win_last_ce_logtime"][i] <= self.test_end_date) & (windows_dict["win_last_ce_logtime"][i] >= self.test_start_date):
                    test_windows_dict["start_sub_win_indices"].append(
                        windows_dict["start_sub_win_indices"][i])
                    test_windows_dict["end_sub_win_indices"].append(
                        windows_dict["end_sub_win_indices"][i])
                    test_windows_dict["win_last_ce_logtime"].append(
                        windows_dict["win_last_ce_logtime"][i])
                    test_windows_dict["win_end_time"].append(
                        windows_dict["win_end_time"][i])
                    test_windows_dict["labels"].append(0)

            if len(test_windows_dict["start_sub_win_indices"]) > 0:
                with open(os.path.join(test_windows_json_files_dir, sn_file.split(".")[0] + '.json'), "w") as zrg_file:
                    json_str = json.dumps(test_windows_dict, indent=3)
                    zrg_file.write(json_str)
                    zrg_file.close()
        else:
            windows_dict = self._get_windows(sn_file)                    
            sn_name = os.path.splitext(sn_file)[0]
            # -------gen pos windows
            if self.ticket_sn_map.get(sn_name):
                pos_windows_dict = {"start_sub_win_indices": [],
                                    "end_sub_win_indices": [],
                                    "win_last_ce_logtime": [],
                                    "win_end_time": [],
                                    "labels": [],
                                    }
                end_time = self.ticket_sn_map.get(sn_name)
                start_time = end_time - OBS_WIN_LENGHT
                for i in range(len(windows_dict["win_last_ce_logtime"])):
                    if (windows_dict["win_last_ce_logtime"][i] <= end_time) & (windows_dict["win_last_ce_logtime"][i] >= start_time):
                        pos_windows_dict["start_sub_win_indices"].append(
                            windows_dict["start_sub_win_indices"][i])
                        pos_windows_dict["end_sub_win_indices"].append(
                            windows_dict["end_sub_win_indices"][i])
                        pos_windows_dict["win_last_ce_logtime"].append(
                            windows_dict["win_last_ce_logtime"][i])
                        pos_windows_dict["win_end_time"].append(
                            windows_dict["win_end_time"][i])
                        pos_windows_dict["labels"].append(1)
                        

                if len(pos_windows_dict["start_sub_win_indices"]) > 0:
                    with open(os.path.join(self.pos_windows_json_files_dir, sn_file.split(".")[0] + '.json'), "w") as zrg_file:
                        json_str = json.dumps(pos_windows_dict, indent=3)
                        zrg_file.write(json_str)
                        zrg_file.close()
                        
            # -------gen neg windows
            else:
                neg_windows_dict = {"start_sub_win_indices": [],
                                    "end_sub_win_indices": [],
                                    "win_last_ce_logtime": [],
                                    "win_end_time": [],
                                    "labels": [],
                                    }
                # 设负样本的时间范围为某段连续的 30 天
                end_time = self.train_end_date - 30 * ONE_DAY
                start_time = end_time - NEG_GEN_INTERVAL

                for i in range(len(windows_dict["win_last_ce_logtime"])):
                    if (windows_dict["win_last_ce_logtime"][i] <= end_time) & (windows_dict["win_last_ce_logtime"][i] >= start_time):
                        neg_windows_dict["start_sub_win_indices"].append(
                            windows_dict["start_sub_win_indices"][i])
                        neg_windows_dict["end_sub_win_indices"].append(
                            windows_dict["end_sub_win_indices"][i])
                        neg_windows_dict["win_last_ce_logtime"].append(
                            windows_dict["win_last_ce_logtime"][i])
                        neg_windows_dict["win_end_time"].append(
                            windows_dict["win_end_time"][i])
                        neg_windows_dict["labels"].append(0)

                if len(neg_windows_dict["start_sub_win_indices"]) > 0:
                    with open(os.path.join(neg_windows_json_files_dir, sn_file.split(".")[0] + '.json'), "w") as zrg_file:
                        json_str = json.dumps(neg_windows_dict, indent=3)
                        zrg_file.write(json_str)
                        zrg_file.close()
                        
            # -------gen test windows
            test_windows_dict =  {"start_sub_win_indices": [],
                                    "end_sub_win_indices": [],
                                    "win_last_ce_logtime": [],
                                    "win_end_time": [],
                                    "labels": [],
                                    }

            for i in range(len(windows_dict["win_last_ce_logtime"])):
                if (windows_dict["win_last_ce_logtime"][i] <= self.test_end_date) & (windows_dict["win_last_ce_logtime"][i] >= self.test_start_date):
                    test_windows_dict["start_sub_win_indices"].append(
                        windows_dict["start_sub_win_indices"][i])
                    test_windows_dict["end_sub_win_indices"].append(
                        windows_dict["end_sub_win_indices"][i])
                    test_windows_dict["win_last_ce_logtime"].append(
                        windows_dict["win_last_ce_logtime"][i])
                    test_windows_dict["win_end_time"].append(
                        windows_dict["win_end_time"][i])
                    test_windows_dict["labels"].append(0)

            if len(test_windows_dict["start_sub_win_indices"]) > 0:
                with open(os.path.join(test_windows_json_files_dir, sn_file.split(".")[0] + '.json'), "w") as zrg_file:
                    json_str = json.dumps(test_windows_dict, indent=3)
                    zrg_file.write(json_str)
                    zrg_file.close()


    def _get_processed_df(self, sn_file: str) -> pd.DataFrame:
        assert (os.path.exists(os.path.join(
            processed_win1h_df_files_dir, sn_file.split('.')[0] + '.feather')))
        processed_df = feather.read_dataframe(os.path.join(
            processed_win1h_df_files_dir, sn_file.split('.')[0] + '.feather'))
        return processed_df

    def _get_windows(self, sn_file: str) -> Dict:
        assert (os.path.exists(os.path.join(
            windows_json_files_dir, sn_file.split('.')[0] + '.json')))
        with open(os.path.join(windows_json_files_dir, sn_file.split('.')[0] + '.json'), "r") as win_file:
            windows_dict = json.load(win_file)
            win_file.close()
            return windows_dict


'''
目标：提取1Hour的子窗口特征（作为token），大窗口长度为72Hour（1 Seq： 72 tokens），大窗口步长：6Hour
方案1：
    原始数据(单个Sn的feather文件) -> 按照1Hour子窗口提取特征的feather文件（读写快速） -> 大窗口的起止index（json或feather）
        优点：feather读写快速，节省空间
             可以修快大窗口长度或步长
        缺点：加载时读俩文件
方案二：
    直接生成大窗口的所有特征的json文件
        优点：只读取一个数据文件
        缺点：窗口重合度高，占存储空间
'''


if __name__ == "__main__":
    # -------------获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--sn_type', choices=["A", "B"], default="A")
    parser.add_argument('--test_stage', type=int, default=1)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--ticket_path', type=str, required=True)
    parser.add_argument('--s1_train_start_month', type=str, default="01-01")
    parser.add_argument('--s1_train_end_month', type=str, default="06-01")
    parser.add_argument('--s1_test_start_month', type=str, default="06-01")
    parser.add_argument('--s1_test_end_month', type=str, default="08-01")
    parser.add_argument('--gen_pos', action='store_true')
    parser.add_argument('--gen_neg', action='store_true')
    parser.add_argument('--gen_test', action='store_true')
    parser.add_argument('--gen_interval_1H_for_pos_and_test', action='store_true')
    args = parser.parse_args()

    # -------------判断输入参数是否合理
    # if args.gen_test and (args.gen_pos or args.gen_neg):
    #     assert False
    # if args.gen_test and args.gen_interval_1H_for_pos_and_test:
    #     assert False
    # if args.gen_interval_1H_for_pos_and_test and (args.gen_pos or args.gen_neg):
    #     assert False

    # -------------设置输出路径
    processed_win1h_df_files_dir = f"/mnt/zhangrengang/data/STIM_Data_test_05-06/STIM_processed_win1h_df_type{args.sn_type}"
    processed_ce_df_files_dir = f"/mnt/zhangrengang/data/STIM_Data_test_05-06/STIM_processed_ce_df_type{args.sn_type}"
    # TODO 
    output_feature_dir = f"/mnt/zhangrengang/data/STIM_Data_train_01-04/STIM_win_feature_obv30d_neg60d_{args.sn_type}/"
    # output_feature_dir = f"/mnt/zhangrengang/data/STIM_Data_val_04-06/STIM_win_feature_obv30d_neg30d_{args.sn_type}/"

    if args.gen_pos or args.gen_neg:
        windows_json_files_dir = output_feature_dir + "windows"
    elif args.gen_interval_1H_for_pos_and_test:
        windows_json_files_dir = output_feature_dir + "windows_interval_1H"
    else:
        assert False

    pos_windows_json_files_dir = output_feature_dir + "pos_windows"
    test_windows_json_files_dir = output_feature_dir + "test_windows"
    if args.gen_interval_1H_for_pos_and_test:
        pos_windows_json_files_dir = output_feature_dir + "aug_pos_windows"
        test_windows_json_files_dir = output_feature_dir + "aug_test_windows"
    neg_windows_json_files_dir = output_feature_dir + "neg_windows"
    

    processed_pos_windows_dir = output_feature_dir + "pos_windows_feature"
    processed_test_windows_dir = output_feature_dir + "test_windows_feature"
    if args.gen_interval_1H_for_pos_and_test:
        processed_pos_windows_dir = output_feature_dir + "aug_pos_windows_feature"
        processed_test_windows_dir = output_feature_dir + "aug_test_windows_feature"
    processed_neg_windows_dir = output_feature_dir + "neg_windows_feature"
    
    os.makedirs(processed_ce_df_files_dir, exist_ok=True)
    os.makedirs(processed_win1h_df_files_dir, exist_ok=True)
    os.makedirs(windows_json_files_dir, exist_ok=True)
    os.makedirs(pos_windows_json_files_dir, exist_ok=True)
    os.makedirs(neg_windows_json_files_dir, exist_ok=True)
    os.makedirs(test_windows_json_files_dir, exist_ok=True)
    os.makedirs(processed_pos_windows_dir, exist_ok=True)
    os.makedirs(processed_neg_windows_dir, exist_ok=True)
    os.makedirs(processed_test_windows_dir, exist_ok=True)

    # -------------配置参数Config
    sn_type = args.sn_type  # SN 类型, A 或 B, 这里以 A 类型为例
    test_stage = args.test_stage  # 测试阶段, 1 或 2, 这里以 Stage 1 为例
    # 设置训练数据的时间范围
    train_data_range: tuple = (
        f"2024-{args.s1_train_start_month}", f"2024-{args.s1_train_end_month}")
    # 根据测试阶段设置测试数据的时间范围
    if test_stage == 1:
        test_data_range: tuple = (
            # 第一阶段测试数据范围
            f"2024-{args.s1_test_start_month}", f"2024-{args.s1_test_end_month}")
    else:
        test_data_range: tuple = ("2024-08-01", "2024-10-01")  # 第二阶段测试数据范围

    # 初始化配置类 Config，设置数据路径、特征路径、训练数据路径、测试数据路径等
    config = Config(
        data_path=os.path.join(args.data_path, f"type_{sn_type}"),  # 原始数据集路径
        train_date_range=train_data_range,
        test_data_range=test_data_range,  # 测试数据时间范围
        ticket_path=args.ticket_path,  # 维修单路径
    )

    # 初始化特征工厂类 FeatureFactory，用于处理 SN 文件并生成特征
    feature_factory = FeatureFactory(config, processed_win1h_df_files_dir, processed_ce_df_files_dir, windows_json_files_dir)
        
    feature_factory.process_all_sn()  # 处理所有 SN 文件

    # 生成正负样本、测试样本的的window和label
    label_gennerator = DataGenerator(config, windows_json_files_dir, pos_windows_json_files_dir,
                                     neg_windows_json_files_dir, test_windows_json_files_dir, args.gen_interval_1H_for_pos_and_test)
    label_gennerator.get_data()

    # # 初始化正样本数据生成器，生成并保存正样本数据
    if args.gen_pos and not args.gen_interval_1H_for_pos_and_test:
        process_windows_with_last_subwin_feature(pos_windows_json_files_dir,
                        processed_win1h_df_files_dir, processed_ce_df_files_dir, processed_pos_windows_dir)

    # # 初始化负样本数据生成器，生成并保存负样本数据
    if args.gen_neg and not args.gen_interval_1H_for_pos_and_test:
        process_windows_with_last_subwin_feature(neg_windows_json_files_dir,
                        processed_win1h_df_files_dir, processed_ce_df_files_dir, processed_neg_windows_dir)

    # 初始化测试数据生成器，生成并保存测试数据
    if args.gen_test and not args.gen_interval_1H_for_pos_and_test:
        process_windows_with_last_subwin_feature(test_windows_json_files_dir,
                        processed_win1h_df_files_dir, processed_ce_df_files_dir, processed_test_windows_dir, chunk_size=256)
    
    if args.gen_interval_1H_for_pos_and_test:
        process_windows_with_last_subwin_feature(pos_windows_json_files_dir,
                        processed_win1h_df_files_dir, processed_ce_df_files_dir, processed_pos_windows_dir, chunk_size=1)
        process_windows_with_last_subwin_feature(test_windows_json_files_dir,
                        processed_df_files_dir, processed_ce_df_files_dir, processed_test_windows_dir, chunk_size=256)