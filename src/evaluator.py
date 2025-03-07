import pandas as pd
from datetime import datetime
import argparse

ONE_DAY = 24 * 3600
ONE_HOUR = 3600
ONE_MINUTE = 60

def calculate_f1_score(submission_path, failure_ticket_path, cutoff_date):
    """
    计算预测结果的F1分数

    参数:
    submission_path: 提交文件的路径
    failure_ticket_path: 故障记录文件的路径
    cutoff_date: 截止日期，格式为"YYYY-MM-DD"

    返回:
    dict: 包含precision, recall, f1_score的字典
    """
    # 将截止日期转换为时间戳
    print("-------------sn_level---------------")
    cutoff_timestamp = int(datetime.strptime(cutoff_date, "%Y-%m-%d").timestamp())

    # 读取文件
    sb_df = pd.read_csv(submission_path)
    ft_df = pd.read_csv(failure_ticket_path)

    # 转换时间列为时间戳
    sb_df['prediction_timestamp'] = sb_df['prediction_timestamp'].astype('int64')
    ft_df['alarm_time'] = ft_df['alarm_time'].astype('int64')
    
    # 过滤掉截止日期之前的记录
    test_ft_df = ft_df[ft_df['alarm_time'] >= cutoff_timestamp]
    trainandval_ft_df = ft_df[ft_df['alarm_time'] < cutoff_timestamp]
    
    # 处理sbdf,忽略末尾时间的预测
    test_end_time = "2024-06-01"
    pred_test_end_time = int(datetime.strptime(test_end_time, "%Y-%m-%d").timestamp()) - 7 * 24 * 3600
    sb_df = sb_df[sb_df['prediction_timestamp'] < pred_test_end_time]
    
    
    # 获取唯一的故障记录
    unique_failures = test_ft_df['sn_name'].unique()

    # 初始化计数器
    tp = 0
    fp = 0
    fn = 0

    # 计算TP和FN
    for sn in unique_failures:
        # 获取该机器的所有故障时间
        failure_times = test_ft_df[test_ft_df['sn_name'] == sn]['alarm_time'].values

        # 获取该机器的所有预测时间
        predictions = sb_df[sb_df['sn_name'] == sn]['prediction_timestamp'].values

        if len(predictions) == 0:
            # 如果没有预测记录，增加FN
            fn += 1
            continue

        # 检查是否有预测落在任何故障的有效时间窗口内
        prediction_success = False
        for failure_time in failure_times:
            window_start = failure_time - 7 * 24 * 3600  # 前7天
            window_end = failure_time - 15 * 60  # 前15分钟

            for pred_time in predictions:
                if True: #window_start <= pred_time <= window_end:
                    prediction_success = True
                    break

            if prediction_success:
                break

        if prediction_success:
            tp += 1
        else:
            fn += 1

    # 计算FP
    predicted_machines = set(sb_df['sn_name'].unique())
    trainandval_ue_machines = set(trainandval_ft_df['sn_name'].unique())
    print("len(trainandval_ue_machines)", len(trainandval_ue_machines))
    print(predicted_machines.intersection(trainandval_ue_machines))
    
    actual_failures = set(unique_failures)
    false_predictions = (predicted_machines - trainandval_ue_machines) - actual_failures
    # false_predictions = predicted_machines - actual_failures
    fp = len(false_predictions)

    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }

def calculate_win_level_fp_and_tp(submission_path, failure_ticket_path, cutoff_date):
    # 将截止日期转换为时间戳
    cutoff_timestamp = int(datetime.strptime(cutoff_date, "%Y-%m-%d").timestamp())

    # 读取文件
    sb_df = pd.read_csv(submission_path)
    ft_df = pd.read_csv(failure_ticket_path)

    # 转换时间列为时间戳
    sb_df['prediction_timestamp'] = sb_df['prediction_timestamp'].astype('int64')
    ft_df['alarm_time'] = ft_df['alarm_time'].astype('int64')
    
    # 过滤掉截止日期之前的记录
    test_ft_df = ft_df[ft_df['alarm_time'] >= cutoff_timestamp]
    trainandval_ft_df = ft_df[ft_df['alarm_time'] < cutoff_timestamp]
    
    # 获取唯一的故障记录
    unique_failures = test_ft_df['sn_name'].unique()
    ue_sn_name_list = test_ft_df['sn_name'].values
    ticket_sn_map = {
        sn: sn_t
        for sn , sn_t in zip(list(ft_df["sn_name"]), list(ft_df["alarm_time"]))
    }

    # 初始化计数器
    tp = 0
    fp = 0
    right_sn_but_wrong_time = 0

    # 计算TP和FN
    for index, sb_row in sb_df.iterrows():
        pred_sn_name = sb_row["sn_name"]
        pred_time = sb_row["prediction_timestamp"]
        if pred_sn_name in ue_sn_name_list:
            ue_time = ticket_sn_map[pred_sn_name]
            if pred_time > ue_time - 7 * ONE_DAY and pred_time < ue_time - 15 * ONE_MINUTE:
                tp += 1
            else:
                right_sn_but_wrong_time += 1
                fp += 1
        else:
            fp += 1
            
    print(f"------win level-----: \nfp: {fp}, right_sn_but_wrong_time: {right_sn_but_wrong_time}, tp: {tp} | total: {sb_df.shape[0]}")
    precision = float(tp) / (tp + fp)
    print(f"precision: {precision: .4f}")
    
# test
if __name__ == "__main__":
    # 设置输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_file", type=str, required=True)
    parser.add_argument("--ticket_file", type=str, default="/home/zhangrengang/Doc/competition_data/stage1_feather/ticket.csv")
    parser.add_argument("--cutoff_date", type=str, default="05")
    
    args = parser.parse_args()
    
    submission_file = args.submission_file
    failure_ticket_file = args.ticket_file
    # 只使用故障文件中cutoff_date之后的数据
    cutoff_date = f"2024-{args.cutoff_date}-01"

    # 计算F1分数
    calculate_win_level_fp_and_tp(submission_file, failure_ticket_file, cutoff_date)
    results = calculate_f1_score(submission_file, failure_ticket_file, cutoff_date)

    # 打印结果
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"True Positives: {results['tp']}")
    print(f"False Positives: {results['fp']}")
    print(f"False Negatives: {results['fn']}")
    print(f"[\t {results['tp']} \t {results['fp']} \n \t {results['fn']} \t TN ]")
