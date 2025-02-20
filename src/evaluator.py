import pandas as pd
from datetime import datetime


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
    cutoff_timestamp = int(datetime.strptime(cutoff_date, "%Y-%m-%d").timestamp())

    # 读取文件
    sb_df = pd.read_csv(submission_path)
    ft_df = pd.read_csv(failure_ticket_path)

    # 转换时间列为时间戳
    sb_df['prediction_timestamp'] = sb_df['prediction_timestamp'].astype('int64')
    ft_df['alarm_time'] = ft_df['alarm_time'].astype('int64')

    # 过滤掉截止日期之前的记录
    ft_df = ft_df[ft_df['alarm_time'] >= cutoff_timestamp]

    # 获取唯一的故障记录
    unique_failures = ft_df['sn_name'].unique()

    # 初始化计数器
    tp = 0
    fp = 0
    fn = 0

    # 计算TP和FN
    for sn in unique_failures:
        # 获取该机器的所有故障时间
        failure_times = ft_df[ft_df['sn_name'] == sn]['alarm_time'].values

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
                if window_start <= pred_time <= window_end:
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
    actual_failures = set(unique_failures)
    false_predictions = predicted_machines - actual_failures
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
        'fn': fn
    }


if __name__ == "__main__":
    # 设置输入参数
    submission_file = "submission.csv"
    failure_ticket_file = "failure_ticket.csv"
    # 只使用故障文件中cutoff_date之后的数据
    cutoff_date = "2024-05-01"

    # 计算F1分数
    results = calculate_f1_score(submission_file, failure_ticket_file, cutoff_date)

    # 打印结果
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"True Positives: {results['tp']}")
    print(f"False Positives: {results['fp']}")
    print(f"False Negatives: {results['fn']}")