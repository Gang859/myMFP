import argparse
import pandas as pd
from pathlib import Path
import sys
import glob

def convert_feather_to_csv(input_path, output_path):
    """
    将.feather文件转换为.csv文件
    参数：
        input_path: 输入的.feather文件路径
        output_path: 输出的.csv文件路径
    """
    try:
        # 读取.feather文件
        df = pd.read_feather(input_path)
        
        # 保存为CSV（默认UTF-8编码）
        df.to_csv(output_path, index=False)
        
        print(f"转换成功！保存至：{output_path}")
        print(f"数据维度：{df.shape[0]}行 × {df.shape[1]}列")

    except FileNotFoundError:
        print(f"错误：输入文件不存在 - {input_path}")
    except Exception as e:
        print(f"转换失败：{str(e)}")

if __name__ == "__main__":
    for feather_file in glob.glob("/mnt/zhangrengang/workspace/myMFP/sample_data/release_features/combined_sn_feature/type_A/*.feather"):
        csv_file = feather_file.replace(".feather", ".csv")
        csv_file = csv_file.replace("type_A", "type_A_csv")
        convert_feather_to_csv(feather_file, csv_file)
        
    # # 设置命令行参数
    # parser = argparse.ArgumentParser(description='将.feather文件转换为.csv文件')
    # parser.add_argument('--input', type=str, required=True,
    #                     help='输入的.feather文件路径')
    # parser.add_argument('--output', type=str,
    #                     help='输出的.csv文件路径（可选）')
    
    # args = parser.parse_args()
    
    # # 自动生成输出路径（如果未指定）
    # if not args.output:
    #     input_path = Path(args.input)
    #     output_path = input_path.with_suffix('.csv')
    # else:
    #     output_path = Path(args.output)
    
    # # 检查文件扩展名
    # if not args.input.lower().endswith('.feather'):
    #     print("错误：输入文件必须是.feather格式")
    #     sys.exit(1)
        
    # # 执行转换
    # convert_feather_to_csv(args.input, output_path)