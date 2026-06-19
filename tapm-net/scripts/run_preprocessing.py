# scripts/run_preprocessing.py
import os
import argparse
import sys

# 将项目根目录加入模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import build_cleaned_battery_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="电池老化原始数据清洗流水线")
    parser.add_argument('--raw_dir', type=str, default='data/raw', help='原始数据所在的文件夹路径')
    parser.add_argument('--out_path', type=str, default='data/processed/TJU_NCA_cleaned_dataset.csv',
                        help='保存清洗后特征的CSV路径')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    try:
        data = build_cleaned_battery_dataset(args.raw_dir)
        data.to_csv(args.out_path, index=False)
        print(f"数据处理成功，文件保存在: {args.out_path}")
    except Exception as e:
        print(f"处理数据失败: {str(e)}")