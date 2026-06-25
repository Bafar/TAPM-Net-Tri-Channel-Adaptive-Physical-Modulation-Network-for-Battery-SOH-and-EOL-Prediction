# src/preprocess.py
import os
import re
import numpy as np
import pandas as pd

def parse_filename_metadata(filename):
    """从文件名中解析温度和充电倍率"""
    try:
        temp = float(re.search(r'CY(\d+)', filename).group(1))
        rate_str = re.search(r'CY\d+-(.+?)_', filename).group(1)
        if rate_str.startswith('0'):
            charge_rate = float(rate_str) / 10.0 if len(rate_str) == 2 else float(rate_str[0] + '.' + rate_str[1:])
        else:
            charge_rate = float(rate_str)
        return temp, charge_rate
    except Exception:
        return 25.0, 1.0  # 默认容错值

def hampel_filter_pandas(series, window=11, n_sigmas=3):
    """高性能向量化 Hampel 滤波器，用于检测并修复时序异常值"""
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    rolling_mad = (series - rolling_median).abs().rolling(window=window, center=True, min_periods=1).median()
    threshold = n_sigmas * 1.4826 * rolling_mad
    difference = (series - rolling_median).abs()

    outlier_mask = difference > threshold
    cleaned_series = series.copy()
    cleaned_series[outlier_mask] = rolling_median[outlier_mask]
    return cleaned_series

def build_cleaned_battery_dataset(folder_path):
    """遍历文件夹构建清洗后的电池数据集"""
    all_cells_data = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"未找到指定的原始数据目录: {folder_path}")

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('CY')]
    print(f"检测到 {len(files)} 个老化数据文件，开始提取特征与清洗...")

    for file in files:
        file_full_path = os.path.join(folder_path, file)
        cell_id = file.replace('.csv', '')
        temp, charge_rate = parse_filename_metadata(file)

        df = pd.read_csv(file_full_path)
        df.columns = df.columns.str.strip()

        current_col = '<I>/mA'
        voltage_col = 'Ecell/V'
        cycle_col = 'cycle number'
        time_col = 'time/s'
        q_discharge_col = 'Q discharge/mA.h' if 'Q discharge/mA.h' in df.columns else 'Q discharge/mA. h'
        q_charge_col = 'Q charge/mA.h' if 'Q charge/mA.h' in df.columns else 'Q charge/mA. h'

        # 1. 估算欧姆内阻 R0
        I_raw = df[current_col].values
        V_raw = df[voltage_col].values
        cycle_raw = df[cycle_col].values
        r0_dict = {}
        for i in range(1, len(df)):
            if I_raw[i] < -500 and I_raw[i-1] >= -10:
                v1, v2 = V_raw[i-1], V_raw[i]
                i1, i2 = I_raw[i-1], I_raw[i]
                c_num = cycle_raw[i]
                delta_v = v2 - v1
                delta_i = (i2 - i1) / 1000.0
                if delta_i != 0:
                    r0 = abs(delta_v / delta_i)
                    if 0.001 < r0 < 0.5:
                        r0_dict[c_num] = r0

        grouped = df.groupby(cycle_col)
        # 基准放电容量 Q0 估算
        q0_list = [grouped.get_group(c)[q_discharge_col].max() for c in [2, 3, 4] if c in grouped.groups]
        Q0 = np.mean(q0_list) if len(q0_list) > 0 else 3200.0

        cell_records = []
        for cycle_num, group in grouped:
            if cycle_num < 2:
                continue

            max_disch = group[q_discharge_col].max()
            max_charge = group[q_charge_col].max()

            soh = (max_disch / Q0) * 100
            ce = (max_disch / max_charge) * 100 if max_charge > 0 else np.nan

            disch_segment = group[group[current_col] < -100]
            v_mid = disch_segment[voltage_col].median() if not disch_segment.empty else np.nan

            cv_segment = group[group['control/V'] > 4.19]
            p_cv = np.nan
            if not cv_segment.empty and max_charge > 0:
                q_cv_start = cv_segment[q_charge_col].min()
                q_cv_end = cv_segment[q_charge_col].max()
                p_cv = ((q_cv_end - q_cv_start) / max_charge) * 100

            cc_segment = group[(group[current_col] > 100) & (group['control/V'] < 4.19)]
            t_cc = cc_segment[time_col].max() - cc_segment[time_col].min() if not cc_segment.empty else np.nan

            rest_after_charge = group[(group[current_col] == 0) & (group[q_charge_col] > 0.9 * max_charge)]
            v_relax_drop = rest_after_charge[voltage_col].max() - rest_after_charge[voltage_col].min() if not rest_after_charge.empty else np.nan

            cv_time_segment = group[(group['control/V'] > 4.19) & (group[current_col] > 20)]
            t_cv = cv_time_segment[time_col].max() - cv_time_segment[time_col].min() if not cv_time_segment.empty else np.nan

            r0 = r0_dict.get(cycle_num, np.nan)

            cell_records.append({
                'cycle': cycle_num,
                'SOH': soh,
                'CE': ce,
                'V_mid': v_mid,
                'P_CV': p_cv,
                'R0': r0,
                't_CC': t_cc,
                'V_relax_drop': v_relax_drop,
                't_CV': t_cv
            })

        df_cell = pd.DataFrame(cell_records)

        # 物理剪裁合理范围
        df_cell.loc[(df_cell['SOH'] < 0) | (df_cell['SOH'] > 110), 'SOH'] = np.nan
        df_cell.loc[(df_cell['CE'] < 95.0) | (df_cell['CE'] > 102.0), 'CE'] = np.nan
        df_cell.loc[(df_cell['V_mid'] < 3.0) | (df_cell['V_mid'] > 3.6), 'V_mid'] = np.nan
        df_cell.loc[(df_cell['P_CV'] < 2.0) | (df_cell['P_CV'] > 60.0), 'P_CV'] = np.nan
        df_cell.loc[(df_cell['R0'] < 0.0001) | (df_cell['R0'] > 0.05), 'R0'] = np.nan
        df_cell.loc[(df_cell['t_CC'] < 500) | (df_cell['t_CC'] > 20000), 't_CC'] = np.nan
        df_cell.loc[(df_cell['V_relax_drop'] < 0.001) | (df_cell['V_relax_drop'] > 0.1), 'V_relax_drop'] = np.nan
        df_cell.loc[(df_cell['t_CV'] < 500) | (df_cell['t_CV'] > 20000), 't_CV'] = np.nan

        # 插值重构与滤波
        df_cell = df_cell.interpolate(method='linear', limit_direction='both')
        for col in ['SOH', 'CE', 'V_mid', 'P_CV', 'R0', 't_CC', 'V_relax_drop', 't_CV']:
            df_cell[col] = hampel_filter_pandas(df_cell[col], window=11, n_sigmas=3)
        df_cell = df_cell.interpolate(method='linear', limit_direction='both')

        df_cell.insert(0, 'cell_id', cell_id)
        df_cell.insert(1, 'temperature', temp)
        df_cell.insert(2, 'charge_rate', charge_rate)

        all_cells_data.append(df_cell)
        print(f"-> 电池 {cell_id} 处理完成. 提取圈数: {len(df_cell)}")

    return pd.concat(all_cells_data, ignore_index=True)