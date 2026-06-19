# src/utils.py
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_cleaned_cell(data_df, cell_id):
    """从合并后的 DataFrame 中绘制指定电池的 8 个退化物理特征趋势图"""
    cell_data = data_df[data_df['cell_id'] == cell_id]
    if cell_data.empty:
        print(f"未找到对应的电池 ID: {cell_id}")
        return

    cycles = cell_data['cycle'].values
    fig, axs = plt.subplots(4, 2, figsize=(14, 16))
    axs = axs.ravel()

    features_config = [
        ('SOH', '健康状态 SOH Decline (%) - Cleaned', 'blue', False),
        ('CE', '库仑效率 Coulombic Efficiency (CE, %) - Cleaned', 'green', False),
        ('V_mid', '放电中段电压 Discharge Mid-point Voltage (V_mid, V) - Cleaned', 'orange', False),
        ('P_CV', '恒压充电容量占比 CV Charge Capacity Ratio (P_CV, %)', 'red', False),
        ('R0', '欧姆内阻 Ohmic Resistance (R0, Ohm)', 'purple', True),
        ('t_CC', '恒流充电时长 CC Charge Time (t_CC, seconds)', 'navy', False),
        ('V_relax_drop', '电压弛豫跌落值 Voltage Relaxation Drop (V_relax_drop, V)', 'teal', False),
        ('t_CV', '恒压充电时长 CV Charge Time (t_CV, seconds)', 'magenta', False)
    ]

    for idx, (col_name, title, color, draw_ma) in enumerate(features_config):
        y_values = cell_data[col_name].values
        if draw_ma:
            axs[idx].scatter(cycles, y_values, color=color, alpha=0.3, s=12, label='Raw calculated R0')
            smooth_y = pd.Series(y_values).rolling(window=5, min_periods=1).mean()
            axs[idx].plot(cycles, smooth_y, color='red', linewidth=1.5, label='5-cycle MA')
            axs[idx].legend(loc='upper left')
        else:
            axs[idx].plot(cycles, y_values, color=color, linewidth=1.5)

        axs[idx].set_title(title)
        axs[idx].grid(True)
        axs[idx].set_xlabel('Cycle Number')

    temp = cell_data['temperature'].iloc[0]
    rate = cell_data['charge_rate'].iloc[0]
    plt.suptitle(f'Cleaned 4x2 Features of {cell_id} ({temp}℃, {rate}C)', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()