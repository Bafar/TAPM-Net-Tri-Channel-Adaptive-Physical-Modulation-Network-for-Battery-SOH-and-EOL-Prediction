# configs/config.py
import torch

class TAPMConfig:
    # 路径配置
    raw_data_dir = "data/raw"
    cleaned_csv_path = "data/processed/TJU_NCA_cleaned_dataset.csv"
    model_save_path = "tapm_net_model.pth"

    # 物理通道与观察窗口
    history_window = 30     # 历史观察窗口 (前 M 圈)
    latent_dim = 16         # 每个物理通道隐向量 z 的维度
    feature_dropout = 0.1   # 特征级容错 Dropout 概率

    # 特征分组配置
    target_col = 'SOH'
    pos_exogenous_cols = ['CE', 'V_mid', 't_CC']
    neg_exogenous_cols = ['R0', 'P_CV', 'V_relax_drop', 't_CV']
    static_cols = ['temperature', 'charge_rate']

    # 自动计算特征维度
    pos_features_num = len(pos_exogenous_cols)
    neg_features_num = len(neg_exogenous_cols)
    static_features = len(static_cols)

    # 数据划分
    test_cells = ['CY35-05_1-#2', 'CY25-025_1-#2']
    train_cells = []  # 运行时动态填充

    # 训练超参数
    epochs = 100
    batch_size = 128
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')