# TAPM-Net: Tri-Channel Adaptive Physical Modulation Network for Battery SOH and EOL Prediction

这是一个基于物理调制与多通道特征融合的锂离子电池健康状态（SOH）预测与寿命终止点（EOL）外推推演项目。模型融合了时序统计与电池退化物理特征，并结合 FiLM (Feature Modulation) 机制整合了环境温度和充放电倍率等静态工况。

---

##  项目结构

```text
tapm-net/
│
├── data/                      # 数据集
│   ├── raw/                   # 请在此处放置未处理的 .csv 电池文件
│   └── processed/             # 存放清洗后的时序特征文件
│
├── configs/                   
│   └── config.py              # 项目模型参数、路径及特征配置
│
├── src/                       
│   ├── preprocess.py          # 数据物理剪裁与 Hampel 时序异常检测
│   ├── dataset.py             # 物理通道时序构造与归一化
│   ├── models.py              # 三通道自适应网络 (TAPM-Net)
│   └── utils.py               # 结果可视化辅助工具
│
├── scripts/                   
│   ├── run_preprocessing.py   # 数据清洗与重构入口
│   └── train.py               # 模型训练、指标评估与 EOL 外推外推验证
│
├── requirements.txt           # 基础依赖依赖包
└── README.md                  # 说明文档