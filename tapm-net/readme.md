# TAPM-Net-Tri-Channel-Adaptive-Physical-Modulation-Network-for-Battery-SOH-and-EOL-Prediction
这是一个基于物理调制与多通道特征融合的锂离子电池健康状态（SOH）预测与寿命终止点（EOL）外推推演项目。模型融合了时序统计与电池退化物理特征，并结合 FiLM (Feature Modulation) 机制整合了环境温度和充放电倍率等静态工况。

##  快速开始 (Quick Start)

按照以下步骤，您可以在本地环境中快速部署并运行 TAPM-Net 进行电池数据处理与模型训练。

### 1. 克隆仓库与配置环境

首先克隆本项目到本地，并安装 `requirements.txt` 中指定的依赖包：

```bash
# 克隆仓库
git clone https://github.com/YourUsername/tapm-net.git
cd tapm-net

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

1. 在项目根目录下创建 `data/raw` 文件夹：
   ```bash
   mkdir -p data/raw
   ```
2. 将您的 `.csv` 原始电池老化数据文件（例如 `CY25-025_1-#1.csv` 等）拷贝到 `data/raw/` 路径下。由于文件大小限制，在此只上传了部分的数据，数据来自2022年的一篇论文Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation，完整数据可以在该论文处下载

### 3. 数据清洗与特征提取

运行预处理脚本。该脚本会自动读取 `data/raw` 中的原始文件，进行物理硬剪裁过滤、时序 Hampel 异常值平滑与线性插值，最终提取出标准退化时序特征并保存至 `data/processed/` 文件夹：

```bash
python scripts/run_preprocessing.py --raw_dir data/raw --out_path data/processed/TJU_NCA_cleaned_dataset.csv
```

### 4. 模型训练与独立测试集评估

执行训练脚本。程序将自动划分测试集（2块电池）与训练集（剩下的电池），你也可以自行修改训练集与测试集，训练三通道自适应物理调制网络（TAPM-Net），评估测试集电池的 MAE、RMSE、$R^2$ 等指标，并绘制全生命周期的 EOL 预测退化曲线：

```bash
python scripts/train.py --csv_path data/processed/TJU_NCA_cleaned_dataset.csv
```

训练完成后：
- 训练好的模型与数据缩放器（Scaler）将自动保存至根目录下的 `tapm_net_model.pth`。
- 测试电池的寿命预测趋势图将保存为项目根目录下的图片文件（例如 `CY35-05_1-#2_prediction.png`）。
```
