# TAPM-Net-Tri-Channel-Adaptive-Physical-Modulation-Network-for-Battery-SOH-and-EOL-Prediction
- 这是一个基于物理调制与多通道特征融合的锂离子电池健康状态（SOH）预测与寿命终止点（EOL）外推推演项目。模型融合了时序统计与电池退化物理特征，并结合 FiLM (Feature Modulation) 机制整合了环境温度和充放电倍率等静态工况。模型大体结构如下图所示
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/fd7e79e3-e62f-4095-90f0-2e3efb238ead" />

- 本模型采用了创新的自适应三通道物理调制网络（TAPM-Net）架构，旨在实现对动力电池全生命周期退化轨迹的精准前向推演。
- 在数据输入端，模型摒弃了冗余的原始时序电信号，清洗出反映电池健康状态的8维次生特征指标；电池健康状态SOH、库仑效率、放电中段电压、恒流充电时长、欧姆内阻、恒压容量占比、电压弛豫跌落、恒压充电时长。
- 在核心编码层，模型构建了三通道解耦架构，将特征严格划分为仅处理历史SOH时序的内生自回归通道、处理随老化单调递减的外生正相关诊断通道（库仑效率、放电中段电压和恒流充电时长）、以及处理随老化单调递增的外生负相关诊断通道（欧姆内阻、恒压容量占比、电压弛豫跌落和恒压充电时长），并利用三个独立的门控循环单元（GRU）将其压缩为表征电池能量存量与功率衰退的多维健康隐向量，有效防范了单一指标失效带来的预测波动；
- 为了避免网络产生对单一特征的过度依赖，在训练过程中引入了随机掩盖策略，每轮训练以一定概率随机丢弃8个特征中的某些特征；
- 为消除环境变量的干扰，模型引入基于多层感知机的特征线性调制机制（FiLM），接收标准化的温度和充放电倍率，动态生成仿射变换参数对三大通道的隐向量进行逐元素修正，从而将不同工况下的老化特征对齐到统一参考系下；
- 最后，轨迹重建解码器将这三个调制后的隐向量与目标未来循环序号进行跨维度非线性融合，通过多层感知机直接预测对应时刻的SOH，解决了在未来次生特征未知的盲测场景下对电池长期寿命曲线进行独立预测的物理难题。
- 这是所使用的电池的原始数据的绘图概览
<img width="1428" height="707" alt="image" src="https://github.com/user-attachments/assets/1cd7ed3a-299a-4d12-8c94-f5ef024fe2d3" />
<img width="1428" height="707" alt="image" src="https://github.com/user-attachments/assets/ebb8ba93-b6a9-43a2-b7c5-0d7267e43260" />
<img width="1428" height="707" alt="image" src="https://github.com/user-attachments/assets/098bd0ca-7684-45e0-9543-ccd43ad93a5e" />
- 如下是程序从原始充放电数据中所提取的与电池寿命退化有关的特征
<img width="1589" height="1769" alt="image" src="https://github.com/user-attachments/assets/06310085-9815-46c0-afd7-9df8ba35b432" />

- 如下是选取一个未知的电池，去预测其寿命终止（以最大放电低于初始的80%为准）的充放电圈数，其中 $R^2$ 指标为97.91%，当然，在未知的电池上预测或有误差，这就是我们的迁移学习后续要解决的问题。

<img width="3000" height="1500" alt="CY25-025_1-#2_prediction" src="https://github.com/user-attachments/assets/2397c71d-38cf-40e2-b5f5-d79772da8137" />


##  快速开始 (Quick Start)

按照以下步骤，快速部署并运行 TAPM-Net 进行电池数据处理与模型训练。

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
- 将 `.csv` 原始电池老化数据文件（例如 `CY25-025_1-#1.csv` 等）拷贝到 `data/raw/` 路径下。由于文件大小限制，在此只上传了部分的数据，数据来自2022年的一篇论文Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation，完整数据可以在该论文处下载

### 3. 数据清洗与特征提取

- 运行预处理脚本。该脚本会自动读取 `data/raw` 中的原始文件，进行物理硬剪裁过滤、时序 Hampel 异常值平滑与线性插值，最终提取出标准退化时序特征并保存至 `data/processed/` 文件夹：

```bash
python scripts/run_preprocessing.py --raw_dir data/raw --out_path data/processed/TJU_NCA_cleaned_dataset.csv
```

### 4. 模型训练与独立测试集评估

- 执行训练脚本。程序将自动划分测试集（2块电池）与训练集（剩下的电池），你也可以自行修改训练集与测试集，训练三通道自适应物理调制网络（TAPM-Net），评估测试集电池的 MAE、RMSE、 $R^2$ 等指标，并绘制全生命周期的 EOL 预测退化曲线：

```bash
python scripts/train.py --csv_path data/processed/TJU_NCA_cleaned_dataset.csv
```

训练完成后：
- 训练好的模型与数据缩放参数将自动保存至根目录下的 `tapm_net_model.pth`。
- 测试电池的寿命预测趋势图将保存为项目根目录下的图片文件（例如 `CY35-05_1-#2_prediction.png`）。
- 模型还应该能够迁移到其它特征的电池数据，具体可详见本栏目后续项目。
