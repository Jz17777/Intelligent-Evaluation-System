# 🧠 Intelligent Evaluation System (智能评价系统)


> A Chinese text sentiment analysis system based on **PyTorch + GRU**.  
> 一个基于 **PyTorch + GRU** 的中文文本情感分析系统，用于自动识别文本的情感倾向。

---

## 📚 Overview | 项目简介

**Intelligent Evaluation System** aims to build a lightweight yet powerful **Chinese text sentiment classification** model using GRU (Gated Recurrent Unit).  
The system can automatically evaluate and classify text polarity (positive/negative) and is suitable for applications in:
- 🛒 E-commerce review analysis  
- 📰 Public opinion monitoring  
- 💬 Intelligent customer service  
- 🧑‍🏫 Education and psychological evaluation

---

## 🧩 Project Structure | 项目结构
```text
Intelligent-Evaluation-System/
├─ .gitignore
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ processed/
│  └─ raw/
├─ logs/
├─ models/
└─ src/
   ├─ config.py
   ├─ dataset.py
   ├─ evaluate.py
   ├─ model.py
   ├─ predict.py
   ├─ process.py
   ├─ train.py
   └─ utils.py
```
---

## ⚙️ Environment Setup | 环境配置

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Jz17777/Intelligent-Evaluation-System.git
cd Intelligent-Evaluation-System
```
### 2️⃣ Create virtual environment and install dependencies
```bash
conda create -n evalsys python=3.10
conda activate evalsys
pip install -r requirements.txt
```

---

## 🔄 Model Training | 模型训练

You can switch between GRU / LSTM / RNN using the --arch argument:

```bash
# Train GRU model (default)
python src/train.py --arch GRU

# Train LSTM model
python src/train.py --arch LSTM

# Train RNN model
python src/train.py --arch RNN
```

Training process will automatically:

Display training & validation loss / accuracy

Save the best model under models/best_GRU_model.pt

Log training metrics to logs/ for visualization

---

## Model Evaluation | 模型评估
After training, you can evaluate your model performance on the validation or test dataset using:
```bash
# Evaluate GRU model (default)
python src/evaluate.py --arch GRU

# Evaluate LSTM model
python src/evaluate.py --arch LSTM

# Evaluate RNN model
python src/evaluate.py --arch RNN
```
This script will:

Load the best trained model (models/best_GRU_model.pt)

Evaluate performance metrics such as:

✅ Accuracy

🎯 Precision

🔁 Recall

🧮 F1-score

Optionally generate a confusion matrix for analysis

## 📊 Visualization with TensorBoard | 可视化

You can monitor model performance using TensorBoard:

```bash
tensorboard --logdir=logs
```

Then open in your browser: 👉 http://localhost:6006

---
## 🧠 Prediction | 模型预测
To perform batch predictions using the trained model:
```bash
# Predict GRU model
python src/predict.py --arch GRU

# Predict LSTM model
python src/predict.py --arch LSTM

# Predict RNN model
python src/predict.py --arch RNN
```

---
## 📁 Directory Summary | 文件目录说明

| Folder  | Description |
|----------------|------|
| `data/` | Stores raw and processed datasets (excluded from GitHub) |
| `models/` | Model saving directory (e.g., best_GRU_model.pt) |
| `logs/` | TensorBoard logs for visualization |
| `src/` | Core source code directory |

---
## 💡 Configuration | 参数配置（config.py）

You can modify training hyperparameters in src/config.py:
```bash
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
BIDIRECTIONAL = True
```
---
## 🧑‍💻 Author Information | 作者信息

Author: Christopher Zhou(Jz17777)  
Email: christopherzjz@outlook.com  
University: University of Glasgow, James Watt School of Engineering  
Language: Python 3.10, PyTorch 2.x  
