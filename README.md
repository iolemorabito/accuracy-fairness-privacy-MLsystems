# 👨‍🎓 Joint Assessment of Accuracy, Fairness, and Privacy in Machine Learning Systems

## 📖 Overview
This project evaluates **machine learning models** from three critical perspectives: **accuracy, fairness, and privacy**. It includes accuracy improvement, bias mitigation, and privacy risk mitigation techniques.

## 🛠️ Setup Instructions

### 1️⃣ Navigate to the Project Directory
```bash
cd path/to/project
```

### 2️⃣ Create and Activate a Python Virtual Environment
```bash
python -m venv venv
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1  # For Windows
source venv/bin/activate       # For macOS/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Main Script
```bash
python -W ignore main.py --dataset dataset_name --model model_name --technique technique_name
```

## 📂 Project Structure
```
├── data/
├── datasets/
├── metrics/
├── models/
├── techniques/
├── main.py
├── hyperparam_opt.py
├── crossvalid.py
├── pca.py
├── pca_plot.py
├── manova.py
├── manovaNoPlot.py
├── requirements.txt
├── Thesis.pdf
├── README.md
```

## 📌 Key Files & Functionalities
| File | Description |
|------------|-------------|
| `main.py` | Main script for running experiments |
| `hyperparam_opt.py` | Implements **hyperparameter optimization** |
| `crossvalid.py` | Handles **cross-validation** techniques |
| `pca.py` | Implements **Principal Component Analysis (PCA)** |
| `pca_plot.py` | Generates **PCA visualization plots** |
| `manova.py` | Implements **MANOVA test** with visualizations |
| `manovaNoPlot.py` | Implements **MANOVA test** without plots |
| `results.xlsx` | Stores analysis results |
| `requirements.txt` | Contains required Python libraries |


