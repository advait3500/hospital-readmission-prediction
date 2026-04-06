# Predicting 30-Day Hospital Readmissions with AutoML (PyCaret)

## 📌 Project Overview
This project applies **PyCaret**, an AutoML library in Python, to predict whether a patient will be readmitted within 30 days of discharge.  
Hospital readmissions are costly for healthcare systems, and early prediction can help providers improve care and allocate resources more effectively.

The project demonstrates **end-to-end machine learning**, from data cleaning and feature selection to automated modeling, evaluation, and insights.

## 🛠 Tools & Libraries
- **Python** (Pandas, NumPy)
- **PyCaret** (automated machine learning for classification)
- **Missingno** (data quality visualization)
- **Matplotlib / Seaborn** (data visualization)

## 📊 Workflow
1. **Data Exploration** – Loaded the dataset, checked dimensions, inspected column types, and visualized missing values with `missingno` to identify data quality issues.  
2. **Feature Engineering** – Defined the target variable (`Readmit30`). Dropped low-quality features (`Weight`, `Height`, `Temperature`) due to large amounts of missing data and potential bias. Encoded categorical variables and normalized numeric features.  
3. **Modeling** – Used PyCaret’s `setup()` and `compare_models()` functions to preprocess the dataset and benchmark multiple machine learning algorithms automatically.  
4. **Results** – Evaluated models on metrics such as **Accuracy**, **Precision**, **Recall**, and **ROC-AUC**.  
5. **Conclusion** – Selected the best-performing model and discussed trade-offs (e.g., prioritizing recall to catch more at-risk patients, even at the cost of false positives).  

## 🚀 Results
- **Best Model:** Random Forest Classifier (via PyCaret AutoML)  
- **Accuracy:** ~83%  
- **ROC-AUC:** ~0.87  
- **Recall (Sensitivity):** ~79%  



## 🔮 Next Steps
- Incorporate additional clinical features (e.g., comorbidities, lab test results) to improve prediction accuracy  
- Apply **SHAP values** for explainability, making predictions more interpretable for healthcare professionals  
- Deploy the model as a **Streamlit web app** or **Flask API** for real-world use in hospital decision support  

## 📂 Repository Structure
- `Polished_Project_Portfolio.ipynb` – Final polished notebook with explanations and results  
- `requirements_core.txt` – Minimal dependencies needed to run this project  
- `requirements.txt` – Full Colab environment export (includes many extra packages not directly used)  

## 📦 Dependencies
- If you just want to run the project normally: install from `requirements_core.txt`  
- If you want to replicate the **exact Colab environment** used: install from `requirements.txt` (note: this file is much larger)  

Install with pip:
```bash
pip install -r requirements_core.txt
