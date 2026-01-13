# IEEE-CIS-Fraud-Detection
# IEEE-CIS Fraud Detection & Real-Time Alert System 🛡️

This project implements an end-to-end machine learning solution to identify fraudulent credit card transactions using the IEEE-CIS dataset. It features a high-performance predictive model and a live Streamlit dashboard for real-time risk assessment.

## 📊 Project Highlights
- **Model Performance**: Utilized a Random Forest Classifier to achieve a **ROC-AUC score of 0.8516**.
- **Data Handling**: Addressed extreme class imbalance (only ~2.5% fraud) using stratified sampling and specialized evaluation metrics.
- **Deployment**: Developed an interactive dashboard for financial investigators to monitor transactions and adjust risk sensitivity thresholds.

## 📁 Repository Structure
- **`Fraud_Detection_Project.ipynb`**: The complete data science lifecycle, including exploratory data analysis (EDA), data imputation, feature encoding, and model validation.
- **`app.py`**: Python script for the Streamlit web application that serves as the user interface for the detection system.
- **`fraud_model_v1.pkl`**: The serialized Random Forest model used for making real-time predictions.
- **`model_features.pkl`**: Metadata containing the exact feature order required for model inference.
- **`.gitignore`**: Configuration to prevent the upload of large raw datasets to GitHub.

## 🛠️ Technology Stack
- **Python**: Primary programming language.
- **Scikit-Learn**: For building and tuning the Random Forest model.
- **Pandas & NumPy**: For advanced data manipulation and cleaning.
- **Streamlit**: For deploying the interactive fraud monitoring dashboard.
- **Matplotlib & Seaborn**: For statistical data visualization.

## 🚀 How to Run the App
To run the live dashboard on your local machine:
1. Clone this repository: `git clone https://github.com/YOUR_USERNAME/REPO_NAME.git`
2. Install dependencies: `pip install streamlit pandas joblib scikit-learn`
3. Launch the app: `streamlit run app.py`

## ✅ Key Metrics
- **ROC-AUC**: 0.8516
- **PR-AUC**: 0.4749 (High precision for imbalanced data)
