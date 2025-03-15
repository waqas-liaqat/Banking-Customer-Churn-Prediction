# Banking Customer Churn Prediction

## 📌 Project Overview
Customer churn is a critical problem in the banking industry, as retaining customers is often more cost-effective than acquiring new ones. This project aims to **predict customer churn** using **Machine Learning (ML) and Deep Learning (ANN)** approaches and deploy the solution via **Streamlit**.

## 📂 Dataset Information
- **Dataset Name:** Banking Customer Churn Prediction
- **Source:** [Kaggle](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset)
- **Rows:** 10,000
- **Columns:** 14
- **Target Variable:** `Exited` (1 = Churned, 0 = Not Churned)

### 📊 Features:
| Feature           | Description |
|------------------|-------------|
| CreditScore      | Customer's credit score |
| Geography        | Customer's country |
| Gender          | Male/Female |
| Age             | Customer's age |
| Tenure          | Number of years with the bank |
| Balance         | Account balance |
| NumOfProducts   | Number of bank products used |
| HasCrCard       | If the customer has a credit card (1/0) |
| IsActiveMember  | If the customer is active (1/0) |
| EstimatedSalary | Estimated annual salary |

---
## 🔍 Exploratory Data Analysis (EDA)

### ✅ Key Findings:
- **Churn Rate:** 20.37% of customers churned.
- **Geography Influence:** Customers from **Germany** had the highest churn rate (~50%).
- **Gender Impact:** More **female customers churned** compared to males.
- **Age Factor:** Customers aged **46-60** had the highest churn rate.
- **Account Balance:** Higher balance customers churned more.
- **Product Usage:** Customers with **3 or more products** had a high churn rate.

### 🛠 Data Preprocessing:
- Removed irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
- Encoded categorical variables (`OneHotEncoder`)
- Scaled numerical variables (`MinMaxScaler`)
- Checked and handled class imbalance using **Random Under Sampling**

---
## 🏗️ Model Building
### **Traditional Machine Learning Models**
| Model                | Recall Score |
|---------------------|-------------|
| Logistic Regression | 0.66 |
| Decision Tree (Best) | **0.72** |
| Random Forest       | 0.68 |
| AdaBoost            | 0.68 |
| Gradient Boosting   | 0.71 |

- The **Decision Tree (Depth = 3, Splitter = Best)** was selected as the best model based on **recall score**.
- The model was **not highly accurate** in predicting churn, leading to testing **Deep Learning (ANN).**

---
## 🤖 Deep Learning: Artificial Neural Network (ANN)

### 🔹 **Model Architecture:**
- **Input Layer:** 12 features
- **Hidden Layers:** 3 layers with **ReLU activation**
- **Dropout Layer:** 20% dropout for regularization
- **Output Layer:** 1 neuron with **Sigmoid activation**
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy

### 🏆 **Performance:**
| Metric      | Score  |
|------------|--------|
| Accuracy   | **82.4%** |
| Precision  | 0.85 (Not Churned) / 0.60 (Churned) |
| Recall     | 0.95 (Not Churned) / **0.29 (Churned)** |

🔹 **Best Hyperparameters (from GridSearchCV):**
- Optimizer: **Adam**
- Learning Rate: **0.001**
- Neurons: **32**
- Dropout: **0.2**
- Batch Size: **32**
- Epochs: **50**

---
## 🚀 Deployment (Streamlit)
The model was deployed as a **Streamlit web application** where users can input customer details and predict churn probability.

### 🔧 **How to Run Locally**
1️⃣ **Clone this repository:**
```bash
git clone https://github.com/waqas-liaqat/Banking-Customer-Churn-Prediction.git

cd customer-churn-prediction
```
2️⃣ **Install dependencies:**
```bash
pip install -r requirements.txt
```
3️⃣ **Run the Streamlit app:**
```bash
streamlit run app.py
```
4️⃣ **Open in browser:** `http://localhost:8501/`

---
## 📢 Future Improvements
- **Better Imbalance Handling**: Use **SMOTE** or weighted loss functions to improve recall.
- **Threshold Tuning**: Adjust decision thresholds for better recall.
- **Explainability**: Implement **SHAP or LIME** for better feature importance visualization.
- **Cloud Deployment**: Deploy on **Streamlit Cloud / Hugging Face Spaces / Render**.

---
## 🏆 Key Takeaways
✅ **ML models struggled**, but **ANN improved performance**.

✅ **Decision Tree performed best among ML models.**

✅ **Streamlit deployment made model accessible.**

✅ **More improvements needed in recall for churners.**

---
## 📬 Contact Information
💡 **Author:** Muhammad Waqas  
📧 **Email:** waqasliaqat630@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/muhammad-waqas-liaqat)  
🔗 [GitHub](https://github.com/waqas-liaqat)  
🔗 [Kaggle](https://www.kaggle.com/muhammadwaqas630)  

🚀 **Star this repo if you liked this project!** ⭐

