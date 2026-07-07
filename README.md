# Sales-Forecasting
Machine learning-based sales prediction using Random Forest Regression and DataCo Supply Chain Dataset.

## 📌 Project Overview

This project is a Machine Learning-based Sales Prediction System that forecasts sales using historical supply chain data. It leverages the Random Forest Regression algorithm to analyze various business factors and predict future sales accurately.

The application performs data preprocessing, model training, performance evaluation, and interactive sales prediction based on user-provided inputs.

---

## 🚀 Features

- Data cleaning and preprocessing
- Sales prediction using Random Forest Regression
- Interactive user input for real-time prediction
- Product-specific prediction (when sufficient data is available)
- Model evaluation using multiple performance metrics
- Confusion Matrix visualization
- Exception handling and input validation

---

## 📂 Dataset

**Dataset Used:** DataCo Supply Chain Dataset

The dataset contains real-world supply chain and retail information, including:

- Days for Shipping (Real)
- Days for Shipment (Scheduled)
- Sales per Customer
- Late Delivery Risk
- Product Price
- Sales (Target Variable)

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🤖 Machine Learning Model

**Algorithm Used:**
- Random Forest Regressor

### Why Random Forest?

- Handles nonlinear relationships effectively
- Reduces overfitting through ensemble learning
- Performs well on structured tabular data
- Provides reliable prediction accuracy

---

## 📊 Model Evaluation Metrics

The model is evaluated using:

- Accuracy (Range-based)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Confusion Matrix Visualization

---

## 📌 Input Features

The model uses the following features:

- Days for Shipping (Real)
- Days for Shipment (Scheduled)
- Sales per Customer
- Late Delivery Risk
- Product Price

### Target Variable

- Sales

---

## 📷 Project Workflow

```
            DataCo Supply Chain Dataset
                        │
                        ▼
               Data Preprocessing
                        │
                        ▼
               Feature Selection
                        │
                        ▼
               Train-Test Split
                        │
                        ▼
      Random Forest Regression Model
                        │
                        ▼
             Model Evaluation
                        │
                        ▼
           Interactive User Prediction
                        │
                        ▼
              Predicted Sales Output
```

---

## ▶️ How to Run

### Clone the Repository

```bash
git clone https://github.com/your-username/Sales-Prediction-Using-RandomForest.git
```

### Navigate to the Project

```bash
cd Sales-Prediction-Using-RandomForest
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python predict_sales.py
```

---

## 📈 Output

The application provides:

- Predicted Sales
- Sales Range Category
- Confidence Interval
- Model Performance Metrics
- Confusion Matrix (Saved as `confusion_matrix.png`)

---

## 📁 Project Structure

```
Sales-Prediction-Using-RandomForest/
│
├── predict_sales.py
├── DataCoSupplyChainDataset.csv
├── confusion_matrix.png
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔮 Future Enhancements

- Hyperparameter tuning using GridSearchCV
- Comparison with XGBoost and Gradient Boosting
- Web application using Flask or Streamlit
- Real-time sales dashboard
- Sales trend visualization
- Deployment on cloud platforms

---

## 👨‍💻 Author

**Renu Thrisha**

