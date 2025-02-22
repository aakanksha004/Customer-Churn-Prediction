# Customer Churn Prediction

## Overview
Customer churn prediction is a machine learning project that aims to identify customers who are likely to leave a service. This project uses various customer-related features such as demographics, service usage, and contract details to predict churn using machine learning models.

## Dataset
The dataset contains customer information with the following features:
- **Gender**: Customer's gender (Male/Female)
- **SeniorCitizen**: Whether the customer is a senior citizen (0: No, 1: Yes)
- **Partner**: Whether the customer has a partner (Yes/No)
- **Dependents**: Whether the customer has dependents (Yes/No)
- **Tenure**: Number of months the customer has stayed with the company
- **PhoneService**: Whether the customer has phone service (Yes/No)
- **MultipleLines**: Whether the customer has multiple lines (Yes/No/No phone service)
- **InternetService**: Type of internet service (DSL/Fiber optic/No)
- **OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies**: Whether the customer has subscribed to these services (Yes/No)
- **Contract**: Type of contract (Month-to-month/One year/Two year)
- **PaperlessBilling**: Whether billing is paperless (Yes/No)
- **PaymentMethod**: Method of payment (Electronic check, Mailed check, etc.)
- **MonthlyCharges**: Monthly amount charged to the customer
- **TotalCharges**: Total amount charged
- **Churn**: Whether the customer has churned (Yes/No)

## Project Workflow
### 1. Importing the Dependencies
Necessary libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and XGBoost are imported.

### 2. Data Loading and Understanding
- The dataset is loaded and explored.
- **Class Imbalance**: The dataset is imbalanced with 5174 "No" and 1869 "Yes" in the Churn column.
- We apply **Upsampling or Downsampling** to handle the imbalance before training.

### 3. Data Preprocessing
- **CustomerID column removed** as it is not relevant for modeling.
- **No missing values** in the dataset, except for `TotalCharges`, which had empty strings replaced with "0.0" and converted to float.
- **Label Encoding**: Categorical variables are encoded.
- **Data Splitting**: The dataset is split into training and testing sets.
- **Handling Imbalance**: Synthetic Minority Oversampling Technique (SMOTE) is applied.

### 4. Exploratory Data Analysis (EDA)
- Histograms, boxplots, heatmaps, and count plots are used to visualize data distributions and relationships.

### 5. Model Training
#### Decision Tree
- **Cross-validation accuracy**: 0.78

#### Random Forest
- **Cross-validation accuracy**: 0.84

#### XGBoost
- **Cross-validation accuracy**: 0.83

#### Cross-validation Scores:
```
{'Decision Tree': array([0.68297101, 0.71601208, 0.81993958, 0.83564955, 0.83746224]),
 'Random Forest': array([0.72826087, 0.7734139 , 0.90332326, 0.89969789, 0.8978852 ]),
 'XGBoost': array([0.71135266, 0.74864048, 0.91178248, 0.88640483, 0.91117825])}
```
- **Random Forest achieved the highest accuracy among all models with default parameters.**

### 6. Model Evaluation
#### Accuracy Score:
- **0.777**

#### Confusion Matrix:
```
[[879 157]
 [157 216]]
```

#### Classification Report:
```
               precision    recall  f1-score   support

           0       0.85      0.85      0.85      1036
           1       0.58      0.58      0.58       373

    accuracy                           0.78      1409
   macro avg       0.71      0.71      0.71      1409
weighted avg       0.78      0.78      0.78      1409
```

### 7. Deployment & Prediction
- The trained model is saved and can be loaded for making predictions on new customer data.

## Future Improvements
- Improve feature engineering techniques.
- Tune hyperparameters for better accuracy.
- Test deep learning models.
- Deploy model with an API for real-time predictions.

## Requirements
Install necessary dependencies using:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## How to Run
1. Open the Jupyter Notebook:
   ```sh
   jupyter notebook CustomerChurnPrediction.ipynb
   ```
2. Run all cells to preprocess data, train models, and evaluate results.



