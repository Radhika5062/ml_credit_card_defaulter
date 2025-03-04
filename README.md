# Credit Card Fraud Detection

This project aims to predict and detect fraudulent credit card transactions using machine learning algorithms. It leverages various machine learning models and techniques to analyze transaction data and classify them as either legitimate or fraudulent.
It utilises **FastAPI** for serving the model, **Docker** for containerisation and is deployed on **AWS EC2** using **AWS ECR**.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Installation](#installation)
- [Tools and Technologies Used](#tools-and-technologies-used)
- [Approach](#approach)


## Project Overview

Credit card fraud is a major issue in the financial industry, costing businesses and individuals billions every year. The goal of this project is to build a robust machine learning model that can automatically detect fraudulent transactions based on historical transaction data. This project includes data preprocessing, feature engineering, model building, model evaluation alogn with deployment over AWS to create an accurate fraud detection system.

## Data

This project uses the publicly available **Credit Card Fraud Detection** dataset. The dataset consists of credit card transactions, including information like:

 - LIMIT_BAL: Credit Limit of the person.
 - SEX: Gender of the person
 - EDUCATION: Highest education level attained by the person
 - MARRIAGE: Marital Status of the person
 - AGE: Age of the person
 - PAY_0 to PAY_6: History of past payment. These are the past monthly payment records (from April to September, 2005). If it is 0 then it means the person paid in time. If it negative the it means there was a delay and the amount of delay is represented by the numnber. Positive means the payment was made in time and the number next to it represents the number of times this happened.
 - BILL_AMT1 to BILL_AMT6: Amount of bill statements.
 - PAY_AMT1 to PAY_AMT6: Amount of previous payments. 

Dataset : [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).


## Installation

To run this project locally, you will need to have Python installed. Additionally, the following Python packages are required:

- `pandas`
- `xlrd`
- `pandas`
- `numpy`
- `matplotlib`
- `plotly`
- `seaborn`
- `statsmodels`
- `scikit-learn`
- `imblearn`
- `xgboost`
- `python-dotenv`
- `pymongo[srv]`
- `certifi`
- `dill`
- `pyYAML`
- `scipy`
- `kneed`
- `fastapi`
- `uvicorn==0.18.3`
- `gunicorn`


You can install these dependencies by running:

```bash
pip install -r requirements.txt

## Tools and Technologies Used
- Python: The main programming language used.
- pandas: For data manipulation and preprocessing.
- scikit-learn: For machine learning model building and evaluation.
- imbalanced-learn: To handle imbalanced classes 
- matplotlib & seaborn: For data visualization.
- Docker: For containerization
- AWS EC2: For deployment
- AWS ECR: For container registry
- FastAPI: For serving the model as API's which can be utilized internally or by 3rd party vendors. 
- CI/CD pipelines
- Github Actions



## Approach
The workflow of this project includes the following key steps:

1. Data Ingestion: 
Data is fetched from MongoDB Server and is stores locally in a csv file. 

2. Exploratory Data Analysis (EDA)
Perform exploratory data analysis (EDA) to understand the data distribution and feature relationships.

3. Data Validation
- Performing checks on data to understand if all the columns are present and also the type of column is coming as expected.
- Data drift is also detected at this stage.

4. Data Transformation
- Handle missing values, if any.
- Applying data transformation steps like converting imbalanced data into balanced data using SmoteTomek. 

5. Model Training
- Apply various classification algorithms to understand which ones are performing the best. Out of all the different classification methods tried during Exploratory Data Analysis, Random Forest and XGBoost performed the best.
- Use techniques like cross-validation and hyperparameter tuning to improve model performance.

6. Model Evaluation:
- Evaluate the models using metrics like:
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC Curve
    - Confusion Matrix
- New model is also being tested against the old saved model to test if the model trained on latest data performs better than the saved model. 

7. Model Pusher:
- If the latest model performs better than the saved model then it is pushed production. 

8. Containerization is done using Docker and AWS ECR is used for container registry to store the container privately.

9. Creating CI/CD pipeline using Github Actions to deploy model in AWS EC2. 

