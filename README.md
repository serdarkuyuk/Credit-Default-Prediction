# Default Credit Prediction

## Credit Sesame Data Challenge

H. Serdar Kuyuk - March 5, 2021

# Summary

The goal of this project is to predict if a client will default next month or not. Machine learning algorithms have been used in many challenging prediction tasks. Although there is no magic in using advanced algorithms, there are always trade-offs on predictions. This notebook is prepared to describe techniques of data manipulation and usage of machine learning.

This data set has been sourced from the Machine Learning Repository of the University of California, Irvine Default of Credit Card Clients Data Set (UC Irvine). This dataset (30k client records) contains information on default payments (about 22% of data), demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

Exploratory data analysis, data visualization, and classification modeling techniques are indispensable in creating a machine learning model. There are a few steps executed in the project, i) exploration of the dataset, ii) Cleaning to make choices about undocumented labels iii) Feature engineering, iv) final result, and lessons learned.

**Can we predict the default with a month of advance?**

Yes. A test has been done if the proposed machine learning model performance is as expected as a trained model. The model can predict 70% percent (recall) of defaults for the next month.

The selected model Light Gradient Boosting Machine is the best model in terms of speed performance and accuracy compared to 15 other models.

In the final test set (1501 accounts), 231 defaults, and 796 non-defaults are correctly identified. There are 101 Type II errors and 372 Type I errors. That means 102 accounts out of 1501 accounts are considered non-default by the model where these accounts were default.

**Can we do better?**

Yes. The model can be improved in two ways. First, more features can be generated. New features might create better. Secondly, the model can be improved. However, to improve the model, we need to talk with the stakeholders and try to understand the data better.

## Objectives:

Identify the key drivers that determine the likelihood of credit card default.
Predict the likelihood of credit card default for customers of the Bank.

## Background

A dataset from UCI called Default of Credit Card Clients Dataset is used in this project. This dataset contains information on default payments, demographic factors, credit data, payment history, and billing statements of credit card clients in Taiwan from April 2005 to September 2005. There are 24,000 samples and 25 features. Short descriptions of each column are as follows:

- **ID:** ID of each client
- **LIMIT_BAL:** Amount of given credit in NT dollars (includes individual and family/supplementary credit)
- **SEX:** Gender (1=male, 2=female)
- **EDUCATION:** (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
  MARRIAGE: Marital status (1=married, 2=single, 3=others)
- **AGE:** Age in years
- **PAY_0 to PAY_6:** Repayment status by n months ago (PAY_0 = last month ... PAY_6 = 6 months ago) (Labels: -1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
  Variables are -2: No consumption; -1: Paid in full; 0: The use of revolving credit; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
- **BILL_AMT1 to BILL_AMT6:** Amount of bill statement by n months ago ( BILL_AMT1 = last_month .. BILL_AMT6 = 6 months ago)
- **PAY_AMT1 to PAY_AMT6:** Amount of payment by n months ago ( BILL_AMT1 = last_month .. BILL_AMT6 = 6 months ago)
- **default:** Default payment (1=yes, 0=no) Target Column

# Request

- For this project the developed model is deployed to this server:
  https://creditdefaultsesame.herokuapp.com/

- The server works as API. Please use below code or "request_default.py" to make a request. You can change the input by changing payload in the file.

```python
import requests

# server host the machine learning model
url = "https://creditdefaultsesame.herokuapp.com/"

# requested data input as dictionary
payload = {
"ID": "1",
"LIMIT_BAL": "20000",
"SEX": "2",
"EDUCATION": "2",
"MARRIAGE": "1",
"AGE": "24",
"PAY_1": "2",
"PAY_2": "2",
"PAY_3": "-1",
"PAY_4": "-1",
"PAY_5": "-2",
"PAY_6": "-2",
"BILL_AMT1": "3913",
"BILL_AMT2": "3102",
"BILL_AMT3": "689",
"BILL_AMT4": "0",
"BILL_AMT5": "0",
"BILL_AMT6": "0",
"PAY_AMT1": "0",
"PAY_AMT2": "689",
"PAY_AMT3": "0",
"PAY_AMT4": "0",
"PAY_AMT5": "0",
"PAY_AMT6": "0",
"default payment next month": "1",
}

# holds file and headers. These are empty now.
files = []
headers = {}

# request a post
response = requests.request("POST", url, headers=headers, data=payload, files=files)

# response is written in the screen.
print(response.text)
```
