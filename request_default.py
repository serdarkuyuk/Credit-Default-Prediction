import requests

# url = "http://127.0.0.1:5000/"
url = "https://creditdefaultsesame.herokuapp.com/"
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
files = []
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)