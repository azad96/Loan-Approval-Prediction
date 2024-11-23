import requests

host = "loan-approval-env.eba-mzcu93pg.eu-west-1.elasticbeanstalk.com"
url = f"http://{host}/predict"
client = {
    "person_age": 28,
    "person_gender": "male",
    "person_education": "master",
    "person_income": 60000.0,
    "person_home_ownership": "rent",
    "loan_amnt": 49500.0,
    "loan_intent": "homeimprovement",
    "loan_int_rate": 16.5,
    "loan_percent_income": 0.30,
    "credit_score": 600,
    "previous_loan_defaults_on_file": "yes"
}
result = requests.post(url, json=client).json()
print(result)