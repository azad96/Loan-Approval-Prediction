import requests

"""
Set the host to
    1. 'localhost' if you want to run the server locally.
    2. the URL of the deployed environment if you want to send a request to the cloud server.
"""
host = "localhost"
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