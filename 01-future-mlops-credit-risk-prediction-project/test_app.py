from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk API is Live! Go to /docs for testing."}

def test_predict():
    # send fake data for testing
    payload = {
        "person_age": 25,
        "person_income": 50000,
        "person_emp_length": 2,
        "loan_grade": 1,
        "loan_amnt": 10000,
        "loan_int_rate": 10.0,
        "loan_percent_income": 0.2,
        "cb_person_default_on_file": 0,
        "cb_person_cred_hist_length": 3,
        "person_home_ownership_OTHER": 0,
        "person_home_ownership_OWN": 0,
        "person_home_ownership_RENT": 1,
        "loan_intent_EDUCATION": 0,
        "loan_intent_HOMEIMPROVEMENT": 0,
        "loan_intent_MEDICAL": 1,
        "loan_intent_PERSONAL": 0,
        "loan_intent_VENTURE": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "result_text" in response.json()