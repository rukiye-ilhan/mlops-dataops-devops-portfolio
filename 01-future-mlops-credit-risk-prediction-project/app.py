import  joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

#Load the train model
#we load the frozen model from the hard drive into RAM
model = joblib.load('models/credit_risk_model.joblib')

#Initialize the app
app = FastAPI(title="Credit Risk ML API", version="1.0")

#Define input data schema 
#Pydantic checks if the user sends the correct data types
class LoanApplication(BaseModel):
   # Veri setindeki sıraya göre tanımlamasak eşleşmeme hatası alırız
    person_age: int
    person_income: int
    person_emp_length: float
    loan_grade: int               
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: int
    cb_person_cred_hist_length: int
    # One-Hot Encoded columns
    person_home_ownership_OTHER: int
    person_home_ownership_OWN: int
    person_home_ownership_RENT: int
    
    loan_intent_EDUCATION: int
    loan_intent_HOMEIMPROVEMENT: int
    loan_intent_MEDICAL: int
    loan_intent_PERSONAL: int
    loan_intent_VENTURE: int

@app.post("/predict")
def predict_loan_status(data: LoanApplication):

    feature_order = [
        'person_age', 
        'person_income', 
        'person_emp_length', 
        'loan_grade',         
        'loan_amnt', 
        'loan_int_rate', 
        'loan_percent_income', 
        'cb_person_default_on_file', 
        'cb_person_cred_hist_length', 
        'person_home_ownership_OTHER', 
        'person_home_ownership_OWN', 
        'person_home_ownership_RENT', 
        'loan_intent_EDUCATION', 
        'loan_intent_HOMEIMPROVEMENT', 
        'loan_intent_MEDICAL', 
        'loan_intent_PERSONAL', 
        'loan_intent_VENTURE'
    ]
    #Convert to JSON input into a Pandas DataFrame
    df = pd.DataFrame([data.dict()])

    df = df[feature_order]

    #Make the prediction using the loaded pipeline
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]

    #Return the result as JSON
    result = "DEFAULT" if prediction[0] == 1 else "PAID"

    return {
        "prediction": int(prediction[0]),
        "probability_of_default": float(probability[0]),
        "result_text": result
    }

#Root endpoint just to check if API is alive
@app.get("/")
def home():
    return {"message": "MLOps API is running! Go to /docs to test the model."}
