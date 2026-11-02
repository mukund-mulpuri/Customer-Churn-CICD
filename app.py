
import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="Customer Churn Prediction API")

# Load trained model and feature names
model = joblib.load("churn_model.pkl")
features = joblib.load("feature_columns.pkl")

@app.get("/health")
def health():
    return {"status": "Application is running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Feature engineering (same as training)
    df["Tenure_Category"] = pd.cut(
        df["Tenure_Months"],
        bins=[-1, 12, 36, 72],
        labels=["New", "Medium", "Long"]
    )

    df["Average_Monthly_Spending"] = (
        df["Total_Charges_INR"] /
        df["Tenure_Months"].replace(0, float("nan"))
    )

    df["Service_Usage_Indicator"] = (
        df["Tech_Support"].eq("Yes").astype(int) +
        df["Online_Security"].eq("Yes").astype(int)
    )

    df["Monthly_to_Total_Ratio"] = (
        df["Monthly_Charges_INR"] /
        df["Total_Charges_INR"].replace(0, float("nan"))
    )

    df = df[features]

    prediction = model.predict(df)[0]

    return {
        "Prediction": "Churn" if prediction == 1 else "Not Churn"
    }
