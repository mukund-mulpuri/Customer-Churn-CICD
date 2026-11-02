from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_prediction():
    response = client.post("/predict", json={
        "Gender": "Male",
        "Senior_Citizen": "No",
        "Tenure_Months": 12,
        "Contract_Type": "Month-to-month",
        "Internet_Service": "Fiber",
        "Payment_Method": "Cash",
        "Monthly_Charges_INR": 974.56,
        "Total_Charges_INR": 12432.49,
        "Tech_Support": "No",
        "Online_Security": "Yes",
        "Support_Tickets_6M": 1,
        "Avg_Data_Usage_GB": 111.82
    })

    assert response.status_code == 200
    assert response.json()["Prediction"] in ["Churn", "Not Churn"]