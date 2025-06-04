from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load your model
model = pickle.load(open("model.pkl", "rb"))

# Define input schema
class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}
