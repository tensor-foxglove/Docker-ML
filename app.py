from fastapi import FastAPI, Request
import pickle
import numpy as np

app = FastAPI()
model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def root():
    return {"message": "ML model is ready!"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    input_array = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
