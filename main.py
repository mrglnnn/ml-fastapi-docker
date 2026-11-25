from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title='Iris Classification API')
model = joblib.load(Path(__file__).with_name('model.joblib'))
classes = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']

class Features(BaseModel):
    sepal_length: float = Field(gt=0, examples=[5.1])
    sepal_width: float = Field(gt=0, examples=[3.5])
    petal_length: float = Field(gt=0, examples=[1.4])
    petal_width: float = Field(gt=0, examples=[0.2])

@app.get('/')
def root():
    return {'message': 'ML API is running'}

@app.post('/predict')
def predict(data: Features):
    values = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = int(model.predict(values)[0])
    return {'prediction': classes[prediction], 'class_id': prediction}
