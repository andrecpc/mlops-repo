import pandas as pd
import mlflow
import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

app = FastAPI()

os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

class Model():
    def __init__(self, model_name, model_stage):
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions

model = Model("keras_main", "Staging")

@app.post("/invocations")
async def create_upload_file(file: UploadFile = File(...)):

    if file.filename.endswith(".jpg"):


        class_map = {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli',
                    5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya',
                    11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}
        
        # 1. Load and preprocess the image
        test_img = image.load_img(file.filename, target_size=(150, 150))
        test_img_arr = image.img_to_array(test_img)/255.0
        test_img_input = test_img_arr.reshape((1, test_img_arr.shape[0], test_img_arr.shape[1], test_img_arr.shape[2]))

        os.remove(file.filename)

        # 2. Make Predictions
        predicted_label = np.argmax(model.predict(test_img_input))
        predicted_vegetable = class_map[predicted_label]

        
        # Return a JSON object containing the model predictions
        return predicted_vegetable
 
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request 
        # (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only jpg Files accepted.")
# Check if the environment variables for AWS access are available. 
# If not, exit the program
if os.getenv("AWS_ACCESS_KEY_ID") is None or os.getenv("AWS_SECRET_ACCESS_KEY") is None:
    exit(1)