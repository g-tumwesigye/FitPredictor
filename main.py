from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import uvicorn
import os
import traceback

# Initialize FastAPI
app = FastAPI(title="FitPredictor API", description="Predict BMI Case based on user inputs", version="1.0")

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Model, Scaler, and Label Encoders
try:
    MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "sgd_momentum_model.h5")
    SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler.pkl")
    ENCODER_PATH = os.path.join(BASE_DIR, "saved_models", "label_encoders.pkl")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODER_PATH)

    print("Model, Scaler, and Label Encoders Loaded Successfully")

except Exception as e:
    print(f"Error loading model or preprocessing files: {e}")

# Input Data Schema
class InputData(BaseModel):
    Weight: float
    Height: float
    BMI: float
    Age: int
    Gender: str  # "Male" or "Female"

@app.post("/predict/")
def predict(data: InputData):
    try:
        # Convert Gender to numerical
        gender_encoded = label_encoders["Gender"].transform([data.Gender])[0]

        # Compute BMI_to_Weight
        bmi_to_weight = data.BMI / data.Weight

        # Prepare input array
        input_array = np.array([[data.Weight, data.Height, data.BMI, data.Age, bmi_to_weight, gender_encoded]])

        # Debugging: Print input shape before scaling
        print(f"\nInput array shape before scaling: {input_array.shape}")
        print(f"Expected features: Weight, Height, BMI, Age, BMI_to_Weight, Gender (Total = {input_array.shape[1]})")

        # Scale the input
        input_scaled = scaler.transform(input_array)

        # Debugging: Print shape after scaling
        print(f"Scaled input shape: {input_scaled.shape}")

        # Make prediction
        predictions = model.predict(input_scaled)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Convert class index to label
        predicted_label = label_encoders["BMIcase"].inverse_transform([predicted_class])[0]

        return {
            "BMIcase": predicted_label,
            "Confidence": f"{np.max(predictions) * 100:.2f}%"  # Highest confidence score
        }

    except Exception as e:
        traceback.print_exc()  # Print error in terminal
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# Run API
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)

