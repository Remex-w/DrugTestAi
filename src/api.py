from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize FastAPI app
app = FastAPI()

# Load model and encoders
model = load_model("./model/DrugTestMultiOutputModelNew1.keras")

with open('./model/encoders/category_encoder.json', 'r') as f:
    category_data = json.load(f)

with open('./model/encoders/result_type_encoder.json', 'r') as f:
    result_type_data = json.load(f)

# Create LabelEncoder objects
category_classes = np.array(list(category_data.keys()))
result_classes = np.array(list(result_type_data.keys()))

# Image preprocessing function
def preprocess_image(file_path):
    try:
        # Load image using load_img from keras
        image = load_img(file_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")


# Input and Output Models
class PredictionResponse(BaseModel):
    Drug_Type: str
    Test_Result: str

# Route to upload and classify image
@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_location = f"./{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Preprocess and predict
        processed_image = preprocess_image(file_location)
        predictions = model.predict(processed_image)

        # Decode predictions
        drug_pred = category_classes[np.argmax(predictions[0])]
        result_pred = result_classes[np.argmax(predictions[1])]

        return PredictionResponse(Drug_Type=drug_pred, Test_Result=result_pred)
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
if __name__ == "__main__":
    port = int(os.environ.get("PORT"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)