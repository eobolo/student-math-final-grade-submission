import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import joblib

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the meta data path
metadata_file = os.path.join(os.path.dirname(__file__), '../METADATA/metadata.json')

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), "../MODELS/best_model.pkl")
best_model = joblib.load(model_path)

# Load the training data columns for alignment during one-hot encoding
X_path = os.path.join(os.path.dirname(__file__), "../MODELS/X_train_columns.pkl")
X_columns = joblib.load(X_path)

# Categorical columns for one-hot encoding
categorical_cols = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

# Constraints for validation
constraints = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "age": (14, 22),
    "address": ["U", "R"],
    "famsize": ["GT3", "LE3"],
    "Pstatus": ["A", "T"],
    "Medu": [0, 1, 2, 3, 4],
    "Fedu": [0, 1, 2, 3, 4],
    "Mjob": ["at_home", "health", "other", "services", "teacher"],
    "Fjob": ["at_home", "health", "other", "services", "teacher"],
    "reason": ["course", "home", "other", "reputation"],
    "guardian": ["mother", "father", "other"],
    "traveltime": (1, 5),
    "studytime": (1, 5),
    "failures": [0, 1, 2, 3],
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"],
    "famrel": [1, 2, 3, 4, 5],
    "freetime": [1, 2, 3, 4, 5],
    "goout": [1, 2, 3, 4, 5],
    "Dalc": [1, 2, 3, 4, 5],
    "Walc": [1, 2, 3, 4, 5],
    "health": [1, 2, 3, 4, 5],
    "absences": (0, 100)
}

# Pydantic model for input data with validation
class StudentInput(BaseModel):
    school: str
    sex: str
    address: str
    famsize: str
    Pstatus: str
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    age: int = Field(..., ge=14, le=22)
    Medu: int = Field(..., ge=0, le=4)
    Fedu: int = Field(..., ge=0, le=4)
    traveltime: int = Field(..., ge=1, le=5)
    studytime: int = Field(..., ge=1, le=5)
    failures: int = Field(..., ge=0, le=3)
    famrel: int = Field(..., ge=1, le=5)
    freetime: int = Field(..., ge=1, le=5)
    goout: int = Field(..., ge=1, le=5)
    Dalc: int = Field(..., ge=1, le=5)
    Walc: int = Field(..., ge=1, le=5)
    health: int = Field(..., ge=1, le=5)
    absences: int = Field(..., ge=0, le=100)

    @model_validator(mode="before")
    def validate_constraints(cls, values):
        for key, value in values.items():
            if key in constraints:
                allowed_values = constraints[key]
                if isinstance(allowed_values, list) and value not in allowed_values:
                    raise ValueError(f"{key} must be one of {allowed_values}")
                elif isinstance(allowed_values, tuple) and not (allowed_values[0] <= value <= allowed_values[1]):
                    raise ValueError(f"{key} must be between {allowed_values[0]} and {allowed_values[1]}")
        return values

# Function to one-hot encode input data
def convert_input_to_binary(input_data, categorical_cols, X_columns):
    input_df = pd.DataFrame([input_data])  # Create a DataFrame from input data
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Align with training data columns
    missing_cols = set(X_columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0

    input_encoded = input_encoded[X_columns]
    return input_encoded

@app.post("/predict")
def predict_grade(input_data: StudentInput):
    """
    Predict the student's final grade based on the input data.
    """
    try:
        # Convert input to dictionary and one-hot encode
        input_dict = input_data.dict()
        encoded_input = convert_input_to_binary(input_dict, categorical_cols, X_columns)

        # Predict using the best model
        prediction = best_model.predict(encoded_input)[0]

        return {"final_grade_prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/columns")
def get_column_details():
    with open(metadata_file, "r") as file:
        metadata = json.load(file)
    return {"columns": metadata}

# Run the app if the script is executed
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
