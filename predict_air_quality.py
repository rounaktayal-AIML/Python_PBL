import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    with open("air_quality.pkl", "rb") as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
except Exception as e:
    print(f"Error loading air_quality.pkl: {e}")
    exit()

all_features = ['CO', 'NO2', 'Temperature', 'Humidity']

print(" Air Quality Predictor ")
print(f"Please enter {len(all_features)} values separated by spaces.")
print("Order: CO  NO2  Temperature  Humidity")

values = input("\nEnter values: ")
input_list = values.split()

if len(input_list) != 4:
    print(f"\n Expected 4 values, but you entered {len(input_list)}")
else:
    try:
        processed = [float(v) for v in input_list]

        input_df = pd.DataFrame([processed], columns=all_features)
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)

        print(f"\n PREDICTED AIR QUALITY: {prediction[0]}")

    except ValueError as v:
        print(f" Data Error: Could not convert input to number. ({v})")
    except Exception as e:
        print(f" Unexpected Error: {e}")