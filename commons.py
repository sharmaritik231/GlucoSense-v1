import numpy as np
import pandas as pd
import pickle
import feature_eng
import constants

def load_model(file_path):
    with open(file_path, mode='rb') as my_file:
        model = pickle.load(my_file)
    return model

def perform_diabetes_test(test_data):
    model = load_model('models/stack.pkl')
    test_label = model.predict(test_data)
    if test_label == 0:
        return "Your Blood Sugar is Low"
    elif test_label == 1:
        return "Your Blood Sugar is Medium"
    else:
        return "Your Blood sugar is High"

def perform_bgl_test(test_data):
    scaler = load_model("models/coll_scaler.pkl")
    features = scaler.transform(test_data)
    bgl_model = load_model(file_path='models/AdaBoost.pkl')
    bgl_value = bgl_model.predict(features)[0]
    return np.round(bgl_value, 2)

def remove_irrelevant_data(file_path):
    df = pd.read_csv(file_path, skiprows=3).iloc[:, 1:]
    df.columns = ['H', 'MQ138', 'MQ2', 'SSID', 'T', 'TGS2600', 'TGS2602', 'TGS2603', 'TGS2610', 'TGS2611', 'TGS2620', 'TGS822', 'Device', 'Time']
    df = df.drop(['SSID', 'Device', 'H', 'T', 'Time'], axis=1)
    return df.reset_index(drop=True)

def generate_data(file_path, sensor_cols=constants.SENSOR_COLS):
    cleaned_df = remove_irrelevant_data(file_path)
    features_df = feature_eng.generate_features(df=cleaned_df, sensor_cols=sensor_cols)
    return features_df

# Removed unused functions and tkinter-related code
