import numpy as np
import pandas as pd
import pickle
import feature_eng

def load_model(file_path):
    with open(file_path, mode='rb') as my_file:
        model = pickle.load(my_file)
    return model

def perform_diabetes_test(test_data):
    model = load_model('models/RF.pkl')
    test_label = model.predict(test_data)
    if test_label == 0:
        return "Your Blood Sugar is Low"
    elif test_label == 1:
        return "Your Blood Sugar is Medium"
    else:
        return "Your Blood sugar is High"

def perform_feature_selection(test_data):
    selector = load_model("models/selector.pkl")
    names = test_data.columns[selector.get_support()]
    return pd.DataFrame(data=selector.transform(test_data), columns=names)

def perform_bgl_test(test_data):
    scaler = load_model("models/coll_scaler.pkl")
    features = scaler.transform(test_data)
    bgl_model = load_model(file_path='models/AdaBoost.pkl')
    bgl_value = bgl_model.predict(features)[0]
    return np.round(bgl_value, 2)

def remove_irrelevant_data(df):
    # Read the CSV file into a DataFrame, skipping the first 3 rows and setting the 4th row as header
    df.columns = ['H', 'MQ138', 'MQ2', 'SSID', 'T', 'TGS2600', 'TGS2602', 'TGS2603', 'TGS2610', 'TGS2611', 'TGS2620', 'TGS822', 'Device', 'Time']
    df = df.drop(['SSID', 'Device', 'H', 'T', 'Time'], axis=1)
    return df.reset_index(drop=True)

def generate_data(sensors_data, body_vitals):
    cleaned_df = remove_irrelevant_data(sensors_data)
    features_df = feature_eng.generate_features(df=cleaned_df)
    final_df = pd.concat([body_vitals, features_df], axis=1)
    return final_df
