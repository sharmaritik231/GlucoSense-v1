import numpy as np
import pandas as pd
import pickle
import feature_eng
from sklearn.base import BaseEstimator, TransformerMixin

def load_model(file_path):
    with open(file_path, mode='rb') as my_file:
        model = pickle.load(my_file)
    return model

def perform_diabetes_test(test_data):
    model = load_model('models/stack.pkl')
    test_label = model.predict(test_data)
    if test_label == 0:
        return "Non-diabetic"
    elif test_label == 1:
        return "Pre-diabetic"
    else:
        return "Highly diabetic"

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

class RemoveHighlyCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        # Calculate the correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than the specified threshold
        self.to_drop_ = [column for column in upper.columns if any(upper[column] >= self.threshold)]
        return self

    def transform(self, X, y=None):
        # Drop the highly correlated features
        if self.to_drop_ is not None:
            return X.drop(columns=self.to_drop_)
        else:
            return X

    def fit_transform(self, X, y=None, **fit_params):
        # Use the fit method and then the transform method
        return super().fit_transform(X, y, **fit_params)

    def __getstate__(self):
        # Return the object's state as a dictionary
        return self.__dict__

    def __setstate__(self, state):
        # Restore the object's state from the dictionary
        self.__dict__.update(state)
