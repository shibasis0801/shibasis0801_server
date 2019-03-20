from flask import Flask, request
import numpy as np 
import pandas as pd 

from sklearn.externals import joblib

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_household = X [ : , rooms_ix ] / X [ : , household_ix ]
        population_per_household = X [ : , population_ix ] / X [ : , household_ix ]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X [ : , bedrooms_ix ] / X [ : , rooms_ix ]
            return np.c_[ X, rooms_per_household, population_per_household, bedrooms_per_room ]
        else:
            return np.c_[ X, rooms_per_household, population_per_household ]

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

full_pipeline = joblib.load('full_pipeline.pkl')
forest_regressor = joblib.load('forest_regressor_housing.pkl')

def create_dataframe(value_dict):
    columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'ocean_proximity']
    data = np.array([[
        value_dict[column] for column in columns
    ]])
    return pd.DataFrame(data = data, columns = columns)

def transform(dataframe):
    global full_pipeline
    return full_pipeline.transform(dataframe)

def get_predictions(sparse_matrix):
    global forest_regressor
    return forest_regressor.predict(sparse_matrix)

def predict(value_dict):
    df = create_dataframe(value_dict)
    transformed = transform(df)
    return get_predictions(transformed)

app = Flask(__name__)

@app.route('/query-example')
def query_example():
    return "Todo"

@app.route('/form-example')
def form_example():
    return str(predict(request.args.to_dict()))

@app.route('/json-example')
def json_example():
    return "Todo"

if __name__ == "__main__":
    app.run(debug = True, port = 5000)