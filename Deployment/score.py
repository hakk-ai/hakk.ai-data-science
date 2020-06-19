import json
import numpy as np
import joblib
from azureml.core.model import Model
import xgboost
import pandas as pd

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

def haversine_km(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def haversine_m(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 3956 * c
    return m

def preprocessing(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data['timestamp'] = pd.DatetimeIndex(data.timestamp)

    data['is_weekend'] = np.where(data['day_of_week'].isin([5,6]), 1, 0)
    data['is_weekday'] = np.where(data['day_of_week'].isin([5,6]), 0, 1)

    data['is_wee_hours'] = np.where(data['hour_of_day'].isin([17,18,19,20,21]), 1, 0)

    data['is_rush_hours_morning'] = np.where(data.timestamp.dt.strftime('%H:%M:%S').between('11:30:00', '01:30:00'), 1, 0)
    data['is_rush_hours_evening'] = np.where(data.timestamp.dt.strftime('%H:%M:%S').between('09:00:00', '12:00:00'), 1, 0)

    data['sin_hour_of_day'] = np.sin(2*np.pi*data.hour_of_day/24)
    data['cos_hour_of_day'] = np.cos(2*np.pi*data.hour_of_day/24)
    data['sin_day_of_week'] = np.sin(2*np.pi*data.day_of_week/7)
    data['cos_day_of_week'] = np.cos(2*np.pi*data.day_of_week/7)


    data['haversine_km'] = haversine_km(data['longitude_origin'], data['latitude_origin'], 
                                 data['longitude_destination'], data['latitude_destination'])

    data['haversine_m'] = haversine_m(data['longitude_origin'], data['latitude_origin'], 
                                 data['longitude_destination'], data['latitude_destination'])


    from sklearn.decomposition import PCA
    coords = np.vstack((data[['latitude_origin', 'longitude_origin']].values,
                    data[['latitude_destination', 'longitude_destination']].values))
    
    pca = PCA().fit(coords)

    data['pickup_pca0'] = pca.transform(data[['latitude_origin', 'longitude_origin']])[:, 0]
    data['pickup_pca1'] = pca.transform(data[['latitude_origin', 'longitude_origin']])[:, 1]
    data['dropoff_pca0'] = pca.transform(data[['latitude_destination', 'longitude_destination']])[:, 0]
    data['dropoff_pca1'] = pca.transform(data[['latitude_destination', 'longitude_destination']])[:, 1]

    data = data.drop(['timestamp'], axis=1)
    
    return data

def init():
    global model

    model_path = Model.get_model_path(model_name='finalized_model')
    
    model = joblib.load(model_path)

    
input_sample = pd.DataFrame(data=[{
    "latitude_origin": -6.141255,
    "longitude_origin": 106.692710,
    "latitude_destination": -6.141150,
    "longitude_destination": 106.693154,
    "timestamp": 1590487113,
    "hour_of_day": 9,
    "day_of_week": 1
}])

# This is an integer type sample. Use the data type that reflects the expected result.
output_sample = np.array([360.00])

# To indicate that we support a variable length of data input,
# set enforce_shape=False
@input_schema('data', PandasParameterType(input_sample, enforce_shape=False))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        #preprocessing
        data = preprocessing(data)
        #result
        result = model.predict(data)    
        return result.tolist()
    
    except Exception as e:
        result = str(e)
        return result
