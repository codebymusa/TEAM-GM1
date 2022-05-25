"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    df_test = pd.read_csv('df_test.csv', index_col = 0)
    df = pd.read_csv('df_train.csv', index_col=0)
    
    # remove missing values/ features
    df_new = df
    df_new['Valencia_pressure'].fillna(df_new['Valencia_pressure'].median(), inplace = True)
    df_test['Valencia_pressure'].fillna(df_test['Valencia_pressure'].median(), inplace = True)

    #converting categorical variables to numerical
    df_new['time'] = pd.to_datetime(df_new['time'])
    df_test['time'] = pd.to_datetime(df_test['time'])

    df_new["Valencia_wind_deg"] = df_new['Valencia_wind_deg'].str.extract('(\d+)')
    df_new['Valencia_wind_deg'] = pd.to_numeric(df_new['Valencia_wind_deg'])

    df_test["Valencia_wind_deg"] = df_test['Valencia_wind_deg'].str.extract('(\d+)')
    df_test['Valencia_wind_deg'] = pd.to_numeric(df_test['Valencia_wind_deg'])

    df_new["Seville_pressure"] = df_new['Seville_pressure'].str.extract('(\d+)')
    df_new['Seville_pressure'] = pd.to_numeric(df_new['Seville_pressure'])
    
    df_test["Seville_pressure"] = df_test['Seville_pressure'].str.extract('(\d+)')
    df_test['Seville_pressure'] = pd.to_numeric(df_test['Seville_pressure'])
    
    df_new['Year']  = df_new['time'].astype('datetime64').dt.year
    df_new['Month_of_year']  = df_new['time'].astype('datetime64').dt.month
    df_new['Week_of_year'] = df_new['time'].astype('datetime64').dt.weekofyear
    df_new['Day_of_year']  = df_new['time'].astype('datetime64').dt.dayofyear
    df_new['Day_of_month']  = df_new['time'].astype('datetime64').dt.day
    df_new['Day_of_week'] = df_new['time'].astype('datetime64').dt.dayofweek
    df_new['Hour_of_week'] = ((df_new['time'].astype('datetime64').dt.dayofweek) * 24 + 24) - (24 - df_new['time'].astype('datetime64').dt.hour)
    df_new['Hour_of_day']  = df_new['time'].astype('datetime64').dt.hour

    df_test['Year']  = df_test['time'].astype('datetime64').dt.year
    df_test['Month_of_year']  = df_test['time'].astype('datetime64').dt.month
    df_test['Week_of_year'] = df_test['time'].astype('datetime64').dt.weekofyear
    df_test['Day_of_year']  = df_test['time'].astype('datetime64').dt.dayofyear
    df_test['Day_of_month']  = df_test['time'].astype('datetime64').dt.day
    df_test['Day_of_week'] = df_test['time'].astype('datetime64').dt.dayofweek
    df_test['Hour_of_week'] = ((df_test['time'].astype('datetime64').dt.dayofweek) * 24 + 24) - (24 - df_test['time'].astype('datetime64').dt.hour)
    df_test['Hour_of_day']  = df_test['time'].astype('datetime64').dt.hour

    df_train = df_new.drop('time',axis = 1)
    df_test1 = df_test.drop('time', axis = 1)
    return df_train, df_test1

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
