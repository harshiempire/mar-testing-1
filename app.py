#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import streamlit as st
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Car Price Prediction App", page_icon=None, layout='centered', initial_sidebar_state='auto')

# In[ ]:


@st.cache(allow_output_mutation=True)
def load(scaler_path, ohe_path, model_path):
    sc = joblib.load(scaler_path)
    ohe = joblib.load(ohe_path)
    model = joblib.load(model_path)
    return sc , ohe, model


# In[ ]:


def inference(row, cols, scaler, ohe, model):
    print('Inference Function')
    print('Input Row:', row)
    print('Input Columns:', cols)
    
    df = pd.DataFrame([row], columns=cols)
    
    # Get numeric and categorical columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    car_num_cols = list(df.select_dtypes(include=numerics).columns)
    car_cat_cols = list(df.select_dtypes(exclude=numerics).columns)
    
    print('Numeric Columns:', car_num_cols)
    print('Categorical Columns:', car_cat_cols)
    
    # Scale numeric features
    df[car_num_cols] = scaler.transform(df[car_num_cols])
    
    # One-hot encode categorical features
    car_ohe = ohe.transform(df[car_cat_cols])
    
    print('One-Hot Encoded Features Shape:', car_ohe.shape)
    
    # Get feature names using the newer method, fall back to older method if needed
    try:
        feature_names = ohe.get_feature_names_out(input_features=car_cat_cols)
    except AttributeError:
        feature_names = ohe.get_feature_names(input_features=car_cat_cols)
    
    print('Feature Names:', feature_names)
    
    # Create DataFrame with one-hot encoded features
    car_df_ohe = pd.DataFrame(car_ohe, columns=feature_names)
    
    # Combine numeric and one-hot encoded features
    df = df.drop(car_cat_cols, axis=1)
    df = pd.concat([df, car_df_ohe], axis=1)
    
    print('Final DataFrame Shape:', df.shape)
    
    # Make prediction
    price = model.predict(df)[0]
    
    print('Predicted Price:', price)
    
    return price


# In[ ]:


st.title('Car Price Prediction App')
st.write('Predicting Price of a Car based on its features')
image = Image.open('data/car.jpg')
st.image(image, use_column_width=True)
st.write('Please fill in the details of the car under consideration in the left sidebar and click on the button below!')



fuel_type =     st.sidebar.selectbox("Fuel Type", ('diesel','gas'))
aspiration =     st.sidebar.selectbox("Aspiration", ('std','turbo'))
door_number =     st.sidebar.selectbox("Door Number", ('two','four'))
car_body =     st.sidebar.selectbox("Car Body", ('convertible','hardtop','hatchback','sedan','wagon'))
drive_wheel =     st.sidebar.selectbox("Drive Wheel", ('rwd','fwd','4wd'))
engine_location =     st.sidebar.selectbox("Engine Location", ('front','rear'))
wheelbase =   st.sidebar.number_input("Wheel Base", 0.0, 130.0, 86.6, 1.0)
carlength =   st.sidebar.number_input("Car Length", 0.0, 210.0, 141.1, 1.0)
carwidth =   st.sidebar.number_input("Car Width", 0.0, 75.0, 60.3, 1.0)
carheight =   st.sidebar.number_input("Car Length", 0.0, 60.0, 47.8, 1.0)
curbweight =   st.sidebar.number_input("Curb Weight", 0, 4070, 1488, 100)
engine_type =     st.sidebar.selectbox("Engine Type", ('dohc','dohcv','l','ohc','ohcf','ohcv','rotor'))
cylinder_number =     st.sidebar.selectbox("Cylinder Number", ('two','three','four','five','six','eight','twelve'))
enginesize =   st.sidebar.number_input("Engine Size", 0.0, 330.0, 61.0, 10.0)
fuel_system =     st.sidebar.selectbox("Fuel System", ('1bbl','2bbl','4bbl','idi','mfi','mpfi','spdi','spfi'))
boreratio =   st.sidebar.number_input("Bore Ratio", 0.0, 4.0, 2.54, 0.1)
stroke =   st.sidebar.number_input("Stroke", 0.0, 4.5, 2.07, 0.1)
compression_ratio =   st.sidebar.number_input("Compression Ratio", 0.0, 23.0, 7.0, 0.5)
horsepower =       st.sidebar.slider("Horsepower", 0, 300, 48, 1)
peakrpm =       st.sidebar.slider("Peak RPM", 0, 6600, 4150, 1)
citympg =       st.sidebar.slider("City MPG", 0, 49, 13, 1)
highwaympg =       st.sidebar.slider("Highway MPG", 0, 54, 16, 1)



row = [fuel_type, aspiration, door_number, car_body, drive_wheel, engine_location, wheelbase, carlength, carwidth, carheight, curbweight, engine_type, cylinder_number, enginesize, fuel_system, boreratio, stroke, compression_ratio, horsepower, peakrpm, citympg, highwaympg]


cols = ['fueltype', 'aspiration',
   'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase',
   'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype',
   'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke',
   'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']

# In[ ]:


if (st.button('Predict Car Price')):    
    sc, ohe, model = load('models/scaler.joblib', 'models/ohe.joblib', 'models/XGBoost.joblib')
    result = inference(row, cols, sc, ohe, model)
    st.write(result)
