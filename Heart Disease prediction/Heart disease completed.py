# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 04:12:37 2023

@author: sudha
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C://Users//sudha//Downloads//Heart disease prediction bot trained_model.sav', 'rb'))


# creating a function for prediction

def heart_disease_prediction(input_data):
    

    # changing data to numpy array 
    input_data_as_numpy_array= np.asarray(input_data)

    #reshape the array ass we are predicting for one instance
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    if (prediction[0]==0):
      return'The person does not have a Heart Disease'
    else:
      return'The person has Heart Disease'
      
      
      
def main():
    
    
    # giving a title
    st.title('Heart Disease Prediction Web App')
    
    
    # getting the inpput data from the user
	
    
    Age = st.text_input("Age")
    Sex = st.text_input("Gender")
    Chest_pain = st.text_input("Chest pain type")
    Blood_Pressure = st.text_input("Blood Pressure")
    Cholestrol = st.text_input("Cholestrol")
    FBS = st.text_input("FBS")
    ECG = st.text_input("ECG")
    MaxHR = st.text_input("MaxHR")
    Exercise_angina = st.text_input("Exercise angina")
    ST_depression = st.text_input("STdepression")
    Slope_of_ST = st.text_input("Slope of ST")
    Number_of_vessels_fluro = st.text_input("Number of vessels fluro")
    Thallium = st.text_input("Thallium")
    
    
    # code for Prediction
    diagnosis =""
    
    # creating a button for Prediction
    
    if st.button("Heart Disease Test Result"):
        diagnosis=heart_disease_prediction([Age,Sex,Chest_pain,Blood_Pressure,Cholestrol,FBS,ECG,MaxHR,Exercise_angina,ST_depression,Slope_of_ST,Number_of_vessels_fluro,Thallium])
    
    st.success(diagnosis)





if __name__ == '__main__':
    main()