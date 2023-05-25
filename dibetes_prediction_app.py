#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:11:45 2023

@author: riadanas
"""

import numpy as np
import pickle #used for loading the ML model
import streamlit as st #Used for deployment

loaded_model = pickle.load(open('/Users/riadanas/Desktop/trained_model.sav', 'rb'))

def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    #We used the loaded model for prediction
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    #Title of the app
    st.title('Diabetes prediction App')
    
    #input data from user
    Glucose = st.number_input('Levels of glucose')
    BMI = st.number_input('Body mass index')
    Pregnancies = st.number_input("Number of pregnancies")
    DiabetesPedigreeFunction = st.number_input('Diabetes pedigree function levels')
    
    
    
    #Output stored in 'diagnosis variable'
    diagnosis = ''
    
    
    #button creation for diagnosis
    if st.button('Diabetes test result'):
        diagnosis = diabetes_prediction([Glucose, BMI, Pregnancies, DiabetesPedigreeFunction])
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()