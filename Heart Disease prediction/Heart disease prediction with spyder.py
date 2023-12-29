# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C://Users//sudha//Downloads//Heart disease prediction bot trained_model.sav', 'rb'))


input_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)

# changing data to numpy array 
input_data_as_numpy_array= np.asarray(input_data)

#reshape the array ass we are predicting for one instance
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshape)
print(prediction)

if (prediction[0]==0):
  print('The person does not have a Heart Disease')
else:
  print('The person has Heart Disease')