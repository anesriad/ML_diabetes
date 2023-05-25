# -*- coding: utf-8 -*-

import numpy as np
import pickle

loaded_model = pickle.load(open('/Users/riadanas/Desktop/trained_model.sav', 'rb'))

input_data = (137,43.1,0,2.288) #Example of diabetic person => = 1

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#We used the loaded model for prediction
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')