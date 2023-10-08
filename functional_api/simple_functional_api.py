# Importing libraries
import os
import pandas as pd
import numpy as np
import tensorflow as tf

# necessary functions for functional api in tf
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model

# Create the first functional api of keras
input = Input(shape=(28, 28))
# Defining subsequent layers
x = Flatten()(input)
x = Dense(128, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

# Creating the first model
func_model = Model(inputs=input, outputs=predictions)
# func_model is an object or instance of Model class of Keras API
print(func_model)