import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model







class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, state_shape):
        self._input_dim = input_dim  #old, should be deleted everywhere + code adjusted + config changed
        self._state_shape = state_shape
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        # self._model = self._build_model(num_layers, width)
        
    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.expand_dims(state, axis = 0)
        return self._model.predict(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        # print("in predict_batch, shape of states: ", states.shape)
        
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size





class VanillaTrainModel(TrainModel):
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, state_shape):
        super().__init__(num_layers, width, batch_size, learning_rate, input_dim, output_dim, state_shape)
        self._model = self._build_model(num_layers, width)
        
    
    
    def _build_model(self, num_layers, width):
        """
        Build and compile a convolutional deep neural network
        """
        #input layer
        inputs = keras.Input(shape = self._state_shape)
                
        #convolutional layers
        c1 = layers.Conv2D(filters = 128, kernel_size = 4, strides = (2,2), padding = "same", activation = 'relu')(inputs)
        c2 = layers.Conv2D(filters = 128, kernel_size = 4, strides = (2,2), padding = "same", activation = 'relu')(c1)
        c3 = layers.Conv2D(filters = 64, kernel_size = 2, strides = (1,1), padding = "same", activation = 'relu')(c2)
        flat = layers.Flatten()(c3)
        dense = layers.Dense(16, activation='relu')(flat)
        outputs = layers.Dense(self._output_dim, activation='linear')(dense)
        
        model = keras.Model(inputs = inputs, outputs = outputs, name='simple_CNN')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        
        return model
    
    


    
    
    
 
 
 
 
 
 
 
class RNNTrainModel(TrainModel):
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, state_shape, sequence_length, statefulness):
        self._sequence_length = sequence_length
        self._statefulness = statefulness
        super().__init__(num_layers, width, batch_size, learning_rate, input_dim, output_dim, state_shape)
        self._model = self._build_model(num_layers, width)
    
    
    def _build_model(self, num_layers, width):
        """
        Build and compile a deep neural network with convolution as LSTM
        """
        
        
        # sequence_input_shape = (self._sequence_length,) + self._state_shape
        sequence_input_shape = (None,) + self._state_shape
        
        
        # if using the predict model, the batch size is fixed to size 1
        if self._statefulness == False:
            print("enter not stateful model")
            #input layer
            inputs = keras.Input(shape = sequence_input_shape)
        else:
            print("enter stateful model")
            #input layer
            # batch_shape = (1,) + sequence_input_shape
            batch_shape = (1,1) + self._state_shape
            print(batch_shape)
            inputs = keras.Input(batch_shape = batch_shape)
            print("leave stateful model")
        
        
        #input layer
        # inputs = keras.Input(shape = sequence_input_shape)
                
        #convolutional layers
        c1 = layers.TimeDistributed(layers.Conv2D(filters = 128, kernel_size = 4, strides = (2,2), padding = "same", activation = 'relu'))(inputs)
        c2 = layers.TimeDistributed(layers.Conv2D(filters = 128, kernel_size = 4, strides = (2,2), padding = "same", activation = 'relu'))(c1)
        c3 = layers.TimeDistributed(layers.Conv2D(filters = 64, kernel_size = 2, strides = (1,1), padding = "same", activation = 'relu'))(c2)
        flat = layers.TimeDistributed(layers.Flatten())(c3)
        lstm = layers.LSTM(96, activation='tanh', return_sequences=True, stateful = self._statefulness)(flat)
        dense = layers.Dense(16, activation='relu')(lstm)
        outputs = layers.Dense(self._output_dim, activation='linear')(dense)
        
        
        model = keras.Model(inputs = inputs, outputs = outputs, name='CNN_with_LSTM')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        
        # model.summary()
        return model
    
    









# TO DO: change input shape if needed
class TestModel:
    def __init__(self, input_dim, model_path, state_shape):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)
        self._state_shape = state_shape


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.expand_dims(state, axis = 0)
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim