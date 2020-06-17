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
    def __init__(self, batch_size, learning_rate, output_dim, state_shape):
        self._state_shape = state_shape
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate

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
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size





class VanillaTrainModel(TrainModel):
    def __init__(self, batch_size, learning_rate, output_dim, state_shape):
        super().__init__(batch_size, learning_rate, output_dim, state_shape)
        self._model = self._build_model()
        
    
    
    def _build_model(self):
        """
        Build and compile a convolutional deep neural network
        """
                
        #convolutional part
        conv_inputs = keras.Input(shape = self._state_shape[0])
        c1 = layers.Conv2D(filters = 32, kernel_size = 2, strides = (2,2), padding = "same", activation = 'relu')(conv_inputs)
        c2 = layers.Conv2D(filters = 64, kernel_size = 2, strides = (2,2), padding = "same", activation = 'relu')(c1)
        c3 = layers.Conv2D(filters = 32, kernel_size = 2, strides = (1,1), padding = "same", activation = 'relu')(c2)
        flat = layers.Flatten()(c3)


        #current green phase layer
        phase_inputs = keras.Input(shape = (self._state_shape[1],))
        
        #elapsed green time layer
        elapsed_time_inputs = keras.Input(shape = (self._state_shape[2],))
        
        
        #combine elapsed time and green time layer
        combined_green = layers.concatenate([phase_inputs, elapsed_time_inputs])
        green_dense = layers.Dense(10, activation='relu')(combined_green)
        
        #combine green layer with conv layer
        all_combined = layers.concatenate([green_dense, flat])
        dense = layers.Dense(32, activation='relu')(all_combined)
        dense = layers.Dense(16, activation='relu')(dense)
        outputs = layers.Dense(self._output_dim, activation='linear')(dense)
        
        model = keras.Model(inputs = [conv_inputs, phase_inputs, elapsed_time_inputs], outputs = outputs, name='simple_CNN')       
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        
        return model
    
    
    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        # print("predict one state shape: ", state.shape)

        s0 = np.expand_dims(state[0], axis = 0)
        s1 = np.expand_dims(state[1], axis = 0)
        s2 = np.expand_dims(state[2], axis = 0)
        
        
        # state = np.expand_dims(state, axis = 0)
        # return self._model.predict(state)
        
        return self._model.predict([s0,s1,s2])
        # prediction = self._model.predict([s0,s1,s2])
        # print("predicted chosen action: ", prediction)
        # return prediction
    
    
    
    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        #states is a batch with shape (#samples, 3): [conv shape, phase shape, elapsed shape] in each row
        
        # print("in predict_batch, shape of states: ", states.shape)
        # print("in predict batch, shape of entries: ", states[:,0].shape, states[:,1].shape)
        
        # print("in predict batch, shape of states 1: ", states[:,1])
        
        
        
        s0 = np.concatenate(states[:,0]).reshape((self._batch_size, ) + self._state_shape[0])
        # print("shapes after reshaping:  s0: ", s0.shape, s0.dtype )
        
        
        s1 = np.concatenate(states[:,1]).reshape((self._batch_size, self._state_shape[1]))
        # print("shapes after reshaping:  s1: ", s1.shape, s1.dtype)
        
        s2 = np.array(states[:,2], dtype=np.float)
        # print("shapes after reshaping:  s2: ", s2.shape, s2.dtype)

        return self._model.predict([s0,s1,s2])
        
        # return self._model.predict(states)
        
        
        # print("states 0 shape: ", states[0].shape)
        # print("states 1 shape: ", states[1].shape)
        # print("states 2 shape: ", states[2].shape)
        
        # return self._model.predict([states[0], states[1], states[2]])


    
    
    
 
 
 
 
 
 
 
class RNNTrainModel(TrainModel):
    def __init__(self, batch_size, learning_rate, output_dim, state_shape, sequence_length, statefulness):
        self._sequence_length = sequence_length
        self._statefulness = statefulness
        super().__init__(batch_size, learning_rate, output_dim, state_shape)
        self._model = self._build_model()
    
    
    def _build_model(self):
        """
        Build and compile a deep neural network with convolution as LSTM
        """
        # expand the input dimensions to match the required input shape
        # if using the predict model, the batch size is fixed to size 1
        if self._statefulness == False: 
            sequence_state_shape = []
            sequence_state_shape.append((None,)+self._state_shape[0])
            sequence_state_shape.append((None, self._state_shape[1]))
            sequence_state_shape.append((None, self._state_shape[2]))
            
            conv_inputs = keras.Input(shape = sequence_state_shape[0])
            phase_inputs = keras.Input(shape = sequence_state_shape[1])
            elapsed_time_inputs = keras.Input(shape = sequence_state_shape[2])
        else:
            batch_state_shape = []
            batch_state_shape.append((1,1)+ self._state_shape[0])
            batch_state_shape.append((1,1, self._state_shape[1]))
            batch_state_shape.append((1,1, self._state_shape[2]))
            
            conv_inputs = keras.Input(batch_shape = batch_state_shape[0])
            phase_inputs = keras.Input(batch_shape = batch_state_shape[1])
            elapsed_time_inputs = keras.Input(batch_shape = batch_state_shape[2])
        
        #conv layers        
        c1 = layers.TimeDistributed(layers.Conv2D(filters = 32, kernel_size = 2, strides = (2,2), padding = "same", activation = 'relu'))(conv_inputs)
        c2 = layers.TimeDistributed(layers.Conv2D(filters = 64, kernel_size = 2, strides = (2,2), padding = "same", activation = 'relu'))(c1)
        c3 = layers.TimeDistributed(layers.Conv2D(filters = 32, kernel_size = 2, strides = (1,1), padding = "same", activation = 'relu'))(c2)
        flat = layers.TimeDistributed(layers.Flatten())(c3)
        
        #combine elapsed time and green time layer
        combined_green = layers.concatenate([phase_inputs, elapsed_time_inputs])
        green_dense = layers.TimeDistributed(layers.Dense(10, activation='relu'))(combined_green)
        
        #combine green layer with conv layer, LSTM and output 
        all_combined = layers.concatenate([green_dense, flat])
        lstm = layers.LSTM(96, activation='tanh', return_sequences=True, stateful = self._statefulness)(all_combined)
        dense = layers.Dense(32, activation='relu')(lstm)
        dense = layers.Dense(16, activation='relu')(dense)
        outputs = layers.Dense(self._output_dim, activation='linear')(dense)
        
        model = keras.Model(inputs = [conv_inputs, phase_inputs, elapsed_time_inputs], outputs = outputs, name='CNN_with_LSTM')       
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
                
        return model
    
    
    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        # print("predict one state shape: ", state.shape)

        s0 = np.expand_dims(state[0], axis = (0,1))
        s1 = np.expand_dims(state[1], axis = (0,1))
        s2 = np.expand_dims((state[2], ), axis = (0,1))
        
        return self._model.predict([s0,s1,s2])

    
    
    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        #states is a batch with shape (#samples, sequence length, 3): [conv shape, phase shape, elapsed shape] in each row
        
        # print("in predict_batch, shape of states: ", states.shape)
        # print("in predict batch, shape of entries: ", states[:,0].shape, states[:,1].shape)
        
        # print("in predict batch, shape of states 1: ", states[:,1])
        
        
        
        s0 = np.concatenate(np.concatenate(states[:,:,0])).reshape((self._batch_size, self._sequence_length) + self._state_shape[0])
        # print("shapes after reshaping:  s0: ", s0.shape, s0.dtype )

        s1 = np.concatenate(np.concatenate(states[:,:,1])).reshape((self._batch_size, self._sequence_length, self._state_shape[1]))
        # print("shapes after reshaping:  s1: ", s1.shape, s1.dtype)
        
        s2 = np.expand_dims(np.array(states[:,:,2], dtype=np.float), axis = 2)
        # print("shapes after reshaping:  s2: ", s2.shape, s2.dtype)

        return self._model.predict([s0,s1,s2])
        










class TestModel:
    def __init__(self, model_path, state_shape):
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

        #if it is a recurrent model:
        if len(self._model.layers[0].input.shape) > len(self._state_shape)+1:
            s0 = np.expand_dims(state[0], axis = (0,1))
            s1 = np.expand_dims(state[1], axis = (0,1))
            s2 = np.expand_dims((state[2], ), axis = (0,1))  
        #if it is not a recurrent model
        else:
            s0 = np.expand_dims(state[0], axis = 0)
            s1 = np.expand_dims(state[1], axis = 0)
            s2 = np.expand_dims(state[2], axis = 0)
         
         
            
        # state = np.expand_dims(state, axis = 0)
        # return self._model.predict(state)

        return self._model.predict([s0,s1,s2])
        
        
        
        
        
        