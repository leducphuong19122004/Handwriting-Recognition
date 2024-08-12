from keras.api.layers import Dense, LSTM, Reshape, Input, Conv1D, Conv2D, MaxPool2D, Lambda, Add, Activation, Bidirectional
from keras.api.layers import BatchNormalization
from keras.api.models import Model
from keras.api.activations import relu, sigmoid, softmax
import tensorflow as tf
import keras.api.ops as ops
from keras.api.callbacks import ModelCheckpoint
from generateInputForModel import InputGenerator
import keras
import math

import numpy as np

from loadDataset import DataLoader
from preprocess import Preprocess

class CNN_and_RNN:
    def __init__(self) -> None:
        # set up cnn layers
        # input wit shape of height=32 and width=128
        self.inputs = Input(shape=(64,128,1), name="image")
        
        # convolution layer with kernel size (3,3)
        layers = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(self.inputs)
        layers = BatchNormalization()(layers)
        layers = Activation('relu')(layers)
        layers = MaxPool2D(pool_size=(2, 2))(layers)
        
        layers = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(layers)
        layers = BatchNormalization()(layers)
        layers = Activation('relu')(layers)
        layers = MaxPool2D(pool_size=(2, 2))(layers)
        
        layers = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(layers)
        layers = BatchNormalization()(layers)
        layers = Activation('relu')(layers)
        layers = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(layers)
        layers = BatchNormalization()(layers)
        layers = Activation('relu')(layers)
        layers = MaxPool2D(pool_size=(1, 2))(layers)  

        layers = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(layers)
        layers = BatchNormalization()(layers)
        layers = Activation('relu')(layers)
        layers = Conv2D(512, (3, 3), padding='same')(layers)
        layers = BatchNormalization()(layers)
        layers = Activation('relu')(layers)
        layers = MaxPool2D(pool_size=(1, 2))(layers)

        layers = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal')(layers)
        layers = BatchNormalization()(layers)
        layers = Activation('relu')(layers)

        # CNN to RNN
        layers = Reshape(target_shape=((32, 2048)))(layers)
        
        # bidirectional LSTM layers with units=128
        blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(layers)
        blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)
        
        self.outputs = Dense(len(Preprocess().char_list)+1, activation = 'softmax')(blstm_2) # len(Preprocess().char_list)+1 = 80
        # model to be used at test time
        self.act_model = Model(self.inputs, self.outputs)

class CTC:
    def __init__(self, inputs, outputs) -> None:
         # ctc definition part   
        self.labels = Input(name='the_labels', shape=[19], dtype='float32') # 19 = max_text_length
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        # A CTC loss function requires four arguments to compute the loss, predicted outputs, 
        # ground truth labels, input sequence length to LSTM and ground truth label length
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args # y_pred - outputs    
            return self.ctc_batch_cost(labels, y_pred, input_length, label_length) 
        
        # The Lambda layer is normally used to implement a custom function, in this case is custom loss function
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, self.labels, self.input_length, self.label_length])
        #model to be used at training time
        self.model = Model(inputs=[inputs, self.labels, self.input_length, self.label_length], outputs=loss_out)
        

    def ctc_batch_cost(self, y_true, y_pred, input_length, label_length):
        label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
        input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
        sparse_labels = ops.cast(
            self.ctc_label_dense_to_sparse(labels = y_true, label_lengths = label_length), dtype="int32"
        )

        y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

        return ops.expand_dims(
            tf.compat.v1.nn.ctc_loss(
                inputs=y_pred, labels=sparse_labels, sequence_length=input_length
            ),
            1,
        )
    
    def ctc_label_dense_to_sparse(self, labels, label_lengths):
        label_shape = ops.shape(labels)
        num_batches_tns = ops.stack([label_shape[0]])
        max_num_labels_tns = ops.stack([label_shape[1]])

        def range_less_than(old_input, current_input):
            return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(
                max_num_labels_tns, current_input
            )

        init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
        dense_mask = tf.compat.v1.scan(
            range_less_than, label_lengths, initializer=init, parallel_iterations=1
        )
        dense_mask = dense_mask[:, 0, :]

        label_array = ops.reshape(
            ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape
        )
        label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

        batch_array = ops.transpose(
            ops.reshape(
                ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),
                tf.reverse(label_shape, [0]),
            )
        )
        batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
        indices = ops.transpose(
            ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1])
        )

        vals_sparse = tf.compat.v1.gather_nd(labels, indices)

        return tf.SparseTensor(
            ops.cast(indices, dtype="int64"), 
            vals_sparse, 
            ops.cast(label_shape, dtype="int64")
        )

class CRNN_CTCModel:
    def __init__(self) -> None:
        cnn_rnn = CNN_and_RNN()
        ctc = CTC(cnn_rnn.inputs, cnn_rnn.outputs)
        self.model = ctc.model
    
    def train(self):
        # Now that the loss function is implemented, the second part of the code bypasses the Keras built-in loss functions
        # by returning the prediction tensor y_pred as is. 
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
        filepath = "CRNN_model.keras"
        # ModelCheckpoint callback is used in conjunction with training using model.fit() to save a model or weights
        # so the model or weights can be loaded later to continue the training from the state saved.
        model_checkpoint_callback = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        # Load train data set and validation data set
        data_loader = DataLoader()

        train_data = InputGenerator(data_loader.train_data_set)
        train_data.build_data()
        train_x, train_y = next(train_data.next_batch())

        validation_data = InputGenerator(data_loader.validation_data_set)
        validation_data.build_data()
        val_x, val_y = next(validation_data.next_batch())

        self.model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), steps_per_epoch=math.ceil(train_data.n/train_data.batch_size), validation_steps=math.ceil(validation_data.n/validation_data.batch_size), epochs=30, verbose=1, callbacks=[model_checkpoint_callback])