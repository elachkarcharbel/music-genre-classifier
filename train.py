import os
import math
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, Lambda, Dropout, Activation, LSTM, GRU, \
        TimeDistributed, Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D, \
        BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, \
        ZeroPadding2D, Reshape, merge, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.layers.local import LocallyConnected1D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from keras import backend as K
from keras.models import Model
from keras.models import load_model  
from sklearn.model_selection import train_test_split


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow.compat.v1.keras.backend as KTF
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

import librosa as librosa
import os
import ast
import librosa.display
import numpy as np
import pandas as pd
from tqdm import tqdm
from subprocess import Popen, PIPE, STDOUT

from libs import utils, dataset

sampling_rate = 44100

path_to_training_labels = "preprocessing/CSV_FILENAME.csv"

labels = pd.read_csv(path_to_training_labels)
print(labels.head())

genres = utils.get_genre()

print(genres)

train_x, train_y, validation_x, \
validation_y, test_x, test_y = dataset.create_dataset_from_slices(3000, genres, 128, 0.1, 0.1)

# DEFINE BASE CONV BLOCK

def base_conv_block(num_conv_filters, kernel_size):
    def f(input_):
        x = BatchNormalization()(input_)
        x = Activation('relu')(x)
        out = Convolution2D(num_conv_filters, kernel_size, padding='same')(x)
        return out
    return f

# DEFINE MULTI SCALE BLOCK

def multi_scale_block(num_conv_filters):
    def f(input_):

        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input_)
  
        branch3x3 = base_conv_block(num_conv_filters, 1)(input_)
        branch3x3 = base_conv_block(num_conv_filters, 3)(branch3x3)
          
        branch3x3_2 = base_conv_block(num_conv_filters, 1)(input_)
        branch3x3_2 = base_conv_block(num_conv_filters, (1, 7))(branch3x3_2)
        branch3x3_2 = base_conv_block(num_conv_filters, (7, 1))(branch3x3_2)
        branch3x3_2 = base_conv_block(num_conv_filters, 3)(branch3x3_2)
         
        out = concatenate([branchpool,branch3x3,branch3x3_2], axis=-1)
        return out
    return f
# DEFINE DENSE BLOCK

def dense_block(num_dense_blocks, num_conv_filters):
    def f(input_):
        x = input_
        for _ in range(num_dense_blocks):
            out = multi_scale_block(num_conv_filters)(x)
            x = concatenate([x, out], axis=-1)
        return x
    return f

# DEFINE TRANSITION BLOCK

def transition_block(num_conv_filters):
    def f(input_):
        x = BatchNormalization()(input_)
        x = Activation('relu')(x)
        x = Convolution2D(num_conv_filters, 1)(x)
        out = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return out
    return f


# DEFINE MULTI SCALE LEVEL CNN

def multi_scale_level_cnn(input_shape, num_dense_blocks, num_conv_filters, num_classes):
    model_input = Input(shape=input_shape)
    
    x = Convolution2D(num_conv_filters, 3, padding='same')(model_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    
    x = dense_block(num_dense_blocks, num_conv_filters)(x)
    x = transition_block(num_conv_filters)(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    
    model_output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=model_input, outputs=model_output)
    
    return model

# DEFINE  PROCESS DATA FOR CONV2D

def process_data_for_conv2D(X, resize_shape=None):
    X_conv2D = []
    for sample in X:
        sample = np.reshape(sample, newshape=(sample.shape[0], sample.shape[1], 1))
        if resize_shape:
            sample = resize(sample, output_shape=resize_shape)
        X_conv2D.append(sample)
    return np.array(X_conv2D, dtype=np.float32)

def data_iter(X, y, batch_size):
    num_samples = X.shape[0]
    idx = list(range(num_samples))
    while True:
        for i in range(0, num_samples, batch_size):
            j = idx[i:min(i+batch_size, num_samples)]
            yield X[j, :], y[j, :]


#check the architecture of the net
model = multi_scale_level_cnn(input_shape=(128, 128, 3), 
                              num_dense_blocks=6, num_conv_filters=32, num_classes=8)
model.summary()

# STARTING THE TRAINING

print("starting the training")

#without data argumatent
k_fold = 10
num_classes = 8

train_size = 0.8
val_size = 0.1
test_size = 0.1

epochs = 100
batch_size = 8
lr = 0.01
file_name0 = 'results/fma_model_inceptionv4_B.hdf5'
path  = 'results/log/'
csv_name0 = 'fma_results.csv'
train_loss_record = []
train_acc_record = []
val_loss_record = []
val_acc_record = []
test_loss_record = []
test_acc_record = []
for i in range(k_fold):
    print('Start %d fold training' % (i+1))
    #X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_melspec, y, train_size=train_size, 
     #                                                                     val_size=val_size, test_size=test_size)
    file_name = 'results/model/'+str(i)+'_fold_'+file_name0
#     log_path  = path+str(i)+'_fold_'+'tensorboard_log'
    csv_path  = path+str(i)+'_fold_'+ csv_name0
    lr_change = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=0.000)
    model_checkpoint = ModelCheckpoint(file_name, monitor='val_acc', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=10, mode='min')
    csv_logger = CSVLogger(csv_path)
#     tb_cb = TensorBoard(log_dir=log_path, write_images=1, histogram_freq=1)
    callbacks =[lr_change, model_checkpoint, early_stopping,csv_logger]
    opt = Adam(lr=lr)
    model = multi_scale_level_cnn(input_shape=(128, 128, 3), 
                              num_dense_blocks=6, num_conv_filters=32, num_classes=num_classes)
    model.compile(
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                optimizer=opt)
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, 
              validation_data=(validation_x, validation_y), verbose=1,
              callbacks=callbacks)
    model_best = load_model(file_name)
    train_loss, train_acc = model_best.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
    val_loss, val_acc = model_best.evaluate(validation_x, validation_y, batch_size=batch_size, verbose=0)
    test_loss, test_acc = model_best.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    
    train_loss_record.append(train_loss)
    train_acc_record.append(train_acc)
    val_loss_record.append(val_loss)
    val_acc_record.append(val_acc)
    test_loss_record.append(test_loss)
    test_acc_record.append(test_acc)
    print('\n\n%d fold train loss %.4f train acc %.4f, val loss %.4f val acc %.4f, test loss %.4f test acc %.4f\n\n' % 
          (i+1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))
train_loss_avg = np.mean(np.array(train_loss_record))
train_acc_avg = np.mean(np.array(train_acc_record))
val_loss_avg = np.mean(np.array(val_loss_record))
val_acc_avg = np.mean(np.array(val_acc_record))
test_loss_avg = np.mean(np.array(test_loss_record))
test_acc_avg = np.mean(np.array(test_acc_record))
print('\n\n%d fold train loss avg %.4f train acc avg %.4f, val loss avg %.4f val acc avg %.4f, test loss avg %.4f test acc avg %.4f' % 
  (k_fold, train_loss_avg, train_acc_avg, val_loss_avg, val_acc_avg, test_loss_avg, test_acc_avg))
