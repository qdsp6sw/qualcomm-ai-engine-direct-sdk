#!/usr/bin/python3
#===============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#===============================================================================
#  MIT License
#
#  Copyright (c) 2018 Mohsin Baig
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#==============================================================================

import argparse
import librosa
import numpy as np
import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def split_train_test(src_path, train_path, test_path):
    os.makedirs(train_path, exist_ok =True)
    os.makedirs(test_path, exist_ok =True)
    for filename in os.listdir(src_path):
        first_split = filename.rsplit('_', 1)[1]
        second_split = first_split.rsplit('.', 1)[0]
        if int(second_split) <= 4:
            shutil.copyfile(src_path + '/' + filename, test_path + '/' + filename)
        else:
            shutil.copyfile(src_path + '/' + filename, train_path + '/' + filename)

def extract_mfcc(file_path, utterance_length):
    # load raw .wav data with librosa
    raw_w, sampling_rate = librosa.load(file_path, mono=True)
    # get mfcc features
    mfcc_features = librosa.feature.mfcc(y=raw_w, sr=sampling_rate, n_mfcc=10)
    if mfcc_features.shape[1] > utterance_length:
        mfcc_features = mfcc_features[:, 0:utterance_length]
    else:
        mfcc_features = np.pad(mfcc_features,
                               ((0, 0),
                               (0, utterance_length - mfcc_features.shape[1])),
                               mode='constant',
                               constant_values=0)
    return mfcc_features

def get_mfcc_batch(file_path, batch_size, utterance_length):
    files = os.listdir(file_path)
    feature_batch = []
    label_batch = []
    while True:
        # shuffle files
        random.shuffle(files)
        for file_name in files:
            # make sure raw files are in .wav format
            if not file_name.endswith('.wav'):
                continue
            # get mfcc features from file_path
            mfcc_features = extract_mfcc(file_path + file_name, utterance_length)
            # one-hot encoded label from 0-9
            label = np.eye(10, dtype=np.float32)[int(file_name[0])]
            # label batch
            label_batch.append(label)
            # feature batch
            feature_batch.append(mfcc_features)
            if len(feature_batch) >= batch_size:
                # yield feature and label batches
                yield feature_batch, label_batch
                # reset batches
                feature_batch = []
                label_batch = []

def create_model(learning_rate, training_epochs, training_batch):
    # create neural network with four fully connected layers and adam optimizer
    sp_network = Sequential()
    sp_network.add(Input(shape=(10, 35)))
    sp_network.add(Dense(256, activation='relu'))
    sp_network.add(Dense(128, activation='relu'))
    sp_network.add(Dense(64, activation='relu'))
    sp_network.add(Flatten())
    sp_network.add(Dense(10, activation='softmax'))
    sp_network.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    return sp_network

def serialize_model():
    model = tf.keras.models.load_model('model/spoken_digit.h5')

    # Get the concrete function from the Keras model
    full_model = tf.function(lambda x: model(x))
    concrete_function = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Convert the model to a frozen graph
    frozen_func = convert_variables_to_constants_v2(concrete_function)
    frozen_func.graph.as_graph_def()

    # Save the frozen graph to a file
    output_graph = 'model/spoken_digit.pb'
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=".",
                      name=output_graph,
                      as_text=False)

def spoken_digit(learning_rate, training_epochs, training_batch):
    # split training and testing data
    split_train_test('free-spoken-digit-dataset/recordings/',
                     'train/',
                     'test/')
    print('Successfully split free-spoken-digit-dataset training/testing data.')
    # get training data
    train_batch = get_mfcc_batch('train/', training_batch*4, utterance_length=35)
    print('Training data created.')
    # create model
    sp_model = create_model(learning_rate, training_epochs, training_batch)
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    # train model
    X_train, y_train = next(train_batch)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test, y_test = next(train_batch)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    sp_model.fit(X_train,
                 y_train,
                 epochs=training_epochs,
                 validation_data=(X_test, y_test),
                 batch_size=training_batch,
                 callbacks=[tensorboard_callback])
    print('Optimization done.')
    # delete training ops
    backend.clear_session()
    # save model in keras format
    sp_model.save('model/spoken_digit.h5')
    # save model in protobuf format
    serialize_model()
    print('Save frozen graph in spoken_digit.pb.')

def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Create and Train Spoken Digit Neural Network model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr', '--learning_rate',
                        help='Learning rate.',
                        default=0.001)
    parser.add_argument('-epochs', '--training_epochs',
                        help='Training epochs.',
                        default=20)
    parser.add_argument('-batch', '--training_batch',
                        help='Training batch size.',
                        default=128)
    args = parser.parse_args()
    spoken_digit(float(args.learning_rate), int(args.training_epochs), int(args.training_batch))

if __name__ == '__main__':
    main()
