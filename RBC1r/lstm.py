#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Author  : Qianxiao Li <qianxiao@nus.edu.sg>
'''

# NOTE: V2 directly works on PCA data

import argparse
import numpy as np
import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tqdm.keras import TqdmCallback
from tqdm import tqdm


def load_data(n_pca=3):
    """Data Loader

    :param n_pca: number of pca components, defaults to 3
    :type n_pca: int, optional
    """

    path = f'../dataRBC/RBC_r28L_T100R100_pca{n_pca}_enc_data.txt.gz'
    traj_data = pd.read_csv(path, header=None).values
    traj_data = traj_data.reshape(-1, 200, n_pca)
    traj_data = traj_data[:, ::2, :]
    traj_train, traj_test = train_test_split(traj_data, test_size=0.2)

    x_train, x_test = traj_train[:, :-1, :], traj_test[:, :-1, :]
    y_train, y_test = traj_train[:, 1:, :], traj_test[:, 1:, :]

    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
    }


def build_lstm_model(n_hidden_lstm, n_hidden_init, n_features):
    """Build LSTM

    :param n_hidden_lstm: number of hidden units
    :type n_hidden_lstm: int
    :param n_hidden_init: number of units in the initialization prediction network
    :type n_hidden_init: int
    :param n_features: number of input features
    :type n_features: int
    :return: LSTM model
    :rtype: tf.keras.Model object
    """
    inputs = Input((None, n_features))

    # Rescale
    scale = 10.0
    inputs_scaled = Lambda(lambda x: x / scale)(inputs)

    # LSTM Initial condition (layer 1)
    lstm_init_hidden_1 = Dense(units=n_hidden_init, activation='sigmoid')(inputs_scaled[:, 0, :])
    lstm_init_1 = Dense(units=2*n_hidden_lstm)(lstm_init_hidden_1)

    # LSTM Initial condition (layer 2)
    lstm_init_hidden_2 = Dense(units=n_hidden_init, activation='sigmoid')(inputs_scaled[:, 0, :])
    lstm_init_2 = Dense(units=2*n_hidden_lstm)(lstm_init_hidden_2)

    # LSTM
    lstm_hidden_1 = LSTM(n_hidden_lstm, return_sequences=True)(
        inputs,
        initial_state=[
            lstm_init_1[:, :n_hidden_lstm],
            lstm_init_1[:, n_hidden_lstm:],
        ],
    )
    lstm_hidden_2 = LSTM(n_hidden_lstm, return_sequences=True)(
        lstm_hidden_1,
        initial_state=[
            lstm_init_1[:, :n_hidden_lstm],
            lstm_init_1[:, n_hidden_lstm:],
        ],
    )
    outputs = Dense(n_features)(lstm_hidden_2)

    # Rescale
    outputs = Lambda(lambda x: x * scale)(outputs)

    model = Model(inputs, outputs)

    return model


def train_model(model, data, epochs=1000):
    """Training Routine

    :param model: Model
    :type model: tf.keras.Model object
    :param data: Data
    :type data: tuple
    :param epochs: Number of Epochs, defaults to 1000
    :type epochs: int, optional
    :return: History of training
    :rtype: dict
    """

    # Data
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # Compile
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mae'],
    )

    # Train
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=20,
        epochs=epochs,
        validation_data=(x_test, y_test),
        # callbacks=[TqdmCallback()],
        verbose=0,
    )

    return history


def predict_sequence(model, nT, x_init):
    """Using model to predict sequence

    :param model: model
    :type model: tf.keras.Model object
    :param nT: number of time steps
    :type nT: int
    :param x_init: initial input
    :type x_init: ndarray
    :return: predicted time series
    :rtype: ndarray
    """
    x_pred_seq = x_init[:, None, :]  # reshape to [Batch, 1, Feature]

    # for _ in tqdm(range(nT)):
    for _ in range(nT):
        y_pred_seq = model.predict(x_pred_seq)
        x_pred_seq = np.concatenate([x_pred_seq, y_pred_seq[:, -1:, :]], axis=1)

    return x_pred_seq  # includes the initial condition as first time slice


def validate(model, data):
    """Compute PCA normalized error of model on data

    :param model: model
    :type model: tf.keras.Model object
    :param data: dataset
    :type data: dict
    :return: normalized MSE
    :rtype: float
    """
    x_test = data['x_test']
    x_init = data['x_test'][:, 0, :]
    nT = data['x_test'].shape[1]
    x_pred_seq = predict_sequence(model, nT, x_init)

    # PCA normalized error
    x_last_pred = x_pred_seq[:, -1, :]
    x_last_true = x_test[:, -1, :]
    x_true_l2_squared = np.mean(x_pred_seq**2, axis=1, keepdims=False)
    diff_squared_rel = (x_last_pred - x_last_true)**2 / x_true_l2_squared
    mse_normalized = np.mean(diff_squared_rel)
    print(f'PCA error: {mse_normalized}')

    return mse_normalized


def convert_data(data):
    """Preprocess data by reshaping

    :param data: original data
    :type data: dict
    :return: processed data
    :rtype: dict
    """
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    n_features = x_train.shape[-1]

    x_train = x_train.reshape(-1, 1, n_features)
    y_train = y_train.reshape(-1, 1, n_features)
    x_test = x_test.reshape(-1, 1, n_features)
    y_test = y_test.reshape(-1, 1, n_features)

    return {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
    }


if __name__=="__main__":
    # Here we consider different training/testing settings
    # Long: use the entire sequence as input/outputs of the LSTM. This is the usual setting of LSTM
    # Short: only use 2 time steps. This is not the usual setting of LSTM, but this is required in order
    #        to make one-step predictions in order to have direct comparison with OnsagerNet

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("n_pca", help="number of pca modes", type=int)
    parser.add_argument("n_hidden_lstm", help="number hidden nodes in LSTM", type=int)
    parser.add_argument("n_hidden_init", help="number hidden nodes in intial condition prediction", type=int)
    parser.add_argument("long_train", help="whether to use long trajectory for training", type=int)
    parser.add_argument("long_test", help="whether to use long trajectory for testing", type=int)
    args = parser.parse_args()
    print(f'config: {args}')
    np.random.seed(123)

    # Load data and train model
    data = load_data(n_pca=args.n_pca)
    data_short = convert_data(data)
    model = build_lstm_model(
        n_hidden_lstm=args.n_hidden_lstm,
        n_features=args.n_pca,
        n_hidden_init=args.n_hidden_init,
    )

    # Display results
    if args.long_train:
        print('Training using long trajectories...')
        history = train_model(model, data)
    else:
        print('Training using short trajectories...')
        history = train_model(model, data_short)

    if args.long_test:
        print('Testing using long trajectories...')
        error = validate(model, data)
    else:
        print('Testing using short trajectories...')
        error = validate(model, data_short)

    print(f'Results {args.long_train} {args.long_test} {error}')
