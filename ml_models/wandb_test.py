import wandb

wandb.login()

import math

# Simple Keras Model

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from wandb.keras import WandbCallback
import random

# Launch 20 experiments, trying different dropout rates
for run in range(20):
    # Start a run, tracking hyperparameters
    wandb.init(
        project="keras-intro",
        # Set entity to specify your username or team name
        # ex: entity="carey",
        config={
            "layer_1": 512,
            "activation_1": "relu",
            "dropout": random.uniform(0.01, 0.80),
            "layer_2": 10,
            "activation_2": "softmax",
            "optimizer": "sgd",
            "loss": "sparse_categorical_crossentropy",
            "metric": "accuracy",
            "epoch": 6,
            "batch_size": 32,
        },
    )
    config = wandb.config

    # Get the data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, y_train = x_train[::5], y_train[::5]  # Subset data for a faster demo

    # Build a model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(config.layer_1, activation=config.activation_1),
            tf.keras.layers.Dropout(config.dropout),
            tf.keras.layers.Dense(config.layer_2, activation=config.activation_2),
        ]
    )

    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=[config.metric])

    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=config.epoch,
        batch_size=config.batch_size,
        validation_data=(x_test, y_test),
        # WandbCallback auto-saves all metrics from model.fit()
        callbacks=[WandbCallback()],
    )

    # Mark the run as finished
    wandb.finish()

