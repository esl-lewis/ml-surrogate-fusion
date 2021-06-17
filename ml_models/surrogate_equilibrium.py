import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import wandb

# from matplotlib import cm

# TODO sort big batching
# TODO implement early stopping
# TODO sort GPU use for scaling
# TODO how to optimise hyperparameters? DARTS?
# TODO setup wandb.ai

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from wandb.keras import WandbCallback
import random


# Launch 5 experiments, trying different dropout rates
for run in range(5):
    # Start a run, tracking hyperparameters
    wandb.init(
        project="surrogate-equilibrium",
        entity="EL",
        # Set entity to specify your username or team name
        # ex: entity="carey",
        config={
            "layer_1": 12,
            "activation_1": tf.nn.leaky_relu,
            "dropout": random.uniform(0.01, 0.80),
            "layer_2": 10,
            "activation_2": tf.nn.leaky_relu,
            "layer_3":,7
            "optimizer": "adam",
            "loss": "mse",
            "metric": "mae",
            "epoch": 6,
            "batch_size": 32,
        },
    )
    config = wandb.config


def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization()

    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def df_to_dataset(dataframe_X, dataframe_y, shuffle=True, batch_size=32):
    dataframe = dataframe_X.copy()
    labels = dataframe_y
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


pulse_data = pd.read_csv("../JET_EFIT_magnetic/sampled_data.csv")
pulse_data = pulse_data.dropna(axis=0)

y = pulse_data["FAXS"]
X = pulse_data.drop(["FAXS", "FBND", "Time"], axis=1)

# print(X.head(3))

# Split into train/test with sklearn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=24
)


print(len(X_train), "train examples")
print(len(X_val), "validation examples")
print(len(X_test), "test examples")


# Use tensorflow dataset object for batching
# train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train))
# train_dataset = train_dataset.shuffle(len(X_train)).batch(3)
# train_dataset = train_dataset.prefetch(3)
batch_size = config.batch_size

train_dataset = df_to_dataset(X_train, y_train, batch_size=batch_size)
val_dataset = df_to_dataset(X_val, y_val, shuffle=False, batch_size=batch_size)
test_dataset = df_to_dataset(X_test, y_test, shuffle=False, batch_size=batch_size)


[(train_features, label_batch)] = train_dataset.take(1)
# print("Every feature:", list(train_features.keys()))
# print("A batch of mag probe 13 values:", train_features["BPME_13"])
# print("A batch of target FAXS values:", label_batch)

# Normalise
all_inputs = []
encoded_features = []

# Numeric features.
for header in list(train_features.keys()):
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_dataset)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

all_features = tf.keras.layers.concatenate(encoded_features)
# print(all_features)
x = tf.keras.layers.Dense(config.layer_1, activation=config.activation_1)(all_features)
x = tf.keras.layers.Dense(config.layer_2, activation=tf.nn.activation_2)(x)
x = tf.keras.layers.Dropout(config.dropout)(x)
x = tf.keras.layers.Dense(config.layer_3)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(all_inputs, output)

model.compile(
    optimizer=config.optimizer, loss=config.loss, metrics=[config.metric],
)

history = model.fit(
    train_dataset,
    epochs=config.epoch,
    validation_data=val_dataset,
    batch_size=config.batch_size,
    callbacks=[WandbCallback()],
)

wandb.finish()


"""
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", color="red", label="Validation loss")
plt.title("Training and validation loss")
plt.ylabel("Mean squared error loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()
"""

# loss, accuracy = model.evaluate(test_dataset)
# print("Accuracy", accuracy)

"""
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(10, activation="relu"), tf.keras.layers.Dense(1)]
)

model.compile(
    optimizer="adam", loss="mse", metrics=["mae"],
)

# model.fit(train_dataset, epochs=2)
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    batch_size=len(X_train),
    epochs=5,
)
"""
