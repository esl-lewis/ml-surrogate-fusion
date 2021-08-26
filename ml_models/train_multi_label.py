# Tensorflows make_csv_dataset() does not support multi-label output yet
# need to load csv data to a pandas dataframe instead

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


import wandb
import sys

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from wandb.keras import WandbCallback
from keras.models import load_model

wandb.login()

# default hyperparameter values
defaults = dict(
    dropout=0.6973,
    hidden_layer_size=128,
    l1_size=12,
    l2_size=10,
    initial_learn_rate=0.001,
    decay_rate=1,
    leaky_alpha=0.3,
    epochs=30,
    batch_size=10,
    buffer_size=1000,
)

resume = sys.argv[-1] == "--resume"
wandb.init(config=defaults, resume=resume)
config = wandb.config


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]


def df_to_dataset(dataframe_X, dataframe_y, shuffle=True, batch_size=32):
    dataframe = dataframe_X.copy()
    labels = dataframe_y
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


predict_flux = True  # psi = FAXS and FBND, magnetic flux
predict_R_X = False  # ZXPM and RXPM, R and Z coords of major X point

pulse_data = pd.read_csv("../JET_EFIT_magnetic/all.csv")
pulse_data = pulse_data.dropna(axis=0)

if predict_flux:
    y = pulse_data[["FBND", "FAXS"]]
elif predict_R_X:
    y = pulse_data[["ZXPM", "RXPM"]]

# X = pulse_data.drop(["FBND-FAXS", "FAXS", "FBND", "Time", "ZXPM", "RXPM"], axis=1)
X = pulse_data.drop(["FBND-FAXS", "FAXS", "FBND", "Time"], axis=1)

print(X.shape, y.shape)
# pulse_data.drop(["DFDP","P"]) as well later to test performance

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=24
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=24
)

print(len(X_train), "train samples")
print(len(X_val), "validation samples")
print(len(X_test), "test samples")


FEATURES = 40
BATCH_SIZE = config.batch_size  # can crank this up
BUFFER_SIZE = config.buffer_size
STEPS_PER_EPOCH = BUFFER_SIZE // BATCH_SIZE

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    config.initial_learn_rate,
    (STEPS_PER_EPOCH * 1000),
    config.decay_rate,
    staircase=False,
)

# don't normalise twice, using batchnormalisation later
# normalizer = preprocessing.Normalization(axis=-1)
# normalizer.adapt(np.array(X_train))


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


if wandb.run.resumed:
    print("RESUMING")
    # restore the best model
    model = load_model(wandb.restore("model-best.h5").name)
else:
    tiny_model = tf.keras.Sequential(
        [
            layers.Dense(config.l1_size, input_shape=(FEATURES,)),
            layers.LeakyReLU(alpha=config.leaky_alpha),
            layers.BatchNormalization(),
            layers.Dropout(config.dropout),
            layers.Dense(config.hidden_layer_size, activation="relu"),
            layers.Dense(config.l2_size),
            layers.Dense(2),
        ]
    )

    optimizer = get_optimizer()

    tiny_model.compile(
        optimizer=optimizer,
        loss="mean_absolute_error",
        metrics="mean_absolute_percentage_error",
    )
    print(tiny_model.summary())

print("MODEL COMPILED")

tiny_model.fit(
    X_train,
    y_train,
    steps_per_epoch=STEPS_PER_EPOCH,
    batch_size=config.batch_size,
    epochs=config.epochs,
    initial_epoch=wandb.run.step,  # for resumed runs
    validation_data=(X_val, y_val),
    callbacks=[WandbCallback(), get_callbacks()],
    verbose=0,
)
print("MODEL TRAINED")


# loss, mean_ab_error = model.evaluate(test_dataset)
# print("MAE on test set", mean_ab_error)


wandb.finish()

# y_pred = tiny_model.predict(X_test).flatten()

loss, mean_ab_error = tiny_model.evaluate(X_test, y_test, verbose=2)
print("MAE on test set", mean_ab_error)

y_pred = tiny_model.predict(X_test)

a = plt.axes(aspect="equal")
plt.scatter(y_test.iloc[:, 0], y_pred[:, 0], color="red")
plt.scatter(y_test.iloc[:, 1], y_pred[:, 1], color="blue")
plt.xlabel("True Values [FAXS]")
plt.ylabel("Predictions [FAXS]")
lims = [0, 1.5]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()
"""
# trying a python generator, may lead to performance issues... (limit is buffer size)


FEATURES = 37  # could also FEATURES = len(select_columns) - 2
BATCH_SIZE = config.batch_size  # can crank this up
BUFFER_SIZE = config.buffer_size
STEPS_PER_EPOCH = BUFFER_SIZE // BATCH_SIZE


# assume data already split into train/val/test
train_ds = train_ds.map(pack_features_vector)
val_ds = val_ds.map(pack_features_vector)


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    config.initial_learn_rate,
    (STEPS_PER_EPOCH * 1000),
    config.decay_rate,
    staircase=False,
)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


# some research indicates using both batch normalisation and dropout/regularisation at the same time is counterproductive
# some argument here: https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras about whether BN should be before or after the activation fnc

if wandb.run.resumed:
    print("RESUMING")
    # restore the best model
    model = load_model(wandb.restore("model-best.h5").name)
else:
    tiny_model = tf.keras.Sequential(
        [
            layers.Dense(config.l1_size, input_shape=(FEATURES,)),
            layers.LeakyReLU(alpha=config.leaky_alpha),
            layers.BatchNormalization(),
            layers.Dropout(config.dropout),
            layers.Dense(config.hidden_layer_size, activation="relu"),
            layers.Dense(config.l2_size),
            layers.LeakyReLU(alpha=config.leaky_alpha),
            layers.Dense(1),
        ]
    )

    optimizer = get_optimizer()

    tiny_model.compile(
        optimizer=optimizer,
        loss="mean_absolute_error",
        metrics="mean_absolute_percentage_error",
    )
    print(tiny_model.summary())

print("MODEL COMPILED")

tiny_model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    batch_size=config.batch_size,
    epochs=config.epochs,
    initial_epoch=wandb.run.step,  # for resumed runs
    validation_data=val_ds,
    callbacks=[WandbCallback(), get_callbacks()],
    verbose=0,
)
print("MODEL TRAINED")


# loss, mean_ab_error = model.evaluate(test_dataset)
# print("MAE on test set", mean_ab_error)


wandb.finish()

# remember can seed sweeps from earlier runs: https://docs.wandb.ai/guides/sweeps/existing-project
"""
