import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import wandb
import sys
import time

# from matplotlib import cm

# TODO sort big batching
# TODO sort GPU use for scaling
# TODO how to optimise hyperparameters? DARTS?

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
    epochs=27,
)

resume = sys.argv[-1] == "--resume"
wandb.init(config=defaults, resume=resume)
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


def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]


# leaky_relu = LeakyReLU(alpha=0.01)

pulses_ds = tf.data.experimental.make_csv_dataset(
    file_pattern="../JET_EFIT_MAGNETIC/*_merged.csv",
    batch_size=10,
    num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000,
    label_name="FBND-FAXS",
    header=True,
    shuffle=True,
)


# load data
# pulse_data = pd.read_csv("../JET_EFIT_magnetic/all_data.csv")
pulse_data = pd.read_csv("../JET_EFIT_magnetic/interpolated_99070.csv")
pulse_data = pulse_data.dropna(axis=0)

y = pulse_data["FBND"] - pulse_data["FAXS"]
X = pulse_data.drop(["FAXS", "FBND", "Time"], axis=1)

# Split into train/test with sklearn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=24
)

N_VALIDATION = int(len(X_val))
N_TRAIN = int(len(X_train))
BUFFER_SIZE = N_TRAIN
BATCH_SIZE = 50  # can crank this up
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    config.initial_learn_rate,
    (STEPS_PER_EPOCH * 1000),
    config.decay_rate,
    staircase=False,
)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


if wandb.run.resumed:
    print("RESUMING")
    # restore the best model
    model = load_model(wandb.restore("model-best.h5").name)
else:
    normalizer = preprocessing.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train))

    tiny_model = tf.keras.Sequential(
        [
            normalizer,
            layers.Dense(config.l1_size),
            layers.LeakyReLU(alpha=config.leaky_alpha),
            layers.Dropout(config.dropout),
            layers.Dense(config.hidden_layer_size, activation="relu"),
            layers.Dense(config.l2_size),
            layers.LeakyReLU(alpha=config.leaky_alpha),
            layers.Dense(1),
        ]
    )

    optimizer = get_optimizer()

    tiny_model.compile(
        optimizer=optimizer, loss="mean_absolute_error", metrics="mean_absolute_error"
    )

print("MODEL COMPILED")

tiny_model.fit(
    X_train,
    y_train,
    # steps_per_epoch=STEPS_PER_EPOCH,
    # batch_size=config.batch_size,
    epochs=config.epochs,
    initial_epoch=wandb.run.step,  # for resumed runs
    validation_data=(X_val, y_val),
    callbacks=[WandbCallback(), get_callbacks()],
    verbose=0,
)
print("MODEL TRAINED")


"""
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

x = tf.keras.layers.Dense(config.layer_1, activation=config.activation_1)(
    all_features
)  # normalise layer
x = tf.keras.layers.Dense(config.layer_2, activation=config.activation_2)(x)
x = tf.keras.layers.Dropout(config.dropout)(x)
x = tf.keras.layers.Dense(config.layer_3)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(all_inputs, output)
print("MODEL SUMMARY")
print(model.summary())

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

"""
# loss, mean_ab_error = model.evaluate(test_dataset)
# print("MAE on test set", mean_ab_error)


wandb.finish()

# remember can seed sweeps from earlier runs: https://docs.wandb.ai/guides/sweeps/existing-project
