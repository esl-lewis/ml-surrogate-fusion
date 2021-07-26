import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
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
    epochs=30,
    batch_size=10,
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


column_names = [
    "BPME_0",
    "BPME_10",
    "BPME_11",
    "BPME_12",
    "BPME_13",
    "BPME_14",
    "BPME_15",
    "BPME_16",
    "BPME_17",
    "BPME_18",
    "BPME_2",
    "BPME_3",
    "BPME_4",
    "BPME_5",
    "BPME_6",
    "BPME_7",
    "BPME_8",
    "BPME_9",
    "BVAC",
    "FLME_10",
    "FLME_11",
    "FLME_12",
    "FLME_13",
    "FLME_14",
    "FLME_15",
    "FLME_16",
    "FLME_17",
    "FLME_18",
    "FLME_19",
    "FLME_20",
    "FLME_37",
    "FLME_38",
    "FLME_7",
    "FLME_8",
    "FLME_9",
    "FLX",
    "IPLA",
    "Time",
    "DFDP",
    "FAXS",
    "FBND",
    "P",
    "FBND-FAXS",
]

select_columns = [
    "BPME_0",
    "BPME_10",
    "BPME_11",
    "BPME_12",
    "BPME_13",
    "BPME_14",
    "BPME_15",
    "BPME_16",
    "BPME_17",
    "BPME_18",
    "BPME_2",
    "BPME_3",
    "BPME_4",
    "BPME_5",
    "BPME_6",
    "BPME_7",
    "BPME_8",
    "BPME_9",
    "BVAC",
    "FLME_10",
    "FLME_11",
    "FLME_12",
    "FLME_13",
    "FLME_14",
    "FLME_15",
    "FLME_16",
    "FLME_17",
    "FLME_18",
    "FLME_19",
    "FLME_20",
    "FLME_37",
    "FLME_38",
    "FLME_7",
    "FLME_8",
    "FLME_9",
    "FLX",
    "IPLA",
    "DFDP",
    "P",
    "FBND-FAXS",
]
# removes time, FAXS and FBND
# column names here is label inclusive, ie not just feature names

FEATURES = 39
BATCH_SIZE = 10  # can crank this up
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = BUFFER_SIZE // BATCH_SIZE


train_ds = tf.data.experimental.make_csv_dataset(
    file_pattern="../JET_EFIT_magnetic/train/*_merged.csv",
    batch_size=config.batch_size,
    num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=1000,
    label_name="FBND-FAXS",
    header=True,
    shuffle=True,
    select_columns=select_columns,
)

val_ds = tf.data.experimental.make_csv_dataset(
    file_pattern="../JET_EFIT_magnetic/val/*_merged.csv",
    batch_size=config.batch_size,
    num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=1000,
    label_name="FBND-FAXS",
    header=True,
    shuffle=False,
    select_columns=select_columns,
)

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


# """
# some research indicates using both batch normalisation and dropout/regularisation at the same time is counterproductive

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
        optimizer=optimizer, loss="mean_absolute_error", metrics="mean_absolute_error"
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
