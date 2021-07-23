import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import wandb
import os

# from matplotlib import cm

# TODO sort big batching
# TODO implement early stopping
# TODO sort GPU use for scaling
# TODO how to optimise hyperparameters? DARTS?
# TODO setup wandb.ai
# TODO put leaky relu and adam back in having made wandb work

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from wandb.keras import WandbCallback
from tensorflow.keras.layers import LeakyReLU  # this is already covered by import
import random
import argparse


wandb.login()

# default hyperparameter values
PROJECT_NAME = "surrogate-equilibrium"
MODEL_NOTES = "default small network"
BATCH_SIZE = 10
DROPOUT = 0.25
EPOCHS = 100
L1_SIZE = 12
L2_SIZE = 28
HIDDEN_LAYER_SIZE = 128
INITIAL_LEARNING_RATE = 0.001
DECAY_RATE = 1
LEAKY_ALPHA = 0.3

defaults = dict(
    dropout=0.6973,
    hidden_layer_size=128,
    l1_size=12,
    l2_size=10,
    initial_learn_rate=0.001,
    decay_rate=1,
    leaky_alpha=0.3,
    epochs=100,
    batch_size=10,
)


"""
# "activation_1": leaky_relu, # note leaky relu needs to be its own layer
"activation_1": "relu",
"dropout": random.uniform(0.01, 0.80),
"activation_2": "relu",
"optimizer": "sgd",
"loss": "mae",
"metric": "mae",
"""


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


def train_nn(defaults):
    # initialize wandb logging
    """
    wandb.init(project=args.project_name, notes=args.notes)
    wandb.config.update(args)
    """
    # leaky_relu = LeakyReLU(alpha=0.01)
    wandb.init(config=defaults)
    config = wandb.config

    # load data
    pulse_data = pd.read_csv("../JET_EFIT_magnetic/all_data.csv")
    # pulse_data = pd.read_csv("../JET_EFIT_magnetic/interpolated_99070.csv")
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
        INITIAL_LEARNING_RATE, (STEPS_PER_EPOCH * 1000), DECAY_RATE, staircase=False
    )

    def get_optimizer():
        return tf.keras.optimizers.Adam(lr_schedule)

    normalizer = preprocessing.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train))

    tiny_model = tf.keras.Sequential(
        [
            normalizer,
            layers.Dense(config.l1_size),
            layers.LeakyReLU(alpha=config.leaky_alpha),
            layers.Dropout(config.dropout),
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
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(X_val, y_val),
        callbacks=[WandbCallback(), get_callbacks()],
        verbose=1,
    )
    print("MODEL TRAINED")

    y_pred = tiny_model.predict(X_test).flatten()

    a = plt.axes(aspect="equal")
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values [FAXS]")
    plt.ylabel("Predictions [FAXS]")
    lims = [0, 2]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--notes",
        type=str,
        default=MODEL_NOTES,
        help="Notes about the training run",
    )
    parser.add_argument(
        "-p", "--project_name", type=str, default=PROJECT_NAME, help="Main project name"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=BATCH_SIZE, help="batch_size"
    )
    parser.add_argument(
        "--dropout", type=float, default=DROPOUT, help="dropout before dense layers"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=EPOCHS,
        help="number of training epochs (passes through full training data)",
    )
    parser.add_argument(
        "--hidden_layer_size",
        type=int,
        default=HIDDEN_LAYER_SIZE,
        help="hidden layer size",
    )
    parser.add_argument(
        "-l1", "--layer_1_size", type=int, default=L1_SIZE, help="layer 1 size"
    )
    parser.add_argument(
        "-l2", "--layer_2_size", type=int, default=L2_SIZE, help="layer 2 size"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=INITIAL_LEARNING_RATE,
        help="learning rate",
    )
    parser.add_argument(
        "--decay", type=float, default=DECAY_RATE, help="learning rate decay"
    )
    parser.add_argument(
        "--leaky_alpha", type=float, default=LEAKY_ALPHA, help="leaky relu alpha value"
    )

    parser.add_argument(
        "-q", "--dry_run", action="store_true", help="Dry run (do not log to wandb)"
    )

    args = parser.parse_args()
    """
    """
    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
        print("in dry run mode?")

    train_nn(args)
    """
    train_nn(defaults)


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

