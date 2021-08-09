# for tweaking model without W and B logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# from matplotlib import cm

# TODO sort big batching
# TODO implement early stopping
# TODO sort GPU use for scaling
# TODO how to optimise hyperparameters? DARTS?
# TODO get rid of normalising, is fucking us up

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LeakyReLU  # this is already covered by import
import random

from tensorflow.python.ops.gen_sparse_ops import add_many_sparse_to_tensors_map

leaky_relu = LeakyReLU(alpha=0.01)
"""
config = {
    "layer_1": 12,
    "activation_1": leaky_relu,
    "dropout": random.uniform(0.01, 0.80),
    "layer_2": 10,
    "activation_2": "relu",
    "layer_3": 7,
    "optimizer": "adam",
    "loss": "mae",
    "metric": "mae",
    "epoch": 6,
    "batch_size": 32,
}
"""


class CONFIG:
    def __init__(self):
        CONFIG.layer_1 = 12
        CONFIG.activation_1 = leaky_relu
        CONFIG.dropout = random.uniform(0.01, 0.80)
        CONFIG.layer_2 = 10
        CONFIG.activation_2 = "relu"
        CONFIG.layer_3 = 7
        CONFIG.optimizer = "adam"
        CONFIG.loss = "mae"
        CONFIG.metric = "accuracy"
        CONFIG.epoch = 6
        CONFIG.batch_size = 32


config = CONFIG()


def get_callbacks(name):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]


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


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error [MAE]")
    plt.legend()
    plt.grid(True)
    plt.show()


pulse_data = pd.read_csv("../JET_EFIT_magnetic/all_data.csv")
pulse_data = pulse_data.dropna(axis=0)

y = pulse_data["FBND"] - pulse_data["FAXS"]
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


N_VALIDATION = int(len(X_val))
N_TRAIN = int(len(X_train))
BUFFER_SIZE = N_TRAIN
BATCH_SIZE = 50  # can crank this up
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001, decay_steps=STEPS_PER_EPOCH * 1000, decay_rate=1, staircase=False
)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


def compile_and_fit(model, name, optimizer=None, max_epochs=100):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(
        optimizer=optimizer, loss="mean_absolute_error", metrics="mean_absolute_error"
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks(name),
        verbose=0,
    )
    return history


normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))


tiny_model = tf.keras.Sequential(
    [normalizer, layers.Dense(16, activation="elu"), layers.Dense(1)]
)
size_histories = {}
size_histories["Tiny"] = compile_and_fit(tiny_model, "sizes/Tiny")


large_l2_dropout_model = tf.keras.Sequential(
    [
        normalizer,
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation="elu"),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation="elu"),
        layers.Dense(512, activation="elu", kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation="elu"),
        layers.Dropout(0.01),
        layers.Dense(1),
    ]
)


size_histories["l2_reg"] = compile_and_fit(large_l2_dropout_model, "sizes/large_l2")

plotter.plot(size_histories)
plt.ylim([0, 1.2])
plt.xlim([1, 60])
plt.legend(loc="upper right")


plotter.plot(size_histories)
a = plt.xscale("log")
# plt.xlim([1, max(plt.xlim())])
plt.xlim([1, 100])
plt.legend(loc="upper left")
plt.ylim([0.1, 1.2])
plt.xlabel("Epochs [Log Scale]")

y_pred = large_l2_dropout_model.predict(X_test).flatten()

a = plt.axes(aspect="equal")
plt.scatter(y_test, y_pred)
plt.xlabel("True Values [FAXS]")
plt.ylabel("Predictions [FAXS]")
lims = [0, 1.5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


"""
num_epochs = 100
# Not worried about memory or local minima
batchSize = len(X_train)

# Train on data
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batchSize,
    epochs=num_epochs,
)


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
print("ALL FEATURES")
print(all_features)

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
)


# loss, mean_ab_error = model.evaluate(test_dataset)
# print("MAE on test set", mean_ab_error)
plot_loss(history)
# test_results = {} might be good to see how different feature inputs perform, eg just flux, flux + magnetic, magnetic+ pulse... https://www.tensorflow.org/tutorials/keras/regression

MAE = model.evaluate(test_dataset, verbose=0)
print("MAE on test set", MAE)

test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect="equal")
plt.scatter(test_predictions, y_test)  # get actual true test values in here somehow
plt.xlabel("True Values [FAXS]")
plt.ylabel("Predictions [FAXS]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()
"""
"""
#wandb takes care of this
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
