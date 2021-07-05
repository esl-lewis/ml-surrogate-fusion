import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.tree import DecisionTreeRegressor, export_graphviz, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

pulse_data = pd.read_csv("../JET_EFIT_magnetic/sampled_data.csv")

print(pulse_data.head(3))

pulse_data = pulse_data.dropna(axis=0)

# pulse_data = pulse_data.apply(lambda x: round(x, 5))

# print(pulse_data.head(3))

y = pulse_data["FAXS"]
X = pulse_data.drop(["FAXS", "FBND", "Time"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)

scale_transform = StandardScaler(with_mean=False)
scale_transform.fit(X_train)

X_train = scale_transform.transform(X_train)
X_test = scale_transform.transform(X_test)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")


# print(X_train)

clf = DecisionTreeRegressor(
    random_state=0,
    max_depth=5,
    min_samples_leaf=1,
    min_samples_split=2,
    splitter="best",
).fit(X_train, y_train)


y_predicted = clf.predict(X_test)

print("mean squared error", mean_squared_error(y_test, y_predicted))
