from generate_data import df
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# === Data prep ===
df_as_np = df.to_numpy()
dates = df.index.get_level_values(1).unique()
# reshape 2d -> 3d (num_departments, num_months, num_features)
history_matrix = df_as_np[:, 2:-2].reshape(
    (len(df.index.get_level_values(0).unique()), len(dates), -1)
)
# target variables
y = df_as_np[:, :2].reshape((len(df.index.get_level_values(0).unique()), len(dates), 2))

# dataset split
train_size = int(len(dates) * 0.8)
validation_size = int(len(dates) * 0.9)
dates_train, dates_validation, dates_test = (
    dates[:train_size],
    dates[train_size:validation_size],
    dates[validation_size:],
)
X_train, X_validation, X_test = (
    history_matrix[:, :train_size, :],
    history_matrix[:, train_size:validation_size, :],
    history_matrix[:, train_size:, :],
)
y_train, y_validation, y_test = (
    y[:, :train_size, :],
    y[:, train_size:validation_size, :],
    y[:, train_size:, :],
)

# === Model definition ===
model = Sequential(
    [
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32, return_sequences=True),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(2),
    ]
)

model.compile(optimizer=Adam(learning_rate=0.005), loss="mse", metrics=["mae"])

# === Training ===
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=100)


# === Prediction ===
model.predict(X_test)
