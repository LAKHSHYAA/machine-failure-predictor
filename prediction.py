import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib

# Load dataset
df = pd.read_csv("predictive_maintenance.csv")

# Convert 'Failure Type' into binary 'Machine failure'
df['Machine failure'] = df['Failure Type'].apply(lambda x: 0 if x == 'No Failure' else 1)

# Features and labels
X = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = df['Machine failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# Build deep learning model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),  # 🔁 Added
    Dense(32, activation='relu'),
    Dropout(0.5),  # 🔁 Added
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train_scaled, y_train, epochs=25, batch_size=32, validation_split=0.2)

# Save model
model.save("machine_failure_model.h5")

# Evaluate model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Optional: print class distribution
print(df['Machine failure'].value_counts())
