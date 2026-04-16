import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from dataset import create_sequences
from advanced_models import LSTMModel, GRUModel

DATA_PATH = "data/processed/sg_11_2"

# Experiment Settings
SEQ_LENGTHS = [10, 30, 60]
MODEL_TYPES = ["LSTM", "GRU"]
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 15
LEARNING_RATE = 0.001

results = []

# Filter for CSV files once
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

for seq_len in SEQ_LENGTHS:
    for model_type in MODEL_TYPES:
        print(f"Running: Model={model_type}, seq={seq_len}")

        all_train_mse = []
        all_test_mse = []

        for file in csv_files:
            df = pd.read_csv(os.path.join(DATA_PATH, file))
            data = df[['Smooth_Close', 'Return', 'MA_10', 'EMA_10']].values

            X, y = create_sequences(data, seq_len)

            if len(X) < 50:
                continue

            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Convert to tensors while preserving shape (Batch, Sequence, Features)
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            num_features = X_train.shape[2]

            # Initialize model
            if model_type == "LSTM":
                model = LSTMModel(input_size=num_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
            else:
                model = GRUModel(input_size=num_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            loss_fn = nn.MSELoss()

            # Training loop
            for epoch in range(EPOCHS):
                model.train()
                pred = model(X_train)
                loss = loss_fn(pred, y_train)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train).numpy()
                test_pred = model(X_test).numpy()

            train_mse = mean_squared_error(y_train.numpy(), train_pred)
            test_mse = mean_squared_error(y_test.numpy(), test_pred)

            all_train_mse.append(train_mse)
            all_test_mse.append(test_mse)

        avg_train = np.mean(all_train_mse)
        avg_test = np.mean(all_test_mse)

        print(f"Train MSE: {avg_train:.6f} | Test MSE: {avg_test:.6f}")

        results.append({
            "model": model_type,
            "seq_len": seq_len,
            "train_mse": avg_train,
            "test_mse": avg_test
        })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("advanced_baseline_results.csv", index=False)
print("Advanced baseline experiments completed.")