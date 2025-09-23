import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('spotify_churn_dataset.csv')
# df = df.rename(columns={
#     "user_id": "id_pengguna",
#     "gender": "jenis_kelamin",
#     "age": "usia",
#     "country": "negara",
#     "subscription_type": "jenis_langganan",
#     "listening_time": "waktu_dengar_per_hari(menit)",
#     "songs_played_per_day": "lagu_diputar_per_hari",
#     "skip_rate": "persentase_skip",
#     "device_type": "jenis_perangkat",
#     "ads_listened_per_week": "iklan_didengar_per_minggu",
#     "offline_listening": "waktu_dengar_offline(menit)",
#     "is_churned": "status_churn"
# })
print("Dataset loaded and renamed. Shape:", df.shape)
print(df.head())  # Inspect first few rows


# Preprocessing
# Handle missing values
df = df.dropna()

# One-hot encode categorical columns
categorical_cols = ['jenis_kelamin', 'negara', 'jenis_langganan', 'jenis_perangkat']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # Avoid multicollinearity

# Select features and target
feature_cols = [col for col in df.columns if col not in ['id_pengguna', 'status_churn']]
X = df[feature_cols].values.astype(np.float64)  # Convert to NumPy array
y = df['status_churn'].values.astype(np.float64)

# Standardize features (mean=0, std=1)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std[std == 0] = 1  # Avoid division by zero
X = (X - mean) / std

# Augment X with bias column (column of 1s)
X = np.c_[np.ones(X.shape[0]), X]

# Initialize weights randomly
np.random.seed(42)  # For reproducibility
W = np.random.randn(X.shape[1]) * 0.01  # Small random values

# Manual train/test split (80/20)
n = X.shape[0]
train_size = int(0.8 * n)
indices = np.random.permutation(n)
train_idx = indices[:train_size]
test_idx = indices[train_size:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Preprocessed data. Train shape:", X_train.shape, "Test shape:", X_test.shape)