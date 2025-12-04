import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from IPython.display import display

df = pd.read_excel("OTP_Time_Series_Master.xlsx")

df.head()

df.tail()

df.describe()

print(df.shape)

target_col = "OnTime Arrivals \n(%)"
feature_cols = [
    "Route",
    "Departing Port",
    "Arriving Port",
    "Airline",
    "Month",
    "Sectors Scheduled",
    "Sectors Flown",
    "Cancellations",
    "Departures On Time",
    "Arrivals On Time",
    "Departures Delayed",
    "Arrivals Delayed",
    "OnTime Departures \n(%)",
    "Cancellations \n\n(%)",
]

X = df[feature_cols].copy()
y = df[target_col].copy()

X["Month"] = pd.to_datetime(X["Month"])
X["month_num"] = X["Month"].dt.month
X["year"] = X["Month"].dt.year
X = X.drop(columns="Month")

categorical_cols = ["Route", "Departing Port", "Arriving Port", "Airline"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")  # "na" -> NaN

mask = X[numeric_cols].notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ],
    remainder="drop",
)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", rf_model),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest MAE: {mae:.2f}")
print(f"Random Forest R^2: {r2:.3f}")

print("\nContoh 5 nilai aktual vs prediksi:")
comparison = pd.DataFrame(
    {
        "Actual": y_test[:5].values,
        "Predicted": np.round(y_pred[:5], 2),
    }
)
display(comparison)