import joblib
import pandas as pd

model = joblib.load('volatility_model.pkl')
features = joblib.load('features.pkl')

# predicting spikes

# loading input
def predict_spike(data: dict) -> tuple[bool, float]:
    data["Adj Close"] = data.pop("adj_close")
    data["High"]      = data.pop("high")
    data["Low"]       = data.pop("low")
    data["Open"]      = data.pop("open")
    data["Volume"]    = data.pop("volume")
    data["return"]    = data.pop("return_")
    # close, day_of_week, month, day, week_of_year, volatility stay as it is

    # create dataframe and predict
    df   = pd.DataFrame([data])[features] # ensure correct order of features
    prob = model.predict_proba(df)[0, 1] # probability of spike btw 0 and 1
    return prob >= 0.5, round(float(prob), 4)