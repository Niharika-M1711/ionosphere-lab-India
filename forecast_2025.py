import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

TEC_MAX = 200.0
SEQ_LEN = 28
LT_epochs = {"06LT": 2, "12LT": 26, "18LT": 50, "22LT": 66}
indices = list(LT_epochs.values())

d = np.load(r"C:\Users\hi\Documents\TEC_2025_real\india_tec_2025_raw.npz")
tec = d['tec']
lt_data = tec[:, indices, :, :]
flat_2025 = lt_data.reshape(-1, 40, 40)
flat_norm = flat_2025 / TEC_MAX

conv_model = tf.keras.models.load_model(r"C:\Users\hi\Documents\TEC_2024 Data\convlstm_model.keras")
lstm_model = tf.keras.models.load_model(r"C:\Users\hi\Documents\TEC_2024 Data\lstm_model.keras")

conv_preds, lstm_preds, actuals, doy_list = [], [], [], []

for i in range(0, len(flat_norm) - SEQ_LEN - 4 + 1, 4):
    seq = flat_norm[i:i+SEQ_LEN]
    actual = flat_norm[i+SEQ_LEN:i+SEQ_LEN+4]
    actual = np.moveaxis(actual, 0, -1)

    seq_conv = seq[np.newaxis, ..., np.newaxis]
    conv_p = conv_model.predict(seq_conv, verbose=0)[0]

    seq_lstm = seq.reshape(1, SEQ_LEN, 1600)
    lstm_p = lstm_model.predict(seq_lstm, verbose=0)[0].reshape(4, 40, 40)
    lstm_p = np.moveaxis(lstm_p, 0, -1)

    conv_preds.append(conv_p)
    lstm_preds.append(lstm_p)
    actuals.append(actual)
    doy_list.append((i // 4) + 8)

conv_preds = np.array(conv_preds) * TEC_MAX
lstm_preds = np.array(lstm_preds) * TEC_MAX
actuals = np.array(actuals) * TEC_MAX
doy_list = np.array(doy_list)

for name, preds in [("ConvLSTM", conv_preds), ("LSTM", lstm_preds)]:
    rmse = np.sqrt(np.mean((preds - actuals)**2))
    mae = np.mean(np.abs(preds - actuals))
    r2 = r2_score(actuals.flatten(), preds.flatten())
    print(f"{name} 2025 Forecast — RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.4f}")

np.savez_compressed(r"C:\Users\hi\Documents\TEC_2024 Data\forecast_2025_final.npz",
                     conv_preds=conv_preds, lstm_preds=lstm_preds, actuals=actuals, doys=doy_list)
print("Saved forecast_2025_final.npz")