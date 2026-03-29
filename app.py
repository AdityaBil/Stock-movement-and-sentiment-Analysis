

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# Load model config
# ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")
# Prefer artifacts from clone_file_v2.py; fall back to legacy names
MODEL_PATH  = os.path.join(BASE_DIR, "stock_model.keras")
MODEL_FALLBACK = os.path.join(BASE_DIR, "transformer_with_sentiment.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
SCALER_FALLBACK = os.path.join(BASE_DIR, "scaler_sentiment.pkl")
FINBERT_DIR = os.path.join(BASE_DIR, "finbert_finetuned")

print("=" * 55)
print("  QUANTUS v4 — Flask Backend Starting")
print("=" * 55)

# ── Config ──────────────────────────────────────────────────
cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    print(f"  ✓ Config loaded: {len(cfg.get('features',[]))} features, "
          f"lookback={cfg.get('lookback',20)}, threshold={cfg.get('best_threshold',0.5)}")
else:
    print("  ⚠ model_config.json not found — run clone_file_v2.py first to train the model")

# Default feature list matches clone_file_v2.py when config is missing
DEFAULT_FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "Return",
    "MA_5", "MA_10", "MA_20", "MA_50",
    "MA_5_10_cross", "MA_10_20_cross", "MA_20_50_cross",
    "RSI_7", "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_width", "BB_pos",
    "Volume_Ratio", "High_Low_Ratio", "Close_Open_Ratio",
    "OBV", "Stoch_K", "Stoch_D", "ATR_ratio",
    "CCI", "Williams_R",
    "Momentum_5", "Momentum_10", "Momentum_20",
    "Volatility_5", "Volatility_20",
    "sentiment",
]

LOOKBACK       = cfg.get("lookback", 20)
BEST_THRESHOLD = cfg.get("best_threshold", 0.5)
FEATURES       = cfg.get("features", []) or DEFAULT_FEATURES
FORWARD_DAYS   = cfg.get("forward_days", 3)
RETURN_THRESHOLD = cfg.get("return_threshold", 0.005)
SCALER_TYPE    = cfg.get("scaler_type", "RobustScaler")

# ── Keras custom objects needed for model loading ──────────
def gelu(x):
    return x * 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))

def focal_loss_fn(y_true, y_pred):
    gamma = 2.0; alpha = 0.5
    y_pred  = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
    pt      = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1-alpha)
    fl      = -alpha_t * tf.pow(1-pt, gamma) * tf.math.log(pt)
    return tf.reduce_mean(fl)

from tensorflow.keras import layers

# Layer definitions match clone_file_v2.py (saved models use MultiHeadSelfAttention)
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kw):
        super().__init__(**kw)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.h  = num_heads
        self.dk = d_model // num_heads
        self.d  = d_model
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.wo = layers.Dense(d_model)

    def call(self, x, training=False):
        B = tf.shape(x)[0]

        def split(t):
            return tf.transpose(tf.reshape(t, (B, -1, self.h, self.dk)), [0, 2, 1, 3])

        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        scale = tf.math.sqrt(tf.cast(self.dk, tf.float32))
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / scale, axis=-1)
        out = tf.reshape(tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3]), (B, -1, self.d))
        return self.wo(out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d, "num_heads": self.h})
        return cfg


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, drop, **kw):
        super().__init__(**kw)
        self.attn  = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn   = tf.keras.Sequential([
            layers.Dense(d_ff),
            layers.Activation(gelu),
            layers.Dropout(drop),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(drop)
        self.drop2 = layers.Dropout(drop)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.drop = drop

    def call(self, x, training=False):
        x = x + self.drop1(self.attn(self.norm1(x), training=training), training=training)
        x = x + self.drop2(self.ffn(self.norm2(x), training=training), training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "d_ff": self.d_ff, "drop": self.drop,
        })
        return cfg

# Older checkpoints may serialize as MultiHeadAttention
MultiHeadAttention = MultiHeadSelfAttention

CUSTOM_OBJECTS = {
    "MultiHeadSelfAttention": MultiHeadSelfAttention,
    "MultiHeadAttention":     MultiHeadAttention,
    "TransformerBlock":       TransformerBlock,
    "gelu":                   gelu,
    "focal_loss_fn":          focal_loss_fn,
}

# ── Load model ─────────────────────────────────────────────
model  = None
scaler = None

_load_path = MODEL_PATH if os.path.exists(MODEL_PATH) else (
    MODEL_FALLBACK if os.path.exists(MODEL_FALLBACK) else None
)
if _load_path:
    try:
        model = tf.keras.models.load_model(_load_path, custom_objects=CUSTOM_OBJECTS)
        print("  ✓ Model loaded:", _load_path)
    except Exception as e:
        print(f"  ⚠ Model load failed: {e}")
else:
    print(f"  ⚠ No model file (tried {MODEL_PATH}, {MODEL_FALLBACK})")

_scaler_path = SCALER_PATH if os.path.exists(SCALER_PATH) else (
    SCALER_FALLBACK if os.path.exists(SCALER_FALLBACK) else None
)
if _scaler_path:
    try:
        scaler = joblib.load(_scaler_path)
        print("  ✓ Scaler loaded:", _scaler_path)
    except Exception as e:
        print(f"  ⚠ Scaler load failed: {e}")
else:
    print(f"  ⚠ No scaler (tried {SCALER_PATH}, {SCALER_FALLBACK})")

# ── FinBERT ────────────────────────────────────────────────
finbert_ok  = False
finbert_tok = None
finbert_mdl = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch, torch.nn.functional as torchF

    src = FINBERT_DIR if os.path.exists(FINBERT_DIR) else "ProsusAI/finbert"
    finbert_tok = AutoTokenizer.from_pretrained(src)
    finbert_mdl = AutoModelForSequenceClassification.from_pretrained(src)
    finbert_ok  = True
    print(f"  ✓ FinBERT loaded from: {src}")
except Exception as e:
    print(f"  ⚠ FinBERT unavailable: {e}")

def get_news_sentiment(texts):
    if not texts or not finbert_ok:
        return 0.0
    try:
        inputs = finbert_tok(list(texts), padding=True, truncation=True,
                             return_tensors="pt", max_length=512)
        with torch.no_grad():
            probs = torchF.softmax(finbert_mdl(**inputs).logits, dim=-1).cpu().numpy()
        return float(np.mean(probs[:, 0] - probs[:, 1]))  # pos - neg
    except:
        return 0.0

# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (mirrors clone_file_v2.py — no Target column)
# ──────────────────────────────────────────────────────────────
def rsi(prices, period=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md + 1e-9)


def engineer_features(df):
    c = "Adj Close" if "Adj Close" in df.columns else "Close"

    df["Return"]           = df[c].pct_change()
    df["High_Low_Ratio"]   = df["High"] / df["Low"]
    df["Close_Open_Ratio"] = df["Close"] / df["Open"]

    for w in [5, 10, 20, 50]:
        df[f"MA_{w}"] = df[c].rolling(w).mean()
    df["MA_5_10_cross"]  = df["MA_5"]  - df["MA_10"]
    df["MA_10_20_cross"] = df["MA_10"] - df["MA_20"]
    df["MA_20_50_cross"] = df["MA_20"] - df["MA_50"]

    df["RSI_7"]  = rsi(df[c], 7)
    df["RSI_14"] = rsi(df[c], 14)

    exp1 = df[c].ewm(span=12, adjust=False).mean()
    exp2 = df[c].ewm(span=26, adjust=False).mean()
    df["MACD"]        = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    bb_ma  = df[c].rolling(20).mean()
    bb_std = df[c].rolling(20).std()
    df["BB_upper"]  = bb_ma + 2 * bb_std
    df["BB_lower"]  = bb_ma - 2 * bb_std
    df["BB_width"]  = (df["BB_upper"] - df["BB_lower"]) / (bb_ma + 1e-9)
    df["BB_pos"]    = (df[c] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)

    vol_ma = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"] = df["Volume"] / (vol_ma + 1e-9)
    df["OBV"]          = obv(df[c], df["Volume"])

    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14 + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df[c].shift()).abs()
    lpc = (df["Low"]  - df[c].shift()).abs()
    df["ATR_ratio"] = (
        pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
        / (df[c] + 1e-9)
    )

    df["CCI"]        = cci(df["High"], df["Low"], df["Close"])
    df["Williams_R"] = -100 * (high14 - df["Close"]) / (high14 - low14 + 1e-9)

    for w in [5, 10, 20]:
        df[f"Momentum_{w}"] = df[c] / (df[c].shift(w) + 1e-9) - 1

    df["Volatility_5"]  = df["Return"].rolling(5).std()
    df["Volatility_20"] = df["Return"].rolling(20).std()

    return df, c

def get_latest_sequence(symbol, sentiment_score=0.0):
    """Download recent data and build latest feature sequence for prediction."""
    end   = datetime.today()
    start = end - timedelta(days=max(LOOKBACK * 5, 120))
    raw   = yf.download(symbol, start=start.strftime('%Y-%m-%d'),
                        end=end.strftime('%Y-%m-%d'), progress=False)
    if raw.empty or len(raw) < LOOKBACK + 30:
        raise ValueError(f"Not enough data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]
    raw, _ = engineer_features(raw)
    raw["sentiment"] = sentiment_score
    raw = raw.dropna()

    # Columns the model + scaler expect (from config or training defaults)
    feats = [f for f in FEATURES if f in raw.columns]
    if len(feats) < len(FEATURES) * 0.85:
        raise ValueError(f"Feature mismatch: have {len(feats)}/{len(FEATURES)} columns")

    df_scaled = raw.copy()
    if scaler is not None:
        # Align columns to scaler's expected input
        all_feats = FEATURES if FEATURES else feats
        avail     = [f for f in all_feats if f in raw.columns]
        df_scaled[avail] = scaler.transform(raw[avail])
        feats = avail

    seq = df_scaled[feats].values[-LOOKBACK:]
    if len(seq) < LOOKBACK:
        raise ValueError("Not enough rows after feature engineering")
    return seq[np.newaxis, ...].astype(np.float32), feats, raw

# ──────────────────────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static', template_folder='.')
CORS(app)

@app.route('/')
def index():
    html_path = os.path.join(BASE_DIR, 'index.html')
    if os.path.exists(html_path):
        return send_file(html_path)
    return jsonify({"status": "running", "message": "index.html not found in project dir"})

@app.route('/health')
def health():
    return jsonify({
        "status":     "ok",
        "model":      model is not None,
        "scaler":     scaler is not None,
        "finbert":    finbert_ok,
        "config":     bool(cfg),
        "timestamp":  datetime.now().isoformat()
    })

@app.route('/model_info')
def model_info():
    return jsonify({
        **cfg,
        "model_loaded":   model  is not None,
        "scaler_loaded":  scaler is not None,
        "finbert_loaded": finbert_ok,
    })


def _recent_ohlcv(raw_df, n=60):
    """Last n rows for chart/table (matches frontend recent_data)."""
    cl = "Adj Close" if "Adj Close" in raw_df.columns else "Close"
    take = raw_df.tail(n).copy()
    idx = take.index
    if isinstance(idx, pd.DatetimeIndex):
        dates = idx.strftime("%Y-%m-%d").tolist()
    else:
        dates = [str(x)[:10] for x in idx]
    out = []
    for i, dt in enumerate(dates):
        row = take.iloc[i]
        vol = float(row["Volume"]) if "Volume" in take.columns else None
        out.append({"Date": dt, "Close": float(row[cl]), "Volume": vol})
    return out


@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Run clone_file_v2.py first to train."}), 503

    data   = request.get_json(force=True)
    symbol = data.get("symbol", "AAPL").upper().strip()
    news   = data.get("news", [])   # list of headline strings

    # Sentiment
    sentiment_score = 0.0
    if news and finbert_ok:
        sentiment_score = get_news_sentiment(news)

    try:
        seq, feats_used, raw_df = get_latest_sequence(symbol, sentiment_score)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    prob   = float(model.predict(seq, verbose=0).flatten()[0])
    signal = "UP" if prob > BEST_THRESHOLD else "DOWN"
    conf_pct = round((prob if signal == "UP" else (1.0 - prob)) * 100, 2)

    cl_col = "Adj Close" if "Adj Close" in raw_df.columns else "Close"
    price  = float(raw_df[cl_col].iloc[-1])
    rsi_val = float(raw_df["RSI_14"].iloc[-1]) if "RSI_14" in raw_df.columns else None
    macd_v  = float(raw_df["MACD"].iloc[-1]) if "MACD" in raw_df.columns else None
    vol_ratio = float(raw_df["Volume_Ratio"].iloc[-1]) if "Volume_Ratio" in raw_df.columns else None
    bb_w = float(raw_df["BB_width"].iloc[-1]) if "BB_width" in raw_df.columns else None
    headlines = [str(h).strip() for h in (news or []) if str(h).strip()]
    recent = _recent_ohlcv(raw_df, n=60)

    payload = {
        "symbol":           symbol,
        "signal":           signal,
        "prediction":       signal,
        "probability_up":   round(prob, 4),
        "probability_down": round(1 - prob, 4),
        "probability":       conf_pct,
        "threshold":        BEST_THRESHOLD,
        "sentiment_score":  round(sentiment_score, 4),
        "avg_sentiment":    round(sentiment_score, 4),
        "latest_price":     round(price, 2),
        "current_price":    round(price, 2),
        "rsi":              round(rsi_val, 2) if rsi_val is not None else None,
        "macd":             round(macd_v, 4) if macd_v is not None else None,
        "volume_ratio":     round(vol_ratio, 4) if vol_ratio is not None else None,
        "bb_width":         round(bb_w, 6) if bb_w is not None else None,
        "headlines":        headlines,
        "recent_data":      recent,
        "features_used":    len(feats_used),
        "lookback_days":    LOOKBACK,
        "forward_days":     FORWARD_DAYS,
        "return_threshold_pct": round(float(RETURN_THRESHOLD) * 100, 2),
        "timestamp":        datetime.now().isoformat(),
    }
    return jsonify(payload)

# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  Starting server at http://localhost:5000")
    print("  Endpoints:")
    print("    GET  /           → UI")
    print("    POST /predict, /api/predict → { symbol, news[] } → prediction")
    print("    GET  /health     → status check")
    print("    GET  /model_info → config")
    print("=" * 55)
    app.run(debug=False, port=5000, host='0.0.0.0')
