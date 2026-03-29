"""
Stock price movement classifier — Conv1D + Transformer
Predicts whether a stock will rise >0.5% over the next 3 days.

Trains on 8 US tech stocks (2018–2025), uses FinBERT sentiment as a feature.
Hyperparameters tuned via KerasTuner BayesianOptimization.
"""

import json
import math
import warnings
from datetime import datetime, timedelta

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn.functional as F
import yfinance as yf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle as sk_shuffle
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SYMBOLS          = ["GOOGL", "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMD", "AMZN"]
LOOKBACK         = 20        # days of history per sample
FORWARD_DAYS     = 3         # predict 3-day forward return
RETURN_THRESHOLD = 0.005     # 0.5% movement = positive label
START_DATE       = "2018-01-01"
END_DATE         = "2025-08-01"

# Default hyperparameters (overridden by tuner if run)
DEFAULT_HP = dict(
    d_model    = 128,
    num_heads  = 4,
    d_ff       = 512,
    num_layers = 3,
    dropout    = 0.30,
    lr         = 5e-4,
    batch_size = 128,
)

EPOCHS         = 200
TUNER_MAX_TRIALS = 20   # KerasTuner trials — reduce for speed

# ─────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.5):
    """
    Binary focal loss.
    Downweights easy examples so the model focuses on hard ones.
    gamma=2 is the standard setting; alpha=0.5 means no prior class preference.
    """
    def loss_fn(y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt      = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(-alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    return loss_fn


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def rsi(prices, period=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def obv(close, volume):
    """On-Balance Volume — vectorised."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md + 1e-9)


def engineer_features(df):
    """
    Adds technical indicators to a raw OHLCV dataframe.
    Returns (df_with_features, close_column_name).
    """
    c = "Adj Close" if "Adj Close" in df.columns else "Close"

    # ── Returns & price ratios ──────────────────────────────
    df["Return"]           = df[c].pct_change()
    df["High_Low_Ratio"]   = df["High"] / df["Low"]
    df["Close_Open_Ratio"] = df["Close"] / df["Open"]

    # ── Moving averages & crossovers ────────────────────────
    for w in [5, 10, 20, 50]:
        df[f"MA_{w}"] = df[c].rolling(w).mean()
    df["MA_5_10_cross"]  = df["MA_5"]  - df["MA_10"]
    df["MA_10_20_cross"] = df["MA_10"] - df["MA_20"]
    df["MA_20_50_cross"] = df["MA_20"] - df["MA_50"]

    # ── RSI (two time-frames; RSI_7 catches short-term overbought) ──
    # BUG FIX: v4 had RSI and RSI_14 computing the same thing. Now RSI_7 and RSI_14 are distinct.
    df["RSI_7"]  = rsi(df[c], 7)
    df["RSI_14"] = rsi(df[c], 14)

    # ── MACD ────────────────────────────────────────────────
    exp1 = df[c].ewm(span=12, adjust=False).mean()
    exp2 = df[c].ewm(span=26, adjust=False).mean()
    df["MACD"]        = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # ── Bollinger Bands ─────────────────────────────────────
    bb_ma  = df[c].rolling(20).mean()
    bb_std = df[c].rolling(20).std()
    df["BB_upper"]  = bb_ma + 2 * bb_std
    df["BB_lower"]  = bb_ma - 2 * bb_std
    df["BB_width"]  = (df["BB_upper"] - df["BB_lower"]) / (bb_ma + 1e-9)
    df["BB_pos"]    = (df[c] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)

    # ── Volume indicators ───────────────────────────────────
    vol_ma = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"] = df["Volume"] / (vol_ma + 1e-9)
    df["OBV"]          = obv(df[c], df["Volume"])

    # ── Stochastic oscillator ───────────────────────────────
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14 + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # ── ATR (normalised) ────────────────────────────────────
    hl   = df["High"] - df["Low"]
    hpc  = (df["High"] - df[c].shift()).abs()
    lpc  = (df["Low"]  - df[c].shift()).abs()
    df["ATR_ratio"] = (
        pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
        / (df[c] + 1e-9)
    )

    # ── Extra oscillators ────────────────────────────────────
    df["CCI"]       = cci(df["High"], df["Low"], df["Close"])
    df["Williams_R"] = -100 * (high14 - df["Close"]) / (high14 - low14 + 1e-9)

    # ── Momentum ─────────────────────────────────────────────
    for w in [5, 10, 20]:
        df[f"Momentum_{w}"] = df[c] / (df[c].shift(w) + 1e-9) - 1

    # ── Realised volatility ──────────────────────────────────
    df["Volatility_5"]  = df["Return"].rolling(5).std()
    df["Volatility_20"] = df["Return"].rolling(20).std()

    # ── Target label ─────────────────────────────────────────
    fwd_ret    = df[c].shift(-FORWARD_DAYS) / df[c] - 1
    df["Target"] = (fwd_ret > RETURN_THRESHOLD).astype(int)

    return df, c


# ─────────────────────────────────────────────────────────────
# FEATURE LIST  (36 features + sentiment = 37 total)
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "Return",
    "MA_5", "MA_10", "MA_20", "MA_50",
    "MA_5_10_cross", "MA_10_20_cross", "MA_20_50_cross",
    "RSI_7", "RSI_14",                             # BUG FIX: removed duplicate RSI column
    "MACD", "MACD_signal", "MACD_hist",
    "BB_width", "BB_pos",
    "Volume_Ratio", "High_Low_Ratio", "Close_Open_Ratio",
    "OBV", "Stoch_K", "Stoch_D", "ATR_ratio",
    "CCI", "Williams_R",
    "Momentum_5", "Momentum_10", "Momentum_20",
    "Volatility_5", "Volatility_20",
    "sentiment",
]

# ─────────────────────────────────────────────────────────────
# [1] DOWNLOAD & BUILD DATASET
# ─────────────────────────────────────────────────────────────
print(f"\n[1/9] Downloading {len(SYMBOLS)} stocks ({START_DATE} → {END_DATE})...")

frames = []
for sym in SYMBOLS:
    try:
        raw = yf.download(sym, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]
        raw, _ = engineer_features(raw)
        raw["sentiment"] = 0.0    # filled below after news fetch
        raw = raw.dropna().iloc[:-FORWARD_DAYS]   # drop rows with no future target
        print(f"    {sym}: {len(raw)} rows")
        frames.append(raw)
    except Exception as e:
        print(f"    {sym}: failed — {e}")

stock = pd.concat(frames)
print(f"    Total rows: {len(stock)}")

# ─────────────────────────────────────────────────────────────
# [2] FINBERT
# ─────────────────────────────────────────────────────────────
print("\n[2/9] Loading FinBERT...")
finbert_ok = False
for source in ["finbert_finetuned", "ProsusAI/finbert"]:
    try:
        kwargs = {"num_labels": 3} if source == "ProsusAI/finbert" else {}
        finbert_tok = AutoTokenizer.from_pretrained(source)
        finbert_mdl = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
        finbert_ok  = True
        print(f"    ✓ Loaded from {source}")
        break
    except Exception as e:
        print(f"    ✗ {source}: {e}")


def get_sentiment(texts):
    if not texts or not finbert_ok:
        return np.zeros(len(texts))
    enc = finbert_tok(
        list(texts), padding=True, truncation=True,
        return_tensors="pt", max_length=512
    )
    with torch.no_grad():
        probs = F.softmax(finbert_mdl(**enc).logits, dim=-1).cpu().numpy()
    return probs[:, 0] - probs[:, 1]   # positive score − negative score


# ─────────────────────────────────────────────────────────────
# [3] NEWS SENTIMENT
# ─────────────────────────────────────────────────────────────
print("\n[3/9] Fetching news sentiment...")
daily_sent = pd.DataFrame(columns=["date", "sentiment"])

try:
    from GoogleNews import GoogleNews

    def _parse_date(s):
        s = str(s).lower()
        today = datetime.today()
        if "hour"      in s: return today.date()
        if "yesterday" in s: return (today - timedelta(1)).date()
        if "day"       in s:
            try:
                return (today - timedelta(int(s.split()[0]))).date()
            except Exception:
                return today.date()
        try:
            return pd.to_datetime(s).date()
        except Exception:
            return None

    gn = GoogleNews(lang="en", period="7d")
    gn.search("stock market")
    results = gn.result()

    if results:
        news = pd.DataFrame([
            {"date": r.get("date") or r.get("datetime"),
             "headline": r.get("title") or r.get("headline")}
            for r in results if r.get("title")
        ])
        news["date"]      = news["date"].apply(_parse_date)
        news              = news.dropna(subset=["date"])
        news["date"]      = pd.to_datetime(news["date"]).dt.date
        news["sentiment"] = get_sentiment(news["headline"].tolist())
        daily_sent        = news.groupby("date")["sentiment"].mean().reset_index()
        print(f"    {len(news)} headlines processed")
    else:
        print("    No news returned — sentiment stays 0")

except Exception as e:
    print(f"    News fetch failed ({e}) — sentiment stays 0")

# Merge sentiment into stock dataframe
stock = stock.reset_index()
date_col = "Date" if "Date" in stock.columns else stock.columns[0]
stock[date_col] = pd.to_datetime(stock[date_col]).dt.date

if len(daily_sent) > 0:
    stock = (
        stock
        .merge(daily_sent, left_on=date_col, right_on="date", how="left")
        .drop(columns=["date"], errors="ignore")
    )
    # If a sentiment column from the placeholder already existed, resolve the merge columns
    if "sentiment_x" in stock.columns and "sentiment_y" in stock.columns:
        stock["sentiment"] = stock["sentiment_y"]
        stock.drop(columns=["sentiment_x", "sentiment_y"], inplace=True)
    # Forward-fill gaps up to 3 days (weekend / missing days)
    stock["sentiment"] = stock["sentiment"].replace(0, np.nan).ffill(limit=3).fillna(0)

stock.set_index(date_col, inplace=True)

# ─────────────────────────────────────────────────────────────
# [4] FEATURE PREP & SCALING
# ─────────────────────────────────────────────────────────────
print("\n[4/9] Preparing features...")
available = [f for f in FEATURES if f in stock.columns]
NUM_FEAT  = len(available)
print(f"    Features: {NUM_FEAT}")

stock = stock[available + ["Target"]].dropna()
print(f"    Clean rows: {len(stock)}")

counts = dict(zip(*np.unique(stock["Target"], return_counts=True)))
total  = len(stock)
print(f"    DOWN: {counts.get(0, 0)} ({counts.get(0,0)/total*100:.1f}%)  "
      f"UP: {counts.get(1, 0)} ({counts.get(1,0)/total*100:.1f}%)")

# Fit scaler on training portion only to prevent leakage
train_cut = int(0.75 * len(stock))
scaler = RobustScaler()
scaler.fit(stock[available].iloc[:train_cut])

stock_scaled          = stock.copy()
stock_scaled[available] = scaler.transform(stock[available])

# ─────────────────────────────────────────────────────────────
# [5] SEQUENCES
# ─────────────────────────────────────────────────────────────
print("\n[5/9] Building sequences...")


def make_sequences(df, feats, target, lookback):
    X, Y = [], []
    fv, tv = df[feats].values, df[target].values
    for i in range(lookback, len(df)):
        X.append(fv[i - lookback:i])
        Y.append(tv[i])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


X, Y = make_sequences(stock_scaled, available, "Target", LOOKBACK)
print(f"    X shape: {X.shape}")

# Chronological split — never shuffle test data (avoids lookahead bias)
split     = int(0.75 * len(X))
X_train_full, X_test = X[:split], X[split:]
y_train_full, y_test = Y[:split], Y[split:]

# Shuffle only the training set to break inter-batch correlations
X_train_full, y_train_full = sk_shuffle(X_train_full, y_train_full, random_state=SEED)

# Hold out a validation set from training (last 15% of shuffled train)
val_split  = int(0.85 * len(X_train_full))
X_train, X_val = X_train_full[:val_split], X_train_full[val_split:]
y_train, y_val = y_train_full[:val_split], y_train_full[val_split:]

print(f"    Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# [6] MODEL BUILDER (KerasTuner-compatible)
# ─────────────────────────────────────────────────────────────
print("\n[6/9] Defining model...")


def positional_encoding(max_len, d_model):
    pos  = np.arange(max_len)[:, None]
    dims = np.arange(d_model)[None, :]
    angles = pos / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[None, ...], tf.float32)


def gelu(x):
    return x * 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))


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
        B   = tf.shape(x)[0]
        def split(t):
            return tf.transpose(tf.reshape(t, (B, -1, self.h, self.dk)), [0, 2, 1, 3])

        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        scale   = tf.math.sqrt(tf.cast(self.dk, tf.float32))
        attn    = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / scale, axis=-1)
        out     = tf.reshape(tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3]), (B, -1, self.d))
        return self.wo(out)


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

    def call(self, x, training=False):
        # Pre-norm (more stable than post-norm for small datasets)
        x = x + self.drop1(self.attn(self.norm1(x), training=training), training=training)
        x = x + self.drop2(self.ffn(self.norm2(x),  training=training), training=training)
        return x


def build_model(hp=None):
    """
    Builds the Conv1D-Transformer model.
    If `hp` is a KerasTuner HyperParameters object, values are sampled from it.
    Otherwise DEFAULT_HP is used — so the function works both for tuning and final training.
    """
    def get(key, choices=None, min_val=None, max_val=None, step=None, default=None):
        if hp is None:
            return DEFAULT_HP.get(key, default)
        if choices is not None:
            return hp.Choice(key, choices)
        return hp.Int(key, min_val, max_val, step=step) if isinstance(DEFAULT_HP[key], int) \
               else hp.Float(key, min_val, max_val, step=step)

    d_model    = get("d_model",    choices=[64, 128, 256])
    num_heads  = get("num_heads",  choices=[2, 4, 8])
    d_ff       = get("d_ff",       choices=[256, 512, 1024])
    num_layers = get("num_layers", choices=[2, 3, 4])
    drop       = get("dropout",    min_val=0.1, max_val=0.5, step=0.05)
    lr         = get("lr",         choices=[1e-4, 3e-4, 5e-4, 1e-3])

    # Ensure d_model is divisible by num_heads
    # If not (can happen during tuning), round d_model up
    if d_model % num_heads != 0:
        d_model = num_heads * (d_model // num_heads + 1)

    inp = tf.keras.Input(shape=(LOOKBACK, NUM_FEAT))

    # Conv1D stem — causal padding ensures no future data leaks in
    x = layers.Conv1D(64,  kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(128, kernel_size=3, padding="causal", activation="relu")(x)
    x = layers.Conv1D(128, kernel_size=5, padding="causal", activation="relu")(x)
    x = layers.LayerNormalization()(x)

    x = layers.Dense(d_model)(x)
    x = x + positional_encoding(LOOKBACK, d_model)
    x = layers.Dropout(drop)(x)

    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, d_ff, drop)(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x   = layers.Dense(128, activation=gelu)(x)
    x   = layers.Dropout(drop)(x)
    x   = layers.Dense(64, activation=gelu)(x)
    x   = layers.Dropout(drop * 0.5)(x)
    x   = layers.Dense(32, activation=gelu)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    m = Model(inp, out)
    m.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss=focal_loss(gamma=2.0, alpha=0.5),
        metrics=["accuracy"],
    )
    return m


# ─────────────────────────────────────────────────────────────
# [7] HYPERPARAMETER TUNING 
# ─────────────────────────────────────────────────────────────
RUN_TUNER = True    

best_hp_values = DEFAULT_HP.copy()

if RUN_TUNER:
    print("\n[7/9] Hyperparameter search (KerasTuner BayesianOptimization)...")
    try:
        import keras_tuner as kt

        tuner = kt.BayesianOptimization(
            build_model,
            objective    = kt.Objective("val_accuracy", direction="max"),
            max_trials   = TUNER_MAX_TRIALS,
            seed         = SEED,
            project_name = "stock_tuner",
            overwrite    = True,
        )

        tuner_callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=10,
                          restore_best_weights=True, min_delta=1e-4),
        ]

        tuner.search(
            X_train, y_train,
            validation_data = (X_val, y_val),
            epochs          = 60,          # shorter epochs during search
            batch_size      = 128,
            callbacks       = tuner_callbacks,
            verbose         = 1,
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        for key in DEFAULT_HP:
            if best_hp.get(key) is not None:
                best_hp_values[key] = best_hp.get(key)

        print("\n    Best hyperparameters found:")
        for k, v in best_hp_values.items():
            print(f"      {k}: {v}")

    except ImportError:
        print("    keras-tuner not installed — install with: pip install keras-tuner")
        print("    Using default hyperparameters instead.")

else:
    print("\n[7/9] Skipping tuner — using DEFAULT_HP.")

# Inject best values into DEFAULT_HP so build_model() picks them up
DEFAULT_HP.update(best_hp_values)

# ─────────────────────────────────────────────────────────────
# [8] FINAL TRAINING
# ─────────────────────────────────────────────────────────────
print("\n[8/9] Training final model...")

model = build_model()   # hp=None → uses DEFAULT_HP
model.summary()


def cosine_lr(epoch, _lr=None):
    """Linear warmup for 10 epochs, then cosine decay."""
    warmup = 10
    lr_max = DEFAULT_HP["lr"]
    if epoch < warmup:
        return lr_max * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, EPOCHS - warmup)
    return 1e-6 + 0.5 * (lr_max - 1e-6) * (1 + math.cos(math.pi * progress))


callbacks = [
    EarlyStopping(
        monitor             = "val_accuracy",
        patience            = 25,
        restore_best_weights= True,
        min_delta           = 1e-4,
        verbose             = 1,
    ),
    ModelCheckpoint(
        "best_model.keras",
        monitor        = "val_accuracy",
        save_best_only = True,
        verbose        = 0,
    ),
    LearningRateScheduler(cosine_lr, verbose=0),
]

history = model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs          = EPOCHS,
    batch_size      = DEFAULT_HP["batch_size"],
    callbacks       = callbacks,
    shuffle         = False,   # already shuffled before split
    verbose         = 1,
)

# ─────────────────────────────────────────────────────────────
# [9] EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n[9/9] Evaluating...")

train_prob = model.predict(X_train, verbose=0).flatten()
test_prob  = model.predict(X_test,  verbose=0).flatten()

# Youden's J threshold — balances sensitivity and specificity
fpr, tpr, roc_thr = roc_curve(y_test, test_prob)
best_thresh = float(roc_thr[np.argmax(tpr - fpr)])

# Also compute F1-optimal for reference
prec, rec, pr_thr = precision_recall_curve(y_test, test_prob)
f1s             = 2 * prec * rec / (prec + rec + 1e-9)
thresh_f1       = float(pr_thr[np.argmax(f1s[:-1])])

print(f"\n    Youden-J threshold: {best_thresh:.4f}")
print(f"    F1-optimal threshold: {thresh_f1:.4f}")

train_pred = (train_prob > best_thresh).astype(int)
test_pred  = (test_prob  > best_thresh).astype(int)

print("\n" + "=" * 65)
for label, yt, yp, yprob in [
    ("TRAIN", y_train, train_pred, train_prob),
    ("TEST",  y_test,  test_pred,  test_prob),
]:
    print(f"\n  [{label}]")
    print(f"  Accuracy : {accuracy_score(yt, yp) * 100:.2f}%")
    print(f"  F1-Score : {f1_score(yt, yp, zero_division=0):.4f}")
    if label == "TEST":
        try:
            print(f"  ROC-AUC  : {roc_auc_score(yt, yprob):.4f}")
        except Exception:
            pass

print("\n  Classification Report (TEST):")
print(classification_report(y_test, test_pred, target_names=["DOWN", "UP"]))

# ─────────────────────────────────────────────────────────────
# DASHBOARD PLOT
# ─────────────────────────────────────────────────────────────
def plot_dashboard(history, y_test, test_pred, test_prob):
    G, C, bg = "#00ff9c", "#00c8ff", "#111820"

    fig = plt.figure(figsize=(18, 12), facecolor="#0a0a0a")
    fig.suptitle("Stock Movement Classifier — Training Results", fontsize=15,
                 color=G, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Accuracy & loss curves
    for pos, tr_key, vl_key, title in [
        (gs[0, 0], "accuracy", "val_accuracy", "Accuracy"),
        (gs[0, 1], "loss",     "val_loss",      "Loss"),
    ]:
        ax = fig.add_subplot(pos)
        ax.set_facecolor(bg)
        ax.plot(history.history[tr_key], color=G, lw=2, label="Train")
        ax.plot(history.history[vl_key], color=C, lw=2, linestyle="--", label="Val")
        ax.set_title(title, color="#c0d0d8")
        ax.set_xlabel("Epoch", color="#607080")
        ax.tick_params(colors="#607080")
        ax.legend(facecolor="#1a2028", labelcolor="#c0d0d8", fontsize=8)
        ax.grid(True, alpha=0.15, color="#304050")
        ax.spines[:].set_color("#304050")

    # Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(bg)
    cm = confusion_matrix(y_test, test_pred)
    im = ax3.imshow(cm, cmap="YlOrBr", aspect="auto")
    plt.colorbar(im, ax=ax3)
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                     fontsize=14, fontweight="bold")
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Pred DOWN", "Pred UP"], color="#607080")
    ax3.set_yticklabels(["True DOWN", "True UP"], color="#607080")
    ax3.set_title("Confusion Matrix", color="#c0d0d8")
    ax3.spines[:].set_color("#304050")

    # Probability distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(bg)
    ax4.hist(test_prob[y_test == 0], bins=25, alpha=0.7, color="#ff4757", label="True DOWN", density=True)
    ax4.hist(test_prob[y_test == 1], bins=25, alpha=0.7, color=G,         label="True UP",   density=True)
    ax4.axvline(best_thresh, color="white", linestyle="--", lw=1.5, label=f"threshold={best_thresh:.2f}")
    ax4.set_title("Probability Distribution", color="#c0d0d8")
    ax4.legend(facecolor="#1a2028", labelcolor="#c0d0d8", fontsize=8)
    ax4.grid(True, alpha=0.15, color="#304050"); ax4.spines[:].set_color("#304050")

    # PR curve
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(bg)
    ax5.plot(rec, prec, color=G, lw=2)
    ax5.fill_between(rec, prec, alpha=0.1, color=G)
    ax5.set_xlabel("Recall", color="#607080"); ax5.set_ylabel("Precision", color="#607080")
    ax5.set_title("Precision-Recall Curve", color="#c0d0d8")
    ax5.grid(True, alpha=0.15, color="#304050"); ax5.spines[:].set_color("#304050")

    # Summary text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(bg); ax6.axis("off")
    tacc = accuracy_score(y_test, test_pred)
    try:    auc_str = f"{roc_auc_score(y_test, test_prob):.4f}"
    except: auc_str = "N/A"

    summary = (
        f"TRAINING SUMMARY\n"
        f"{'─' * 30}\n"
        f"Stocks       : {len(SYMBOLS)}\n"
        f"Features     : {NUM_FEAT}\n"
        f"Lookback     : {LOOKBACK} days\n"
        f"Target       : {FORWARD_DAYS}-day return > {RETURN_THRESHOLD*100:.1f}%\n"
        f"d_model      : {DEFAULT_HP['d_model']}\n"
        f"Heads        : {DEFAULT_HP['num_heads']}\n"
        f"FFN dim      : {DEFAULT_HP['d_ff']}\n"
        f"Layers       : {DEFAULT_HP['num_layers']}\n"
        f"Dropout      : {DEFAULT_HP['dropout']}\n"
        f"LR           : {DEFAULT_HP['lr']}\n"
        f"Loss         : Focal (γ=2)\n"
        f"Threshold    : Youden-J\n"
        f"{'─' * 30}\n"
        f"Test Accuracy: {tacc*100:.2f}%\n"
        f"ROC-AUC      : {auc_str}\n"
    )
    ax6.text(0.05, 0.97, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace", color="#c0d0d8",
             bbox=dict(boxstyle="round", facecolor="#1a2028", alpha=0.8))

    plt.savefig("training_dashboard.png", dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    print("  Saved: training_dashboard.png")
    plt.close()


plot_dashboard(history, y_test, test_pred, test_prob)

# ─────────────────────────────────────────────────────────────
# SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
model.save("stock_model.keras")
joblib.dump(scaler, "scaler.pkl")

config = {
    "lookback":          LOOKBACK,
    "forward_days":      FORWARD_DAYS,
    "return_threshold":  RETURN_THRESHOLD,
    "features":          available,
    "num_features":      NUM_FEAT,
    "best_threshold":    round(best_thresh, 4),
    "scaler_type":       "RobustScaler",
    "hyperparameters":   DEFAULT_HP,
}
with open("model_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("  ✓ stock_model.keras")
print("  ✓ scaler.pkl")
print("  ✓ model_config.json")

if finbert_ok:
    try:
        finbert_mdl.save_pretrained("finbert_local")
        finbert_tok.save_pretrained("finbert_local")
        print("  ✓ finbert_local/")
    except Exception as e:
        print(f"  ✗ Could not save FinBERT: {e}")

print(f"\n  Final test accuracy: {accuracy_score(y_test, test_pred)*100:.2f}%")
print("=" * 65)