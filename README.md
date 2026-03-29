# Stock Prediction AI - Frontend

A web application for stock price prediction using LSTM neural networks with sentiment analysis from news articles.

## Features

- 📈 Real-time stock predictions using LSTM model
- 📰 Sentiment analysis from news articles using FinBERT
- 📊 Interactive charts showing recent stock data
- 💹 Detailed price information and predictions
- 🎨 Modern, responsive UI

## Prerequisites

- Python 3.8 or higher
- Trained model files:
  - `lstm_with_sentiment.h5` (LSTM model)
  - `scaler_sentiment.pkl` (Feature scaler)
  - `finbert_finetuned/` (FinBERT model directory, optional - will use pretrained if not found)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure your trained model files are in the project directory:
   - `lstm_with_sentiment.h5`
   - `scaler_sentiment.pkl`
   - `finbert_finetuned/` (optional)

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter a stock symbol (e.g., GOOGL, AAPL, MSFT) and click "Predict"

## Usage

1. Enter a stock symbol in the input field
2. Click the "Predict" button
3. View the prediction (UP/DOWN) with confidence percentage
4. See recent stock data in the chart and table
5. Check sentiment analysis from recent news articles

## API Endpoints

- `GET /` - Main web interface
- `POST /api/predict` - Make a stock prediction
  - Request body: `{"symbol": "GOOGL"}`
  - Returns: Prediction data with confidence, prices, and sentiment
- `GET /api/stock-info` - Get basic stock information
  - Query parameter: `symbol` (e.g., `?symbol=GOOGL`)

## Project Structure

```
.
├── app.py                 # Flask backend application
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Styling
│   └── js/
│       └── main.js       # Frontend JavaScript
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Notes

- The application fetches stock data from Yahoo Finance using `yfinance`
- News articles are fetched using `GoogleNews` library
- Sentiment analysis is performed using FinBERT transformer model
- Predictions are based on the last 5 days of data (lookback period)

## Troubleshooting

- If models fail to load, ensure all model files are in the correct location
- If news fetching fails, the app will still work with sentiment set to 0
- Make sure you have an active internet connection for fetching stock data and news

