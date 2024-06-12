config = {
    'stock_tickers': [], 
    'start_date': '2020-01-01', 
    'end_date': '2023-01-01', 
    'indicators': {
        'SMA': {
            'window': [20, 50, 100]
        },
        'EMA': {
            'span': [20, 50, 100]
        },
        'MA_Envelope': {
            'window': [20],
            'percentage': 0.025
        },
        'MA_Deviation': {
            'window': [20]
        },
        'Bollinger_Bands': {
            'window': 20
        },
        'RSI': {
            'window': 14
        },
        'MACD': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
    }
}