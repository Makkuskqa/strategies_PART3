import pandas as pd
import numpy as np
import talib

def custom_trading_strategy(df, period=878, mode=1, invert_signal=1, filter1=41, filter2=0):
    """
    Custom trading strategy implementation based on the provided TradingView logic.
    
    Args:
        df (pd.DataFrame): A dataframe with columns 'date', 'open', 'high', 'low', 'close', 'volume'.
        period (int): The period for calculating the summation (similar to SMA in Pine Script).
        mode (int): The trading mode (1, 2, 3, or 4) to determine entry and exit conditions.
        invert_signal (int): Signal inversion (1 for normal, -1 for inverse).
        filter1 (float): The first filter threshold for the green condition.
        filter2 (float): The second filter threshold for the red condition.
    
    Returns:
        pd.DataFrame: A dataframe containing the original data with buy/sell signals appended.
    """

    # Calculate body and bodyVar (difference between current and previous body)
    df['body'] = df['close'] - df['open']
    df['bodyVar'] = df['body'].diff()

    # SumVar is similar to a moving average of bodyVar
    df['sumVar'] = talib.SMA(df['bodyVar'], timeperiod=period)

    # Initialize green and red columns
    df['green'] = np.where(df['sumVar'] > filter1, df['sumVar'], np.nan)
    df['red'] = np.where(df['sumVar'] < -filter2, df['sumVar'], np.nan)

    # Initialize signals for c1 and c2 based on trading mode
    df['c1'] = np.nan
    df['c2'] = np.nan

    if mode == 1:
        df['c1'] = df['red'] < df['red'].shift(1)
        df['c2'] = df['green'] > df['green'].shift(1)
    elif mode == 2:
        df['c1'] = df['red'] > df['red'].shift(1)
        df['c2'] = df['green'] < df['green'].shift(1)
    elif mode == 3:
        df['c1'] = df['red'] < df['red'].shift(1)
        df['c2'] = df['green'] < df['green'].shift(1)
    elif mode == 4:
        df['c1'] = df['red'] > df['red'].shift(1)
        df['c2'] = df['green'] > df['green'].shift(1)

    # Generate buy/sell signals based on c1 and c2
    df['signal'] = np.nan
    df['signal'] = np.where(df['c1'], 1 * invert_signal, df['signal'])
    df['signal'] = np.where(df['c2'], -1 * invert_signal, df['signal'])

    # Trading logic: Create buy/sell signals
    df['position'] = df['signal'].shift(1)
    df['buy'] = np.where(df['position'] > 0, 1, 0)  # Buy signal
    df['sell'] = np.where(df['position'] < 0, 1, 0)  # Sell signal

    return df







# Example usage:
# Load data into a dataframe (df should have columns: date, open, high, low, close, volume)
"""
data = {
    'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'open': np.random.rand(100) * 100,
    'high': np.random.rand(100) * 100 + 5,
    'low': np.random.rand(100) * 100 - 5,
    'close': np.random.rand(100) * 100,
    'volume': np.random.randint(1000, 5000, size=100)
}
df = pd.DataFrame(data)
"""


df = pd.read_csv('NDX5years.csv')
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

# Apply the custom trading strategy
df_with_signals = custom_trading_strategy(df)

# Show the dataframe with buy/sell signals
print(df_with_signals[['date', 'open', 'high', 'low', 'close', 'buy', 'sell']])

df_with_signals.to_csv('NDX5years_signals.csv')
