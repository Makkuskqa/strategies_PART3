import pandas as pd
import numpy as np

def one_percent_per_week_strategy(df, timezone='Europe/Berlin'):
    """
    1% per week trading strategy implementation.

    Args:
        df (pd.DataFrame): A dataframe with columns 'date', 'open', 'high', 'low', 'close', 'volume'.
        timezone (str): The timezone for the strategy (not used directly in this example but can be useful for time adjustments).
    
    Returns:
        pd.DataFrame: A dataframe with buy/sell signals and profit/loss for the strategy.
    """
    
    # Convert 'date' to pandas datetime and extract necessary time elements
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Set Monday 15:30 as the starting time and Friday 21:30 as closing time
    start_time = (15, 30)  # 15:30 on Monday
    end_time = (21, 30)    # 21:30 on Friday

    # Initialize signals and columns to hold strategy logic
    df['Kauf'] = np.nan
    df['AnpassungTP'] = np.nan
    df['signal'] = 0
    df['take_profit'] = np.nan
    df['stop_loss'] = np.nan
    df['position'] = 0  # 1 for long, 0 for no position

    for i in range(1, len(df)):
        current_day = df['day_of_week'].iloc[i]
        current_time = (df['hour'].iloc[i], df['minute'].iloc[i])
        
        # Check entry conditions (Monday 15:30)
        if current_day == 0 and current_time == start_time:
            df.at[i, 'Kauf'] = df['close'].iloc[i] - (df['close'].iloc[i] * 0.01)
            df.at[i, 'AnpassungTP'] = df['close'].iloc[i] - (df['close'].iloc[i] * 0.015)
            df.at[i, 'take_profit'] = df['Kauf'].iloc[i] * 1.01
            df.at[i, 'stop_loss'] = df['close'].iloc[i] * 0.95
            df.at[i, 'signal'] = 1  # Buy signal
            df.at[i, 'position'] = 1  # Long position entered

        # Exit conditions based on profit/loss or Friday close
        if df['position'].iloc[i-1] == 1:
            if df['close'].iloc[i] >= df['take_profit'].iloc[i-1]:
                df.at[i, 'signal'] = -1  # Take profit (sell)
                df.at[i, 'position'] = 0  # Exit position
            elif df['close'].iloc[i] <= df['stop_loss'].iloc[i-1]:
                df.at[i, 'signal'] = -1  # Stop loss (sell)
                df.at[i, 'position'] = 0  # Exit position
            elif current_day == 4 and current_time == end_time:
                df.at[i, 'signal'] = -1  # Close all positions on Friday 21:30
                df.at[i, 'position'] = 0

    # Output DataFrame with buy/sell signals and positions
    df['buy_signal'] = np.where(df['signal'] == 1, 1, 0)
    df['sell_signal'] = np.where(df['signal'] == -1, 1, 0)
    
    return df

# Example usage:
# Load data into a dataframe (df should have columns: date, open, high, low, close, volume)
data = {
    'date': pd.date_range(start='2023-01-01', periods=200, freq='H'),  # hourly data
    'open': np.random.rand(200) * 100,
    'high': np.random.rand(200) * 100 + 5,
    'low': np.random.rand(200) * 100 - 5,
    'close': np.random.rand(200) * 100,
    'volume': np.random.randint(1000, 5000, size=200)
}
df = pd.DataFrame(data)

# Apply the strategy
df_with_signals = one_percent_per_week_strategy(df)

# Display the result
print(df_with_signals[['date', 'close', 'buy_signal', 'sell_signal', 'position']])
