# %%
# Add visualization code for balance curve or trades
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import talib
import plotly.express as px


# %%

def custom_trading_strategy(df, initial_balance=5000, period=878, mode=1, invert_signal=1, filter1=41, filter2=0, commission=0.0001):
    """
    Custom trading strategy implementation based on the provided TradingView logic.
    
    Args:
        df (pd.DataFrame): A dataframe with columns 'date', 'open', 'high', 'low', 'close', 'volume'.
        initial_balance (float): Starting capital for backtesting (changed to 5000).
        period (int): The period for calculating the summation (similar to SMA in Pine Script).
        mode (int): The trading mode (1, 2, 3, or 4) to determine entry and exit conditions.
        invert_signal (int): Signal inversion (1 for normal, -1 for inverse).
        filter1 (float): The first filter threshold for the green condition.
        filter2 (float): The second filter threshold for the red condition.
        commission (float): Commission per trade as a fraction of trade size (changed to 0.0001 for 0.01%).
    
    Returns:
        pd.DataFrame: A dataframe containing the original data with buy/sell signals, profits, and positions.
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

    # Initialize position and profit tracking columns
    df['position'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['profit'] = 0
    df['balance'] = initial_balance
    df['buy_signal'] = 0
    df['sell_signal'] = 0
    position_size = 0
    is_long = False
    is_short = False

    # Iterate through the DataFrame and apply the strategy
    for i in range(1, len(df)):
        if df['signal'].iloc[i-1] == 1 and not is_long:
            # Long entry
            df.at[i, 'position'] = 1
            df.at[i, 'entry_price'] = df['open'].iloc[i]  # Changed from close to open
            position_size = df['balance'].iloc[i-1] / df['open'].iloc[i]  # Buy shares
            is_long = True
            is_short = False
            df.at[i, 'balance'] = df['balance'].iloc[i-1] - (df['open'].iloc[i] * position_size * (1 + commission))
            df.at[i, 'buy_signal'] = 1

        elif df['signal'].iloc[i-1] == -1 and not is_short:
            # Short entry
            df.at[i, 'position'] = -1
            df.at[i, 'entry_price'] = df['open'].iloc[i]  # Changed from close to open
            position_size = df['balance'].iloc[i-1] / df['open'].iloc[i]  # Short shares
            is_short = True
            is_long = False
            df.at[i, 'balance'] = df['balance'].iloc[i-1] - (df['open'].iloc[i] * position_size * (1 + commission))
            df.at[i, 'sell_signal'] = 1

        elif df['position'].iloc[i-1] == 1 and is_long:
            # Exit long position if the signal turns negative
            if df['signal'].iloc[i] == -1 or (df['position'].iloc[i] == 0):
                df.at[i, 'exit_price'] = df['open'].iloc[i]  # Changed from close to open
                df.at[i, 'profit'] = (df['exit_price'].iloc[i] - df['entry_price'].iloc[i-1]) * position_size - (df['exit_price'].iloc[i] * position_size * commission * 2)  # Added commission for entry and exit
                df.at[i, 'balance'] = df['balance'].iloc[i-1] + df['profit'].iloc[i]
                position_size = 0
                is_long = False
                df.at[i, 'sell_signal'] = 1

        elif df['position'].iloc[i-1] == -1 and is_short:
            # Exit short position if the signal turns positive
            if df['signal'].iloc[i] == 1 or (df['position'].iloc[i] == 0):
                df.at[i, 'exit_price'] = df['open'].iloc[i]  # Changed from close to open
                df.at[i, 'profit'] = (df['entry_price'].iloc[i-1] - df['exit_price'].iloc[i]) * position_size - (df['exit_price'].iloc[i] * position_size * commission * 2)  # Added commission for entry and exit
                df.at[i, 'balance'] = df['balance'].iloc[i-1] + df['profit'].iloc[i]
                position_size = 0
                is_short = False
                df.at[i, 'buy_signal'] = 1

        # If no trade occurs, carry forward the balance
        if np.isnan(df['balance'].iloc[i]):
            df.at[i, 'balance'] = df['balance'].iloc[i-1]

    # Calculate cumulative balance
    df['balance'] = df['balance'].cumsum()

    return df


# %%
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

df = pd.read_csv('../NDX5years.csv')
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

# Apply the custom trading strategy
df_with_signals = custom_trading_strategy(df)

# Save results
import os
path = os.getcwd()
df_with_signals.to_csv(f'{path}/NDX5years_signals.csv')


# %%
df_with_signals

# %%
# Create subplots: candlestick chart and volume bars
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.03, subplot_titles=('NASDAQ', 'Balance Curve'), 
                    row_width=[0.7, 0.3])

# Add candlestick chart
fig.add_trace(go.Candlestick(
    x=df_with_signals['date'],
    open=df_with_signals['open'],
    high=df_with_signals['high'],
    low=df_with_signals['low'],
    close=df_with_signals['close'],
    name='OHLC'
), row=1, col=1)

# Add balance curve
fig.add_trace(go.Scatter(
    x=df_with_signals['date'],
    y=df_with_signals['balance'],
    mode='lines',
    name='Balance',
    line=dict(color='blue')
), row=2, col=1)

# Update layout
fig.update_layout(
    title='NASDAQ Trading Strategy with Balance Curve',
    xaxis_rangeslider_visible=False,
    height=800,
    showlegend=True,
    xaxis_title='Date',
    yaxis_title='Price',
    yaxis2_title='Balance'
)


# %%
# Show the figure
fig



# %%
# Prepare data for visualizations
df_with_signals['date'] = pd.to_datetime(df_with_signals['date'])
df_with_signals['profit_loss'] = df_with_signals['profit'].fillna(0)
df_with_signals['cumulative_profit_loss'] = df_with_signals['profit_loss'].cumsum()
df_with_signals['days_in_trade'] = df_with_signals['position'].groupby((df_with_signals['position'] != df_with_signals['position'].shift()).cumsum()).cumcount() + 1
df_with_signals['commission_paid'] = abs(df_with_signals['profit']) * commission

# %%
# Chart 1: Profit and Loss grouped by date
fig_pnl = px.bar(df_with_signals, x='date', y='profit_loss', title='Daily Profit and Loss')
fig_pnl.add_trace(go.Scatter(x=df_with_signals['date'], y=df_with_signals['cumulative_profit_loss'], 
                             mode='lines', name='Cumulative P&L'))
fig_pnl.update_layout(xaxis_title='Date', yaxis_title='Profit/Loss')
fig_pnl.show()

# %%
# Chart 2: Candlestick chart with buy/sell signals and exits
fig_signals = go.Figure(data=[go.Candlestick(x=df_with_signals['date'],
                open=df_with_signals['open'], high=df_with_signals['high'],
                low=df_with_signals['low'], close=df_with_signals['close'])])

# Add buy signals
fig_signals.add_trace(go.Scatter(
    x=df_with_signals.loc[df_with_signals['buy_signal'] == 1, 'date'],
    y=df_with_signals.loc[df_with_signals['buy_signal'] == 1, 'low'],
    mode='markers',
    marker=dict(symbol='triangle-up', size=10, color='green'),
    name='Buy Signal'
))

# Add sell signals
fig_signals.add_trace(go.Scatter(
    x=df_with_signals.loc[df_with_signals['sell_signal'] == 1, 'date'],
    y=df_with_signals.loc[df_with_signals['sell_signal'] == 1, 'high'],
    mode='markers',
    marker=dict(symbol='triangle-down', size=10, color='red'),
    name='Sell Signal'
))

# Add exit points
fig_signals.add_trace(go.Scatter(
    x=df_with_signals.loc[df_with_signals['exit_price'].notna(), 'date'],
    y=df_with_signals.loc[df_with_signals['exit_price'].notna(), 'exit_price'],
    mode='markers',
    marker=dict(symbol='circle', size=8, color='purple'),
    name='Exit Point'
))

fig_signals.update_layout(title='NASDAQ with Buy/Sell Signals and Exits',
                          xaxis_title='Date',
                          yaxis_title='Price')
fig_signals.show()

# %%
# Chart 3: Commissions and Days in Order
fig_commission = make_subplots(specs=[[{"secondary_y": True}]])

fig_commission.add_trace(
    go.Bar(x=df_with_signals['date'], y=df_with_signals['commission_paid'], name="Commission Paid"),
    secondary_y=False,
)

fig_commission.add_trace(
    go.Scatter(x=df_with_signals['date'], y=df_with_signals['days_in_trade'], name="Days in Trade"),
    secondary_y=True,
)

fig_commission.update_layout(
    title_text="Commissions Paid and Days in Trade",
    xaxis_title="Date",
)

fig_commission.update_yaxes(title_text="Commission", secondary_y=False)
fig_commission.update_yaxes(title_text="Days in Trade", secondary_y=True)

fig_commission.show()

# %%
# Show all figures
fig_pnl
fig_signals
fig_commission
