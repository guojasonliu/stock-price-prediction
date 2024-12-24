from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd
import os


def load_ticker(ticker='^GSPC', start='1900-01-01', target_file='../data/s&p500.csv', use_if_exists=True):
    try:
        if use_if_exists and os.path.exists(target_file):
            df = pd.read_csv(target_file, header=[0, 1], index_col=0)
        else:
            df = yf.download(ticker, start=start, progress=False)

            if df.empty:
                print('Ticker is empty')
                return None
            
            df.to_csv(target_file)
        
        df.index = pd.to_datetime(df.index)
        date, adj_close = df.index.to_numpy(), df[('Adj Close', ticker)].to_numpy()
        data_df = pd.DataFrame(
        {
            'Date': date,
            'Adj Close': adj_close
        })
        return date, adj_close, data_df

    except Exception as e:
        print(f'Error occured while loading ticker {ticker}: {e}')
        return None

def split_data(x, y, ratio=0.8, shuffle=False):
    n = len(x)
    train_n = int(ratio * n)

    if shuffle:
        pass
    else:
        x_train, y_train = x[:train_n], y[:train_n]
        x_test, y_test = x[train_n:], y[train_n:]
        return x_train, y_train, x_test, y_test

def min_max_scale(data, window_size=-1):
    scaler, data = MinMaxScaler(), data.reshape(-1, 1)
    n = data.shape[0]

    if window_size == -1: window_size = n

    for i in range(0, n, window_size):
        eff_window_size = min(window_size, n - i)
        data[i:i + eff_window_size, :] = scaler.fit_transform(data[i:i + eff_window_size, :])

    data = data.reshape(-1)
    return data, scaler

def exp_mov_avg_smooth(data, gamma=0.1):
    n, ema = len(data), 0.0
    for i in range(n):
        ema = gamma * data[i] + (1 - gamma) * ema
        data[i] = ema
    return data