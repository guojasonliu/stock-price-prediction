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
        return df.index.to_numpy(), df[('Adj Close', '^GSPC')].to_numpy()

    except Exception as e:
        print(f'Error occured while loading ticker {ticker}: {e}')
        return None