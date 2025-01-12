import ccxt
import pandas as pd
import numpy as np
import datetime
import time
import requests
import threading
import yaml             #install package PyYAML
from io import StringIO
import warnings

# Load configuration from config.yaml file
config_file = 'config_BTC.yaml'
with open(config_file) as f:
    config = yaml.safe_load(f)
print('Loading config file',config_file,".....")
time.sleep(1)

API_KEY = config['GLASSNODE']['API_KEY']
MAX_POS = config['BYBIT']['max_pos']  # Maximum position size (BTC)
EXCHANGE = ccxt.bybit({
    'apiKey': config['BYBIT']['apiKey'],
    'secret': config['BYBIT']['secret'],
    'rateLimit': True,  # CCXT will automatically throttle requests to comply with Bybit’s rate limits.
})
SYMBOL = config['ASSET']['cex_symbol']
GLASSNODE_SYMBOL = config['ASSET']['glassnode_symbol']
DATESTART = config['ASSET']['since']

print('Symbol:', SYMBOL)
print('Glassnode price:', GLASSNODE_SYMBOL)
print('Since:', DATESTART)
print('Max. Position:', MAX_POS)
print('Strat1:', config['STRAT1']['ratio'], config['STRAT1']['x'], config['STRAT1']['y'],config['STRAT1']['api'],config['STRAT1']['api_symbol'],config['STRAT1']['resolution'])
print('Strat2:', config['STRAT2']['ratio'], config['STRAT2']['x'], config['STRAT2']['y'],config['STRAT2']['api'],config['STRAT2']['api_symbol'],config['STRAT2']['resolution'])
print('Strat3:', config['STRAT3']['ratio'], config['STRAT3']['x'], config['STRAT3']['y'],config['STRAT3']['api'],config['STRAT3']['api_symbol'],config['STRAT3']['resolution'])


# ===== Data Management in Memory =====
gn_data_1 = pd.DataFrame(columns=['t', 'value', 'price'])
gn_data_2 = pd.DataFrame(columns=['t', 'value', 'price'])
gn_data_3 = pd.DataFrame(columns=['t', 'value', 'price'])
signal_data = pd.DataFrame(columns=['dt', 'pos'])

# ===== Print full data table =====
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ===== Data Fetching Functions =====
def fetch_data(metric_url, asset, resolution, df_name):
    """
    Fetch metric data and BTC price data, and merge them into the specified DataFrame.
    """
    since = DATESTART  # Data start time
    until = int(time.time())
    print("Fetching Data:",metric_url,asset,resolution,df_name)

    # Fetch metric data
    res_value = requests.get(metric_url, params={
        "a": asset,
        "s": since,
        "u": until,
        "api_key": API_KEY,
        "i": resolution
    })
    df_value = pd.read_json(StringIO(res_value.text), convert_dates=['t'])

    # Fetch BTC price data
    res_price = requests.get("https://api.glassnode.com/v1/metrics/market/price_usd_close", params={
        "a": GLASSNODE_SYMBOL,
        "s": since,
        "u": until,
        "api_key": API_KEY,
        "i": resolution
    })
    df_price = pd.read_json(StringIO(res_price.text), convert_dates=['t'])

    # Merge data
    df = pd.merge(df_value, df_price, how='inner', on='t')
    df = df.rename(columns={'v_x': 'value', 'v_y': 'price'})
    print(df.tail(5))

    # Fix warning bug of pd.concat issue
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        # You can use a regex to match the specific warning message
        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. *"
    )

    # Use globals() to dynamically update the specified global DataFrame
    globals()[df_name] = pd.concat([globals()[df_name], df]).drop_duplicates(subset=['t']).reset_index(drop=True)

def print_acct_info():
    print('Bal:', EXCHANGE.fetch_balance()['USDT']['total'], 'Pos:',
          EXCHANGE.fetch_position(SYMBOL)['info']['side'], current_pos())
    print('Unrealized Pnl:', EXCHANGE.fetch_position(SYMBOL)['info']['unrealisedPnl'], 'Realized Pnl:',
          EXCHANGE.fetch_position(SYMBOL)['info']['curRealisedPnl'], 'CumPnl:',
          EXCHANGE.fetch_position(SYMBOL)['info']['cumRealisedPnl'])

def strat_1(x, y):
    """
    Strategy 1: Use ETH price data to predict BTC price.
    """
    df = gn_data_1.copy()
    if df.empty:
        return 0

    df['pct_change'] = df['price'].pct_change()
    df['ma'] = df['value'].rolling(x).mean()
    df['sd'] = df['value'].rolling(x).std()
    df['z'] = (df['value'] - df['ma']) / df['sd']
    df['pos'] = np.where(df['z'] > y, 1, 0)
    print('Strategy1:')
    print(df.tail(5))
    return df['pos'].iloc[-1]


def strat_2(x, y):
    """
    Strategy 2: Predict BTC price based on USDC balance distribution.
    """
    df = gn_data_2.copy()
    if df.empty:
        return 0

    df['pct_change'] = df['price'].pct_change()
    df['min'] = df['value'].rolling(x).min()
    df['max'] = df['value'].rolling(x).max()
    df['pos'] = np.where((df['value'] - df['min']) / (df['max'] - df['min']) > y, 1, -1)
    print('Strategy2:')
    print(df.tail(5))
    return df['pos'].iloc[-1]


def strat_3(x, y):
    """
    Strategy 3: Net Unrealized Profit/Loss (NUPL), hr
    """
    df = gn_data_3.copy()
    if df.empty:
        return 0
    df['ma'] = df['value'].rolling(x).mean()
    df['sd'] = df['value'].rolling(x).std()
    df['z'] = (df['value'] - df['ma']) / df['sd']

    df['pos'] = np.where(df['z'] > y, 1, 0)
    print('Strategy3:')
    print(df.tail(5))
    return df['pos'].iloc[-1]

def calculate_position():
    """
    Calculate overall position.
    """
    global signal_data
    pos_1 = strat_1(config['STRAT1']['x'], config['STRAT1']['y'])
    pos_2 = strat_2(config['STRAT2']['x'], config['STRAT2']['y'])
    pos_3 = strat_3(config['STRAT3']['x'], config['STRAT3']['y'])
    pos = pos_1 * config['STRAT1']['ratio'] + pos_2 * config['STRAT2']['ratio'] + pos_3 * config['STRAT3']['ratio']

#     message = f"strategyA:{pos_1}\nstrategyB:{pos_2}\nstrategyC:{pos_3}"
#     base_url = 'https://api.telegram.org/bot7891141193:AAHi5XkBl6C--9ClpOxkuq5NA502OFHTLuI/sendMessage?chat_id=-4609977958&text='
#     requests.get(base_url+message)

    # Save signal to memory
    new_row = pd.DataFrame([[datetime.datetime.now(), pos]], columns=['dt', 'pos'])
    signal_data = pd.concat([signal_data, new_row], ignore_index=True)
    print('Net position change% from signals:',f"{pos * 100:.{2}f}%")
    return pos

# ===== Trading Functions =====
def current_pos():
    """
    Get current position.
    """
    position = EXCHANGE.fetch_position(SYMBOL)['info']
    if position['side'] == 'Buy':
        return float(position['size'])
    elif position['side'] == 'Sell':
        return -float(position['size'])
    return 0

def execute_trade(signal):
    """
    Execute trade based on signal.
    """
    net_pos = current_pos()
    target_pos = MAX_POS * signal
    bet_size = round(target_pos - net_pos, 3)

    print('Current Position:',net_pos)
    print('Target Position:',target_pos)
    print('Order size:',bet_size)

    # Retry 3 times when exception
    for attempt in range(3):
        try:
            if bet_size > 0:
                EXCHANGE.create_order(SYMBOL, 'market', 'buy', bet_size, None)
                print(f'Buy {bet_size} {SYMBOL} successfully!')
            elif bet_size < 0:
                EXCHANGE.create_order(SYMBOL, 'market', 'sell', abs(bet_size), None)
                print(f'Sell {bet_size} {SYMBOL} successfully!')
            print_acct_info()
            break
        except Exception as e:
            print(f"Retrying ({attempt+1}/3):", str(e))
            time.sleep(2 ** attempt)  # Wait longer with each retry

def main():
    global gn_data_1, gn_data_2, gn_data_3
    print('Start trading',SYMBOL,'.....')
    print_acct_info()

    while True:

        if datetime.datetime.now().second == 0:
        # Handle all new strategy with glassnode api resolution = 10m
            print('Time:', datetime.datetime.now())
            # print("Running strategies with 10m resolution.....")

            if datetime.datetime.now().minute == 11:
                # Handle all new strategy with glassnode api resolution = 1h
                print("Running strategies with 1h resolution.....")
                thread_1 = threading.Thread(
                    target=fetch_data,
                    args=(config['STRAT1']['api'], config['STRAT1']['api_symbol'], config['STRAT1']['resolution'], "gn_data_1"))
                thread_2 = threading.Thread(
                    target=fetch_data,
                    args=(config['STRAT2']['api'], config['STRAT2']['api_symbol'], config['STRAT1']['resolution'], "gn_data_2"))
                thread_3 = threading.Thread(
                    target=fetch_data,
                    args=(config['STRAT3']['api'], config['STRAT3']['api_symbol'], config['STRAT1']['resolution'], "gn_data_3"))

                thread_1.start()
                thread_2.start()
                thread_3.start()

                thread_1.join()
                thread_2.join()
                thread_3.join()
                execute_trade(calculate_position())  # execute trade

                if datetime.datetime.now().hour == 8:
                    # Handle all new strategy with glassnode api resolution = 24h
                    print("Running strategies with 24h resolution.....")

        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Trading Ended!")