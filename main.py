
import datetime
import time
import logging
import os
import csv
import pandas as pd
from collections import defaultdict, deque
import pytz
import pyotp
from NorenRestApiPy.NorenApi import NorenApi
import subprocess

# ======================
# Logging
# ======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================
# Custom API class
# ======================
class ShoonyaApiPy(NorenApi):
    def __init__(self):
        super().__init__(
            host='https://api.shoonya.com/NorenWClientTP/',
            websocket='wss://api.shoonya.com/NorenWSTP/'
        )
        global api
        api = self

# ======================
# API credentials (from environment variables)
# ======================
USERID = os.getenv("SHOONYA_USERID", "FA320445")
PASSWORD = os.getenv("SHOONYA_PASSWORD", "Aadvay@2208")
VENDOR_CODE = os.getenv("SHOONYA_VENDOR_CODE", "FA320445_U")
API_SECRET = os.getenv("SHOONYA_API_SECRET", "774eb4227db5141c91d2d7ed165c9afd")
TOTP_SECRET = os.getenv("SHOONYA_TOTP_SECRET", "TET73I2ON4E652Q664267KKWDW43623S")

# Initialize API
api = ShoonyaApiPy()

# TOTP Login
totp = pyotp.TOTP(TOTP_SECRET)
try:
    ret = api.login(
        userid=USERID,
        password=PASSWORD,
        twoFA=totp.now(),
        vendor_code=VENDOR_CODE,
        api_secret=API_SECRET,
        imei="github_actions"
    )
    logging.info("Login response: %s", ret)
    if not ret or 'susertoken' not in ret:
        raise Exception("Login failed: Invalid response")
except Exception as e:
    logging.error(f"Login failed: {e}")
    exit(1)

# ======================
# Config
# ======================
OUTPUT_DIR = "output"
SYMBOLS_FILE = "NFO_symbols.txt"
NIFTY_TOKEN = "26000"  # Nifty 50 token
IST = pytz.timezone('Asia/Kolkata')
TODAY_DATE = datetime.datetime.now(IST).strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"live_options_data1MIN_{TODAY_DATE}.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# Columns
# ======================
EXPECTED_COLUMNS = ['time']
strike_labels = [
    'Call ITM-2', 'Call ITM-1', 'Call ATM', 'Call OTM-1', 'Call OTM-2', 'Call OTM-3', 'Call OTM-4',
    'Put ITM-2', 'Put ITM-1', 'Put ATM', 'Put OTM-1', 'Put OTM-2', 'Put OTM-3', 'Put OTM-4'
]
for label in strike_labels:
    EXPECTED_COLUMNS.extend([
        f"{label}_symbol", f"{label}_token", f"{label}_option_type", f"{label}_close",
        f"{label}_oi", f"{label}_tbq", f"{label}_tsq", f"{label}_delta"
    ])

# ======================
# Helpers
# ======================
def date_to_epoch(date_str, time_str="09:15:00"):
    dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    dt = IST.localize(dt)
    return int(dt.timestamp())

def parse_api_date(date_str):
    if isinstance(date_str, (datetime.datetime, pd.Timestamp)):
        return date_str.astimezone(IST) if date_str.tzinfo else IST.localize(date_str)
    try:
        dt = datetime.datetime.strptime(date_str, "%d-%m-%Y %H:%M:%S")
        return IST.localize(dt)
    except ValueError:
        logging.warning(f"Failed to parse date: {date_str}")
        return None

def fetch_ltp_at_time(exchange, token, target_date, time_str="09:16:00"):
    target_epoch = date_to_epoch(target_date, time_str)
    start_epoch = target_epoch - 900  # ±15 minutes
    end_epoch = target_epoch + 900
    try:
        time_series = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=start_epoch,
            endtime=end_epoch,
            interval=1
        )
        logging.info(f"Time series response for token {token}: {time_series}")
        if time_series is None or not time_series:
            logging.warning(f"No data returned for token {token}. Attempting real-time quote.")
            quote = api.get_quotes(exchange=exchange, token=token)
            logging.info(f"Real-time quote response: {quote}")
            if quote and quote.get('stat') == 'Ok' and 'lp' in quote:
                return float(quote['lp'])
            else:
                logging.error(f"Real-time quote failed: {quote}")
                return None
        closest_data, min_diff = None, float('inf')
        for data in time_series:
            api_time = parse_api_date(data['time'])
            if api_time:
                ts = int(api_time.timestamp())
                diff = abs(ts - target_epoch)
                if diff < min_diff:
                    min_diff = diff
                    closest_data = data
        return float(closest_data['intc']) if closest_data else None
    except Exception as e:
        logging.error(f"Error fetching LTP for token {token}: {e}")
        try:
            quote = api.get_quotes(exchange=exchange, token=token)
            logging.info(f"Real-time quote response: {quote}")
            if quote and quote.get('stat') == 'Ok' and 'lp' in quote:
                return float(quote['lp'])
            else:
                logging.error(f"Real-time quote failed: {quote}")
                return None
        except Exception as e:
            logging.error(f"Real-time quote failed: {e}")
            return None

def load_symbols_from_file(file_path):
    symbols_dict = {}
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                symbol = row.get('TradingSymbol', '').strip()
                token = row.get('Token', '').strip()
                expiry_str = row.get('Expiry', '').strip()
                instrument = row.get('Instrument', '').strip()
                opt_type = row.get('OptionType', '').strip()
                strike_str = row.get('StrikePrice', '').strip()
                strike = float(strike_str) if strike_str else None
                if symbol and token and expiry_str:
                    try:
                        expiry_dt = datetime.datetime.strptime(expiry_str, "%d-%b-%Y").date()
                        symbols_dict[symbol] = (token, expiry_dt, instrument, opt_type, strike)
                    except ValueError:
                        continue
        logging.info(f"Loaded {len(symbols_dict)} symbols from {file_path}")
        return symbols_dict
    except Exception as e:
        logging.error(f"Error loading symbols from {file_path}: {e}")
        return {}

def get_nearest_expiry(symbols_dict, target_date):
    target_dt = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    expiries = set()
    for symbol, (token, expiry_dt, instrument, opt_type, strike) in symbols_dict.items():
        if symbol.startswith('NIFTY') and instrument == 'OPTIDX' and expiry_dt >= target_dt and strike is not None:
            expiries.add(expiry_dt)
    return min(expiries) if expiries else None

def find_options_by_type(symbols_dict, expiry_date, strikes, option_type):
    selected_options = []
    for strk in strikes:
        found_option = False
        for symbol, (token, exp_dt, instr, sym_opt_type, sym_strike) in symbols_dict.items():
            if (instr == 'OPTIDX' and exp_dt == expiry_date and sym_opt_type == option_type and sym_strike == strk):
                selected_options.append((symbol, token, option_type, strk))
                found_option = True
                break
        if not found_option:
            logging.warning(f"Could not find {option_type} option for strike {strk} on expiry {expiry_date}")
            selected_options.append((None, None, option_type, strk))
    return selected_options

def fetch_time_series_data(exchange, token, target_date):
    if token is None:
        return []
    start_epoch = date_to_epoch(target_date, "09:15:00")
    end_time = datetime.datetime.now(IST)
    end_time = min(end_time, IST.localize(datetime.datetime.strptime(f"{target_date} 15:30:00", "%Y-%m-%d %H:%M:%S")))
    end_epoch = int(end_time.timestamp())
    try:
        time_series = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=start_epoch,
            endtime=end_epoch,
            interval=1
        )
        logging.info(f"Time series data for token {token}: {time_series[:5] if time_series else []}")
        valid_data = [entry for entry in time_series if parse_api_date(entry.get('time')) is not None]
        return valid_data
    except Exception as e:
        logging.error(f"Error fetching time series for token {token}: {e}")
        return []

def process_data(all_data, selected_options, is_live=False):
    if not all_data:
        logging.warning("No data to process")
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    if df.empty:
        logging.warning("Processed data frame is empty")
        return pd.DataFrame()
    # Handle timestamps differently for live vs historical data
    if is_live:
        df['time'] = df['time'].apply(lambda x: x.astimezone(IST) if isinstance(x, (datetime.datetime, pd.Timestamp)) and pd.notnull(x) else None)
    else:
        df['time'] = df['time'].apply(parse_api_date)
    df = df.dropna(subset=['time'])
    df['close'] = pd.to_numeric(df['intc'], errors='coerce')
    df['oi'] = pd.to_numeric(df['oi'], errors='coerce')
    df['tbq'] = pd.to_numeric(df.get('tbq', None), errors='coerce')
    df['tsq'] = pd.to_numeric(df.get('tsq', None), errors='coerce')
    df['delta'] = df['tbq'] - df['tsq'] if is_live else None
    pivot_data = []
    for time_val, group in df.groupby('time'):
        row = {'time': time_val}
        for label, (symbol, token, opt_type, strike) in zip(strike_labels, selected_options):
            if token is not None:
                data = group[group['token'] == token]
            else:
                data = pd.DataFrame()
            if not data.empty:
                row[f"{label}_symbol"] = symbol
                row[f"{label}_token"] = token
                row[f"{label}_option_type"] = opt_type
                row[f"{label}_close"] = data['close'].iloc[-1]
                row[f"{label}_oi"] = data['oi'].iloc[-1]
                row[f"{label}_tbq"] = data['tbq'].iloc[-1] if is_live else None
                row[f"{label}_tsq"] = data['tsq'].iloc[-1] if is_live else None
                row[f"{label}_delta"] = data['delta'].iloc[-1] if is_live else None
            else:
                row[f"{label}_symbol"] = symbol
                row[f"{label}_token"] = token
                row[f"{label}_option_type"] = opt_type
                row[f"{label}_close"] = None
                row[f"{label}_oi"] = None
                row[f"{label}_tbq"] = None
                row[f"{label}_tsq"] = None
                row[f"{label}_delta"] = None
        pivot_data.append(row)
    pivot_df = pd.DataFrame(pivot_data)
    if not pivot_df.empty:
        pivot_df = pivot_df[EXPECTED_COLUMNS]
        logging.info(f"Processed {len(pivot_df)} rows of {'live' if is_live else 'historical'} data")
    return pivot_df

def append_to_csv(df, output_file):
    if df.empty:
        logging.info("No new data to append to CSV")
        return
    try:
        df['time'] = pd.to_datetime(df['time']).apply(lambda x: x.astimezone(IST) if pd.notnull(x) else x)
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file, parse_dates=['time'])
            if list(existing_df.columns) != EXPECTED_COLUMNS:
                logging.warning("CSV format mismatch, overwriting")
                df.to_csv(output_file, index=False)
                logging.info(f"Overwrote CSV with {len(df)} rows")
                return
            existing_df['time'] = pd.to_datetime(existing_df['time']).apply(lambda x: x.astimezone(IST) if pd.notnull(x) else x)
            new_times = set(df['time'])
            existing_times = set(existing_df['time'])
            new_rows = df[~df['time'].isin(existing_times)]
            if not new_rows.empty:
                new_rows.to_csv(output_file, mode='a', header=False, index=False)
                logging.info(f"Appended {len(new_rows)} new rows to {output_file}")
            else:
                logging.info("No new rows to append")
        else:
            df.to_csv(output_file, index=False)
            logging.info(f"Created new CSV with {len(df)} rows")
    except Exception as e:
        logging.error(f"Error updating CSV: {e}")

def commit_and_push_csv(output_file):
    try:
        subprocess.run(["git", "config", "--global", "user.email", "actions@github.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "GitHub Actions"], check=True)
        subprocess.run(["git", "add", output_file], check=True)
        subprocess.run(["git", "commit", "-m", f"Update {output_file} with new data"], check=True)
        subprocess.run(["git", "push"], check=True)
        logging.info(f"Committed and pushed {output_file} to repository")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error committing/pushing CSV: {e}")

def fetch_historical_data(selected_options, target_date):
    all_historical_data = []
    print("📊 Fetching historical data...")
    for symbol, token, option_type, strike in selected_options:
        data = fetch_time_series_data("NFO", token, target_date)
        for entry in data:
            all_historical_data.append({
                'time': entry.get('time'),
                'intc': entry.get('intc'),
                'oi': entry.get('oi'),
                'tbq': None,
                'tsq': None,
                'symbol': symbol,
                'token': token,
                'option_type': option_type,
                'strike': strike
            })
    logging.info(f"Fetched {len(all_historical_data)} historical data points")
    return all_historical_data

# ======================
# Live Data Aggregation
# ======================
live_buffers = defaultdict(lambda: deque(maxlen=50))

def round_to_nearest_minute(dt):
    dt = dt.astimezone(IST)
    discard = datetime.timedelta(seconds=dt.second, microseconds=dt.microsecond)
    return dt - discard

def aggregate_live_data(buffers, selected_options):
    bars = []
    now = datetime.datetime.now(IST)
    bar_time = round_to_nearest_minute(now)
    for label, (symbol, token, opt_type, strike) in zip(strike_labels, selected_options):
        if token is None:
            continue
        quotes = list(buffers[token])
        if not quotes:
            continue
        last_tick = quotes[-1]
        bars.append({
            'time': bar_time,
            'symbol': symbol,
            'token': token,
            'option_type': opt_type,
            'strike': strike,
            'intc': last_tick['intc'],
            'oi': last_tick['oi'],
            'tbq': last_tick['tbq'],
            'tsq': last_tick['tsq'],
            'delta': last_tick['tbq'] - last_tick['tsq']
        })
        buffers[token].clear()
    return bars

def fetch_live_data_buffered(selected_options, target_date):
    last_bar_time = None
    while True:
        now = datetime.datetime.now(IST)
        if now.strftime("%H:%M") >= "15:30":
            print("🛑 Market closed")
            commit_and_push_csv(OUTPUT_FILE)
            break
        for symbol, token, option_type, strike in selected_options:
            if token is None:
                continue
            try:
                quote = api.get_quotes(exchange="NFO", token=token)
                if quote and quote.get('stat') == 'Ok':
                    tick = {
                        'time': datetime.datetime.now(IST),
                        'intc': float(quote.get('lp', 0)),
                        'oi': int(quote.get('oi', 0)),
                        'tbq': int(quote.get('tbq', 0)),
                        'tsq': int(quote.get('tsq', 0)),
                    }
                    live_buffers[token].append(tick)
            except Exception as e:
                logging.error(f"Error fetching quote for token {token}: {e}")
        current_bar_time = round_to_nearest_minute(datetime.datetime.now(IST))
        if last_bar_time is None or current_bar_time > last_bar_time:
            bars = aggregate_live_data(live_buffers, selected_options)
            if bars:
                bar_df = process_data(bars, selected_options, is_live=True)
                append_to_csv(bar_df, OUTPUT_FILE)
                commit_and_push_csv(OUTPUT_FILE)
                print(f"✅ Appended 1-min bar at {current_bar_time}")
            last_bar_time = current_bar_time
        time.sleep(5)

# ======================
# Main
# ======================
def main():
    if not ret:
        logging.error("Login failed")
        return
    try:
        if not os.path.exists(SYMBOLS_FILE):
            logging.error(f"Symbols file not found at {SYMBOLS_FILE}")
            return
        symbols_dict = load_symbols_from_file(SYMBOLS_FILE)
        if not symbols_dict:
            logging.error("No symbols loaded from file")
            return
    except Exception as e:
        logging.error(f"Failed to load symbols file: {e}")
        return
    expiry_date = get_nearest_expiry(symbols_dict, TODAY_DATE)
    if not expiry_date:
        logging.error("No expiry found")
        return
    nifty_ltp = fetch_ltp_at_time("NSE", NIFTY_TOKEN, TODAY_DATE, "09:16:00")
    if not nifty_ltp:
        logging.warning("Failed to fetch NIFTY LTP, using default ATM strike: 24700")
        atm_strike = 24700
    else:
        step = 50
        atm_strike = round(nifty_ltp / step) * step
    call_strikes = [atm_strike - 2 * step, atm_strike - step, atm_strike,
                    atm_strike + step, atm_strike + 2 * step, atm_strike + 3 * step, atm_strike + 4 * step]
    put_strikes = [atm_strike + 2 * step, atm_strike + step, atm_strike,
                   atm_strike - step, atm_strike - 2 * step, atm_strike - 3 * step, atm_strike - 4 * step]
    call_options = find_options_by_type(symbols_dict, expiry_date, call_strikes, 'CE')
    put_options = find_options_by_type(symbols_dict, expiry_date, put_strikes, 'PE')
    selected_options = call_options + put_options
    # Step 1: Historical data
    try:
        hist_data = fetch_historical_data(selected_options, TODAY_DATE)
        hist_df = process_data(hist_data, selected_options, is_live=False)
        append_to_csv(hist_df, OUTPUT_FILE)
        commit_and_push_csv(OUTPUT_FILE)
        print(f"✅ Historical data saved to {OUTPUT_FILE}")
        if not hist_df.empty:
            logging.info(f"First 5 rows of historical data:\n{hist_df.head().to_string()}")
        else:
            logging.warning("Historical DataFrame is empty")
    except Exception as e:
        logging.error(f"Error processing historical data: {e}")
    # Step 2: Check if live data should be fetched
    current_dt = datetime.datetime.now(IST).date()
    current_time = datetime.datetime.now(IST).strftime("%H:%M")
    target_dt = datetime.datetime.strptime(TODAY_DATE, "%Y-%m-%d").date()
    if target_dt == current_dt and current_time < "15:30" and current_time >= "09:15":
        print("🔄 Starting live loop with 1-min aggregation...")
        try:
            fetch_live_data_buffered(selected_options, TODAY_DATE)
        except Exception as e:
            logging.error(f"Error in live data loop: {e}")
            commit_and_push_csv(OUTPUT_FILE)
    else:
        print("🛑 Skipping live loop: Either target date is in the past or market is closed")
        commit_and_push_csv(OUTPUT_FILE)

if __name__ == "__main__":
    main()
