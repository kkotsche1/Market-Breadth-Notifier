import yfinance as yf
import pandas as pd
import requests
import datetime
import os
import json
import smtplib
import argparse
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email import encoders
from io import StringIO
from pathlib import Path
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Constants
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
HISTORY_FILE = DATA_DIR / "breadth_history.csv"
REPORT_FILE = OUTPUT_DIR / "dashboard.html"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_sp500_tickers():
    """Fetches S&P 500 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        tickers = df['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

def get_nasdaq100_tickers():
    """Fetches Nasdaq 100 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        for df in tables:
            if 'Ticker' in df.columns:
                return [t.replace('.', '-') for t in df['Ticker'].tolist()]
            if 'Symbol' in df.columns:
                 return [t.replace('.', '-') for t in df['Symbol'].tolist()]
        print("Could not find Nasdaq 100 ticker table.")
        return []
    except Exception as e:
        print(f"Error fetching Nasdaq 100 tickers: {e}")
        return []

def fetch_historical_data(tickers, lookback_days=400):
    """Downloads historical data for calculating SMAs."""
    if not tickers:
        return pd.DataFrame()
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=lookback_days)
    
    print(f"Downloading data for {len(tickers)} tickers from {start_date.date()} to {end_date.date()}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=True, threads=True, auto_adjust=True)['Close']
        if data.empty:
            print("No data downloaded.")
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

def calculate_metrics_for_date(price_data, date_idx):
    """Calculates breadth metrics for a specific date index."""
    # Slicing data up to the specific date
    historical_slice = price_data.iloc[:date_idx+1]
    
    if len(historical_slice) < 200:
        return None

    current_prices = historical_slice.iloc[-1]
    
    # Calculate SMAs
    sma_50 = historical_slice.rolling(window=50).mean().iloc[-1]
    sma_100 = historical_slice.rolling(window=100).mean().iloc[-1]
    sma_200 = historical_slice.rolling(window=200).mean().iloc[-1]
    
    # Comparison
    above_50 = (current_prices > sma_50)
    above_100 = (current_prices > sma_100)
    above_200 = (current_prices > sma_200)
    
    total = len(current_prices.dropna())
    if total == 0:
        return None

    return {
        'total': total,
        'above_50': int(above_50.sum()),
        'above_100': int(above_100.sum()),
        'above_200': int(above_200.sum()),
        'pct_above_50': round(float(above_50.sum() / total) * 100, 2),
        'pct_above_100': round(float(above_100.sum() / total) * 100, 2),
        'pct_above_200': round(float(above_200.sum() / total) * 100, 2)
    }

def detect_crossovers(price_data, prev_date_idx, curr_date_idx, sma_periods=[50, 100, 200]):
    """
    Detects crossover events between two consecutive dates.
    
    Returns dict with structure:
    {
        50: {'bullish': [{'ticker': 'AAPL', 'close': 185.50, 'sma_value': 182.30}, ...], 'bearish': [...]},
        100: {...},
        200: {...}
    }
    """
    if prev_date_idx < 0 or curr_date_idx >= len(price_data):
        return {p: {'bullish': [], 'bearish': []} for p in sma_periods}
    
    results = {}
    
    for period in sma_periods:
        if len(price_data) < period:
            results[period] = {'bullish': [], 'bearish': []}
            continue
        
        # Calculate SMA for both days
        sma_prev = price_data.iloc[:prev_date_idx+1].rolling(window=period).mean().iloc[-1]
        sma_curr = price_data.iloc[:curr_date_idx+1].rolling(window=period).mean().iloc[-1]
        
        price_prev = price_data.iloc[prev_date_idx]
        price_curr = price_data.iloc[curr_date_idx]
        
        bullish = []
        bearish = []
        
        for ticker in price_data.columns:
            try:
                prev_price = price_prev[ticker]
                curr_price = price_curr[ticker]
                prev_sma = sma_prev[ticker]
                curr_sma = sma_curr[ticker]
                
                # Skip if any NaN
                if pd.isna(prev_price) or pd.isna(curr_price) or pd.isna(prev_sma) or pd.isna(curr_sma):
                    continue
                
                was_below = prev_price < prev_sma
                was_above = prev_price > prev_sma
                now_below = curr_price < curr_sma
                now_above = curr_price > curr_sma
                
                if was_below and now_above:
                    bullish.append({
                        'ticker': ticker,
                        'close': round(float(curr_price), 2),
                        'sma_value': round(float(curr_sma), 2)
                    })
                elif was_above and now_below:
                    bearish.append({
                        'ticker': ticker,
                        'close': round(float(curr_price), 2),
                        'sma_value': round(float(curr_sma), 2)
                    })
            except Exception:
                continue
        
        results[period] = {'bullish': bullish, 'bearish': bearish}
    
    return results

def save_crossover_history(crossovers_sp500, crossovers_nasdaq, date_str):
    """
    Appends crossover events to JSON file with date as key.
    """
    history_file = DATA_DIR / "crossover_history.json"
    
    # Load existing history
    if history_file.exists():
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = {}
    else:
        history = {}
    
    # Format crossovers for storage (convert int keys to strings for JSON)
    def format_crossovers(crossovers):
        return {
            str(period): data for period, data in crossovers.items()
        }
    
    # Add today's crossovers
    history[date_str] = {
        'sp500': format_crossovers(crossovers_sp500),
        'nasdaq': format_crossovers(crossovers_nasdaq)
    }
    
    # Save back
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    
    print(f"Crossover history saved to {history_file}")
    return history[date_str]

def update_history_file(new_records):
    """Appends new records to CSV, ensuring no duplicates."""
    try:
        if HISTORY_FILE.exists():
            df = pd.read_csv(HISTORY_FILE)
            # Convert date column to match format
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        else:
            df = pd.DataFrame(columns=[
                'date', 
                'sp500_total', 'sp500_above_50', 'sp500_above_100', 'sp500_above_200', 
                'sp500_pct_above_50', 'sp500_pct_above_100', 'sp500_pct_above_200',
                'nasdaq_total', 'nasdaq_above_50', 'nasdaq_above_100', 'nasdaq_above_200',
                'nasdaq_pct_above_50', 'nasdaq_pct_above_100', 'nasdaq_pct_above_200'
            ])

        # Convert new records to DataFrame
        new_df = pd.DataFrame(new_records)
        
        # Concatenate and drop duplicates based on date
        combined = pd.concat([df, new_df]).drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date')
        
        combined.to_csv(HISTORY_FILE, index=False)
        print(f"History updated. Total records: {len(combined)}")
        return combined
    except Exception as e:
        print(f"Error updating history file: {e}")
        return pd.DataFrame()

def generate_html_dashboard(history_df, sp500_current, nasdaq_current, crossovers=None):
    """Generates the HTML dashboard with history charts + todays active metrics + crossover table."""
    # Convert history for JS
    dates = history_df['date'].tolist()
    
    # S&P Data
    sp500_50 = history_df['sp500_pct_above_50'].tolist()
    sp500_100 = history_df['sp500_pct_above_100'].tolist()
    sp500_200 = history_df['sp500_pct_above_200'].tolist()
    
    # Nasdaq Data
    nasdaq_50 = history_df['nasdaq_pct_above_50'].tolist()
    nasdaq_100 = history_df['nasdaq_pct_above_100'].tolist()
    nasdaq_200 = history_df['nasdaq_pct_above_200'].tolist()

    # Calculate daily changes
    def get_change_arrow(current, previous):
        diff = current - previous
        color = "#4caf50" if diff >= 0 else "#ff5252"
        arrow = "↑" if diff >= 0 else "↓"
        return f"{arrow} {abs(diff):.1f}%", color

    # Get daily diffs
    try:
        prev_row = history_df.iloc[-2]
        
        # S&P 500 Diffs
        sp_d50 = get_change_arrow(sp500_current['pct_above_50'], prev_row['sp500_pct_above_50'])
        sp_d100 = get_change_arrow(sp500_current['pct_above_100'], prev_row['sp500_pct_above_100'])
        sp_d200 = get_change_arrow(sp500_current['pct_above_200'], prev_row['sp500_pct_above_200'])
        
        # Nasdaq Diffs
        nas_d50 = get_change_arrow(nasdaq_current['pct_above_50'], prev_row['nasdaq_pct_above_50'])
        nas_d100 = get_change_arrow(nasdaq_current['pct_above_100'], prev_row['nasdaq_pct_above_100'])
        nas_d200 = get_change_arrow(nasdaq_current['pct_above_200'], prev_row['nasdaq_pct_above_200'])

    except:
        # Fallback if no history
        dummy = ("—", "#888")
        sp_d50 = sp_d100 = sp_d200 = dummy
        nas_d50 = nas_d100 = nas_d200 = dummy

    # Prepare crossover data for JS embedding
    crossover_json = json.dumps(crossovers) if crossovers else "null"

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Breadth Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.1.0"></script>
    <style>
        :root {{
            --bg-color: #121212;
            --card-bg: #1e1e1e;
            --text-color: #e0e0e0;
            --accent-color: #3f51b5;
            --success-color: #4caf50;
            --danger-color: #ff5252;
            --text-muted: #888;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .date {{ color: var(--text-muted); font-size: 0.9em; margin-bottom: 15px; }}
        
        .controls {{ display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; }}
        .filter-btn {{
            background: #333; color: #ccc; border: 1px solid #444;
            padding: 8px 16px; border-radius: 20px; cursor: pointer;
            transition: all 0.2s; font-size: 0.9em;
        }}
        .filter-btn:hover {{ background: #444; }}
        .filter-btn.active {{ background: var(--accent-color); color: white; border-color: var(--accent-color); }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .card {{
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 1px solid #333;
        }}
        
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .chart-title {{ font-size: 1.2em; font-weight: 500; color: #fff; }}
        
        .chart-container {{ position: relative; height: 400px; width: 100%; }}
        
        /* Detailed Stats Row */
        .stats-grid {{ 
            display: grid; 
            grid-template-columns: repeat(3, 1fr); 
            gap: 15px; 
            margin-top: 25px; 
            padding-top: 20px;
            border-top: 1px solid #333;
        }}
        .stat-box {{ 
            background: #252525; 
            padding: 15px; 
            border-radius: 8px; 
            text-align: center; 
        }}
        .stat-val {{ display: block; font-size: 1.8em; font-weight: bold; line-height: 1.2; }}
        .stat-lbl {{ font-size: 0.85em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 5px; display: block; }}
        .stat-change {{ font-size: 0.9em; font-weight: 500; margin-top: 5px; display: block; }}
        
        .c-50 {{ color: #4caf50; }}   /* Green */
        .c-100 {{ color: #2196f3; }}  /* Blue */
        .c-200 {{ color: #ffc107; }}  /* Amber */
        
        /* Crossover Section Styles */
        .crossover-section {{
            margin-top: 40px;
        }}
        .section-title {{
            font-size: 1.5em;
            font-weight: 600;
            color: #fff;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .collapsible-header {{
            background: var(--card-bg);
            border: 1px solid #333;
            border-radius: 12px;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s;
            margin-bottom: 15px;
        }}
        .collapsible-header:hover {{
            background: #252525;
            border-color: #444;
        }}
        .collapsible-header .toggle-icon {{
            transition: transform 0.3s;
            font-size: 1.2em;
        }}
        .collapsible-header.open .toggle-icon {{
            transform: rotate(180deg);
        }}
        .collapsible-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        .collapsible-content.open {{
            max-height: 2000px;
        }}
        .summary-stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .summary-stat {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.95em;
        }}
        .summary-stat .count {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        .bullish {{ color: var(--success-color); }}
        .bearish {{ color: var(--danger-color); }}
        
        /* Crossover Filters */
        .crossover-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
            align-items: center;
        }}
        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .filter-label {{
            color: var(--text-muted);
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .filter-select {{
            background: #333;
            color: #ccc;
            border: 1px solid #444;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 0.9em;
            cursor: pointer;
        }}
        .filter-select:hover {{
            background: #3a3a3a;
        }}
        
        /* Crossover Table */
        .crossover-table-container {{
            overflow-x: auto;
        }}
        .crossover-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        .crossover-table th {{
            background: #252525;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.8em;
            letter-spacing: 0.5px;
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        .crossover-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #2a2a2a;
            vertical-align: middle;
        }}
        .crossover-table tr:hover td {{
            background: #1a1a1a;
        }}
        .ticker-cell {{
            font-weight: 600;
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        }}
        .direction-badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .direction-badge.bullish {{
            background: rgba(76, 175, 80, 0.15);
            color: var(--success-color);
        }}
        .direction-badge.bearish {{
            background: rgba(255, 82, 82, 0.15);
            color: var(--danger-color);
        }}
        .sma-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        .sma-badge.sma-50 {{ background: rgba(76, 175, 80, 0.15); color: #4caf50; }}
        .sma-badge.sma-100 {{ background: rgba(33, 150, 243, 0.15); color: #2196f3; }}
        .sma-badge.sma-200 {{ background: rgba(255, 193, 7, 0.15); color: #ffc107; }}
        .index-badge {{
            font-size: 0.8em;
            color: var(--text-muted);
        }}
        .price-cell {{
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 0.85em;
        }}
        .no-crossovers {{
            text-align: center;
            padding: 40px 20px;
            color: var(--text-muted);
            font-style: italic;
        }}
        
        /* Responsive Cards for Mobile */
        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
            .crossover-table {{ display: none; }}
            .crossover-cards {{ display: block; }}
        }}
        @media (min-width: 769px) {{
            .crossover-cards {{ display: none; }}
        }}
        .crossover-card {{
            background: #252525;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border: 1px solid #333;
        }}
        .crossover-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .crossover-card-body {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            font-size: 0.85em;
            color: var(--text-muted);
        }}
        
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Market Breadth Dashboard</h1>
            <div class="date">Generated: {datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}</div>
            
            <div class="controls">
                <button class="filter-btn active" data-mode="all" onclick="updateVisibility('all')">All Periods</button>
                <button class="filter-btn" data-mode="50" onclick="updateVisibility('50')">SMA 50 Only</button>
                <button class="filter-btn" data-mode="100" onclick="updateVisibility('100')">SMA 100 Only</button>
                <button class="filter-btn" data-mode="200" onclick="updateVisibility('200')">SMA 200 Only</button>
            </div>
        </div>

        <div class="grid">
            <!-- S&P 500 Card -->
            <div class="card">
                <div class="chart-header">
                    <div class="chart-title">S&P 500 Breadth</div>
                </div>
                <div class="chart-container">
                    <canvas id="sp500Chart"></canvas>
                </div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <span class="stat-val c-50">{sp500_current['pct_above_50']}%</span>
                        <span class="stat-change" style="color: {sp_d50[1]}">{sp_d50[0]}</span>
                        <span class="stat-lbl">Above SMA 50</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val c-100">{sp500_current['pct_above_100']}%</span>
                        <span class="stat-change" style="color: {sp_d100[1]}">{sp_d100[0]}</span>
                        <span class="stat-lbl">Above SMA 100</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val c-200">{sp500_current['pct_above_200']}%</span>
                        <span class="stat-change" style="color: {sp_d200[1]}">{sp_d200[0]}</span>
                        <span class="stat-lbl">Above SMA 200</span>
                    </div>
                </div>
            </div>

            <!-- Nasdaq 100 Card -->
            <div class="card">
                <div class="chart-header">
                    <div class="chart-title">Nasdaq 100 Breadth</div>
                </div>
                <div class="chart-container">
                    <canvas id="nasdaqChart"></canvas>
                </div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <span class="stat-val c-50">{nasdaq_current['pct_above_50']}%</span>
                        <span class="stat-change" style="color: {nas_d50[1]}">{nas_d50[0]}</span>
                        <span class="stat-lbl">Above SMA 50</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val c-100">{nasdaq_current['pct_above_100']}%</span>
                        <span class="stat-change" style="color: {nas_d100[1]}">{nas_d100[0]}</span>
                        <span class="stat-lbl">Above SMA 100</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-val c-200">{nasdaq_current['pct_above_200']}%</span>
                        <span class="stat-change" style="color: {nas_d200[1]}">{nas_d200[0]}</span>
                        <span class="stat-lbl">Above SMA 200</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Crossover Section -->
        <div class="crossover-section" id="crossoverSection">
            <div class="collapsible-header" onclick="toggleCrossovers()">
                <div>
                    <span style="font-size: 1.1em; font-weight: 500;">📊 Today's SMA Crossovers</span>
                    <div class="summary-stats" id="crossoverSummary">
                        <!-- Filled by JS -->
                    </div>
                </div>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="collapsible-content" id="crossoverContent">
                <div class="card">
                    <div class="crossover-controls">
                        <div class="filter-group">
                            <span class="filter-label">Index</span>
                            <select class="filter-select" id="indexFilter" onchange="filterCrossovers()">
                                <option value="all">All</option>
                                <option value="sp500">S&P 500</option>
                                <option value="nasdaq">Nasdaq 100</option>
                            </select>
                        </div>
                        <div class="filter-group">
                            <span class="filter-label">SMA</span>
                            <select class="filter-select" id="smaFilter" onchange="filterCrossovers()">
                                <option value="all">All</option>
                                <option value="50">SMA 50</option>
                                <option value="100">SMA 100</option>
                                <option value="200">SMA 200</option>
                            </select>
                        </div>
                        <div class="filter-group">
                            <span class="filter-label">Direction</span>
                            <select class="filter-select" id="directionFilter" onchange="filterCrossovers()">
                                <option value="all">All</option>
                                <option value="bullish">↑ Bullish</option>
                                <option value="bearish">↓ Bearish</option>
                            </select>
                        </div>
                        <div class="filter-group" style="margin-left: auto;">
                            <span id="filteredCount" style="color: var(--text-muted); font-size: 0.9em;"></span>
                        </div>
                    </div>
                    
                    <div class="crossover-table-container">
                        <table class="crossover-table" id="crossoverTable">
                            <thead>
                                <tr>
                                    <th>Ticker</th>
                                    <th>Direction</th>
                                    <th>SMA</th>
                                    <th>Index</th>
                                    <th>Close</th>
                                    <th>SMA Value</th>
                                </tr>
                            </thead>
                            <tbody id="crossoverTableBody">
                                <!-- Filled by JS -->
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="crossover-cards" id="crossoverCards">
                        <!-- Filled by JS for mobile -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const commonOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            scales: {{
                y: {{
                    min: 0,
                    max: 100,
                    grid: {{ color: '#333' }},
                    ticks: {{ color: '#888' }}
                }},
                x: {{
                    grid: {{ display: false }},
                    ticks: {{ color: '#888' }}
                }}
            }},
            plugins: {{
                annotation: {{
                    annotations: {{
                        line1: {{
                            type: 'line', yMin: 80, yMax: 80, borderColor: 'rgba(255, 82, 82, 0.3)', borderWidth: 1, borderDash: [5, 5]
                        }},
                        line2: {{
                            type: 'line', yMin: 20, yMax: 20, borderColor: 'rgba(76, 175, 80, 0.3)', borderWidth: 1, borderDash: [5, 5]
                        }}
                    }}
                }},
                legend: {{
                    labels: {{ color: '#ccc', font: {{ size: 12 }} }}
                }}
            }}
        }};

        const dates = {json.dumps(dates)};

        // S&P 500 Chart
        const spChart = new Chart(document.getElementById('sp500Chart'), {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: 'SMA 50',
                        data: {json.dumps(sp500_50)},
                        borderColor: '#4caf50',
                        backgroundColor: '#4caf50',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1
                    }},
                    {{
                        label: 'SMA 100',
                        data: {json.dumps(sp500_100)},
                        borderColor: '#2196f3',
                        backgroundColor: '#2196f3',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1,
                        hidden: false
                    }},
                    {{
                        label: 'SMA 200',
                        data: {json.dumps(sp500_200)},
                        borderColor: '#ffc107',
                        backgroundColor: '#ffc107',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1
                    }}
                ]
            }},
            options: commonOptions
        }});

        // Nasdaq Chart
        const nasChart = new Chart(document.getElementById('nasdaqChart'), {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: 'SMA 50',
                        data: {json.dumps(nasdaq_50)},
                        borderColor: '#4caf50',
                        backgroundColor: '#4caf50',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1
                    }},
                    {{
                        label: 'SMA 100',
                        data: {json.dumps(nasdaq_100)},
                        borderColor: '#2196f3',
                        backgroundColor: '#2196f3',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1,
                        hidden: false
                    }},
                    {{
                        label: 'SMA 200',
                        data: {json.dumps(nasdaq_200)},
                        borderColor: '#ffc107',
                        backgroundColor: '#ffc107',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1
                    }}
                ]
            }},
            options: commonOptions
        }});

        // Function to toggle datasets visibility via Title Click or Buttons (if added)
        // Currently legend clicking is native to Chart.js and works perfect.
        // User asked for UI element to select "All" or "Single".
        
        function updateVisibility(mode) {{
            [spChart, nasChart].forEach(chart => {{
                chart.data.datasets.forEach((ds, i) => {{
                    if (mode === 'all') ds.hidden = false;
                    else if (mode === '50') ds.hidden = (i !== 0);
                    else if (mode === '100') ds.hidden = (i !== 1);
                    else if (mode === '200') ds.hidden = (i !== 2);
                }});
                chart.update();
            }});
            
            // Update button styles
            document.querySelectorAll('.filter-btn').forEach(btn => {{
                btn.classList.remove('active');
                if(btn.dataset.mode === mode) btn.classList.add('active');
            }});
        }}
        
        // ========== Crossover Section ==========
        const crossoverData = {crossover_json};
        let allCrossovers = [];
        
        // Flatten crossover data for filtering
        function initCrossovers() {{
            if (!crossoverData) {{
                document.getElementById('crossoverSection').style.display = 'none';
                return;
            }}
            
            allCrossovers = [];
            
            ['sp500', 'nasdaq'].forEach(index => {{
                if (!crossoverData[index]) return;
                ['50', '100', '200'].forEach(sma => {{
                    if (!crossoverData[index][sma]) return;
                    ['bullish', 'bearish'].forEach(direction => {{
                        const items = crossoverData[index][sma][direction] || [];
                        items.forEach(item => {{
                            allCrossovers.push({{
                                ticker: item.ticker,
                                close: item.close,
                                smaValue: item.sma_value,
                                direction: direction,
                                sma: sma,
                                index: index
                            }});
                        }});
                    }});
                }});
            }});
            
            if (allCrossovers.length === 0) {{
                document.getElementById('crossoverSection').style.display = 'none';
                return;
            }}
            
            updateSummary();
            filterCrossovers();
        }}
        
        function updateSummary() {{
            const bullishCount = allCrossovers.filter(c => c.direction === 'bullish').length;
            const bearishCount = allCrossovers.filter(c => c.direction === 'bearish').length;
            
            document.getElementById('crossoverSummary').innerHTML = `
                <div class="summary-stat">
                    <span class="count bullish">${{bullishCount}}</span>
                    <span>Bullish ↑</span>
                </div>
                <div class="summary-stat">
                    <span class="count bearish">${{bearishCount}}</span>
                    <span>Bearish ↓</span>
                </div>
            `;
        }}
        
        function toggleCrossovers() {{
            const header = document.querySelector('.collapsible-header');
            const content = document.getElementById('crossoverContent');
            header.classList.toggle('open');
            content.classList.toggle('open');
        }}
        
        function filterCrossovers() {{
            const indexFilter = document.getElementById('indexFilter').value;
            const smaFilter = document.getElementById('smaFilter').value;
            const directionFilter = document.getElementById('directionFilter').value;
            
            let filtered = allCrossovers;
            
            if (indexFilter !== 'all') {{
                filtered = filtered.filter(c => c.index === indexFilter);
            }}
            if (smaFilter !== 'all') {{
                filtered = filtered.filter(c => c.sma === smaFilter);
            }}
            if (directionFilter !== 'all') {{
                filtered = filtered.filter(c => c.direction === directionFilter);
            }}
            
            // Sort: bullish first, then by ticker
            filtered.sort((a, b) => {{
                if (a.direction !== b.direction) return a.direction === 'bullish' ? -1 : 1;
                return a.ticker.localeCompare(b.ticker);
            }});
            
            renderTable(filtered);
            renderCards(filtered);
            
            document.getElementById('filteredCount').textContent = 
                `Showing ${{filtered.length}} of ${{allCrossovers.length}}`;
        }}
        
        function renderTable(data) {{
            const tbody = document.getElementById('crossoverTableBody');
            
            if (data.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="6" class="no-crossovers">No crossovers match your filters</td></tr>';
                return;
            }}
            
            tbody.innerHTML = data.map(c => `
                <tr data-index="${{c.index}}" data-sma="${{c.sma}}" data-direction="${{c.direction}}">
                    <td class="ticker-cell">${{c.ticker}}</td>
                    <td><span class="direction-badge ${{c.direction}}">${{c.direction === 'bullish' ? '↑' : '↓'}} ${{c.direction}}</span></td>
                    <td><span class="sma-badge sma-${{c.sma}}">SMA ${{c.sma}}</span></td>
                    <td class="index-badge">${{c.index === 'sp500' ? 'S&P 500' : 'Nasdaq 100'}}</td>
                    <td class="price-cell">${{c.close.toFixed(2)}}</td>
                    <td class="price-cell">${{c.smaValue.toFixed(2)}}</td>
                </tr>
            `).join('');
        }}
        
        function renderCards(data) {{
            const container = document.getElementById('crossoverCards');
            
            if (data.length === 0) {{
                container.innerHTML = '<div class="no-crossovers">No crossovers match your filters</div>';
                return;
            }}
            
            container.innerHTML = data.map(c => `
                <div class="crossover-card">
                    <div class="crossover-card-header">
                        <span class="ticker-cell">${{c.ticker}}</span>
                        <span class="direction-badge ${{c.direction}}">${{c.direction === 'bullish' ? '↑' : '↓'}} ${{c.direction}}</span>
                    </div>
                    <div class="crossover-card-body">
                        <span class="sma-badge sma-${{c.sma}}">SMA ${{c.sma}}</span>
                        <span>${{c.index === 'sp500' ? 'S&P 500' : 'Nasdaq 100'}}</span>
                        <span>Close: ${{c.close.toFixed(2)}}</span>
                        <span>SMA: ${{c.smaValue.toFixed(2)}}</span>
                    </div>
                </div>
            `).join('');
        }}
        
        // Initialize on page load
        initCrossovers();
    </script>
</body>
</html>
    """
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Dashboard generated at {REPORT_FILE}")
    return html_content

def send_email(html_path, history_df):
    """Sends email with attachment AND inline summary."""
    
    recipients = [
        email for email in [
            os.getenv("RECIPIENT_EMAIL_1"), 
            os.getenv("RECIPIENT_EMAIL_2"),
        ] 
        if email
    ]
    if not recipients:
        print("No recipients defined.")
        return

    sender_email = os.getenv("GMAIL_SENDING_EMAIL")
    sender_password = os.getenv("GMAIL_APP_PASSWORD")
    
    if not sender_email or not sender_password:
        print("Gmail credentials missing.")
        return

    # Prepare summary for email body
    try:
        latest = history_df.iloc[-1]
        prev = history_df.iloc[-2] if len(history_df) > 1 else latest
        
        sp_diff = latest['sp500_pct_above_50'] - prev['sp500_pct_above_50']
        nas_diff = latest['nasdaq_pct_above_50'] - prev['nasdaq_pct_above_50']
        
        sp_arrow = "↑" if sp_diff >= 0 else "↓"
        nas_arrow = "↑" if nas_diff >= 0 else "↓"
        
        body_summary = (
            f"Market Breadth Update ({latest['date']})\n\n"
            f"S&P 500 (>SMA50): {latest['sp500_pct_above_50']}% ({sp_arrow}{abs(sp_diff):.1f}%)\n"
            f"Nasdaq 100 (>SMA50): {latest['nasdaq_pct_above_50']}% ({nas_arrow}{abs(nas_diff):.1f}%)\n\n"
            "Full details and long-term charts in the attached dashboard."
        )
    except:
        body_summary = "Please find attached the Market Breadth Dashboard."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['Subject'] = f"Market Breadth: SPX {latest['sp500_pct_above_50']}% | NDX {latest['nasdaq_pct_above_50']}%"
    msg.attach(MIMEText(body_summary, 'plain'))

    # Attach HTML
    with open(html_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename=market_breadth_dashboard.html")
        msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        
        for recipient in recipients:
            try:
                if 'To' in msg:
                    del msg['To']
                msg['To'] = recipient
                server.sendmail(sender_email, recipient, msg.as_string())
                print(f"Sent to {recipient}")
            except Exception as e:
                print(f"Failed to send to {recipient}: {e}")
            
        server.quit()
    except Exception as e:
        print(f"Failed to send email: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true", help="Backfill specific months of history")
    parser.add_argument("--months", type=int, default=12, help="Months to backfill")
    parser.add_argument("--no-email", action="store_true", help="Skip sending email")
    args = parser.parse_args()

    print("=== Starting Market Breadth Analysis ===")
    
    # Check if history exists
    if not HISTORY_FILE.exists() or args.backfill:
        print(f"History missing or backfill requested. Starting {args.months} month backfill...")
        
        sp500_tickers = get_sp500_tickers()
        nasdaq_tickers = get_nasdaq100_tickers()
        
        # Determine start date
        start_date = datetime.datetime.now() - relativedelta(months=args.months)
        days_back = (datetime.datetime.now() - start_date).days + 50 # Buffer
        
        print("Fetching bulk historical data...")
        sp500_data = fetch_historical_data(sp500_tickers, lookback_days=days_back)
        nasdaq_data = fetch_historical_data(nasdaq_tickers, lookback_days=days_back)
        
        records = []
        # Iterate through business days
        dates = pd.date_range(start=start_date, end=datetime.datetime.now(), freq='B')
        
        for d in dates:
            date_str = d.strftime('%Y-%m-%d')
            # Find closest index in data (handling potential data gaps)
            try:
                # Find index location for this date (or closest previous)
                sp_idx = sp500_data.index.get_indexer([d], method='nearest')[0]
                nas_idx = nasdaq_data.index.get_indexer([d], method='nearest')[0]
                
                # Verify date is actually close (within 3 days) to handle large gaps
                if abs((sp500_data.index[sp_idx] - d).days) > 3: continue

                sp_metrics = calculate_metrics_for_date(sp500_data, sp_idx)
                nas_metrics = calculate_metrics_for_date(nasdaq_data, nas_idx)
                
                if sp_metrics and nas_metrics:
                    record = {'date': date_str}
                    # Prefix keys
                    for k, v in sp_metrics.items(): record[f'sp500_{k}'] = v
                    for k, v in nas_metrics.items(): record[f'nasdaq_{k}'] = v
                    records.append(record)
                    
            except Exception as e:
                # print(f"Skipping {date_str}: {e}")
                continue
                
        if records:
            history_df = update_history_file(records)
            print("Backfill complete.")
            
    else:
        # Standard Daily Run
        print("Running daily update...")
        history_df = pd.read_csv(HISTORY_FILE)
        
        sp500_tickers = get_sp500_tickers()
        nasdaq_tickers = get_nasdaq100_tickers()
        
        # We need enough lookback for 200 SMA
        sp500_data = fetch_historical_data(sp500_tickers, lookback_days=400)
        nasdaq_data = fetch_historical_data(nasdaq_tickers, lookback_days=400)
        
        today = datetime.datetime.now()
        date_str = today.strftime('%Y-%m-%d')
        
        sp_metrics = calculate_metrics_for_date(sp500_data, len(sp500_data)-1)
        nas_metrics = calculate_metrics_for_date(nasdaq_data, len(nasdaq_data)-1)
        
        # Detect crossovers (compare yesterday to today)
        crossovers_sp500 = None
        crossovers_nasdaq = None
        if len(sp500_data) >= 2 and len(nasdaq_data) >= 2:
            print("Detecting crossovers...")
            crossovers_sp500 = detect_crossovers(sp500_data, len(sp500_data)-2, len(sp500_data)-1)
            crossovers_nasdaq = detect_crossovers(nasdaq_data, len(nasdaq_data)-2, len(nasdaq_data)-1)
            
            # Save to JSON history
            save_crossover_history(crossovers_sp500, crossovers_nasdaq, date_str)
            
            # Count and log
            total_bullish = sum(len(crossovers_sp500[p]['bullish']) + len(crossovers_nasdaq[p]['bullish']) for p in [50, 100, 200])
            total_bearish = sum(len(crossovers_sp500[p]['bearish']) + len(crossovers_nasdaq[p]['bearish']) for p in [50, 100, 200])
            print(f"Found {total_bullish} bullish and {total_bearish} bearish crossovers")
        
        if sp_metrics and nas_metrics:
            record = {'date': date_str}
            for k, v in sp_metrics.items(): record[f'sp500_{k}'] = v
            for k, v in nas_metrics.items(): record[f'nasdaq_{k}'] = v
            
            history_df = update_history_file([record])
        else:
            print("Failed to calculate today's metrics")

    # Generate Output
    crossovers_for_dashboard = None
    if not history_df.empty:
        # Get latest records for potential "Diff" calculation
        current_sp = {k.replace('sp500_',''):v for k,v in history_df.iloc[-1].items() if 'sp500' in k}
        current_nas = {k.replace('nasdaq_',''):v for k,v in history_df.iloc[-1].items() if 'nasdaq' in k}
        
        # Prepare crossover data for dashboard (only for daily runs, not backfill)
        if 'crossovers_sp500' in dir() and crossovers_sp500 and crossovers_nasdaq:
            crossovers_for_dashboard = {
                'sp500': {str(k): v for k, v in crossovers_sp500.items()},
                'nasdaq': {str(k): v for k, v in crossovers_nasdaq.items()}
            }
        
        html = generate_html_dashboard(history_df, current_sp, current_nas, crossovers_for_dashboard)
        
        if not args.no_email:
            send_email(REPORT_FILE, history_df)
    
    print("=== Done ===")

if __name__ == "__main__":
    main()
