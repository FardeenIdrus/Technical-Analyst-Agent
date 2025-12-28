"""
Data Collection Module
Downloads historical OHLCV data for backtesting
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class DataCollector:
    """
    Collects historical price data (OHLCV - Open, High, Low, Close, Volume)
    from Yahoo Finance for backtesting technical strategies.
    """
    
    def __init__(self, data_dir="data/raw"):
        """
        Initialize data collector.
        
        Args:
            data_dir: Directory to save downloaded data
        """
        # Create Path object for the data directory
        self.data_dir = Path(data_dir)
        # Create directory if it doesn't exist (parents=True creates parent dirs too)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_data(self, ticker, years=10, interval='1d'):
        """
        Download historical OHLCV data.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL')
            years: How many years of historical data to download
            interval: Data frequency ('1d' = daily, '1h' = hourly, '1wk' = weekly)
            
        Returns:
            DataFrame with OHLCV data
        """
        
        print(f"\n Downloading {years} years of data for {ticker}...")
        
        # Calculate start date by subtracting years from today
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        try:
            # Download data from Yahoo Finance API
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False  # Don't show download progress bar
            )
            
            # Check if download returned any data
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Yahoo Finance sometimes returns columns as MultiIndex (nested headers)
            # Flatten to single level: ['Open', 'High', 'Low', 'Close', 'Volume']
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Verify all required OHLCV columns are present
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Rename 'Adj Close' to 'Adj_Close' (adjusted for splits/dividends)
            # Some strategies use adjusted prices to account for corporate actions
            if 'Adj Close' in data.columns:
                data['Adj_Close'] = data['Adj Close']
                data = data.drop('Adj Close', axis=1)
            
            # Remove rows with ANY missing values (NaN)
            # We delete entire rows rather than guess missing prices
            # This ensures backtest only uses real prices that existed
            initial_rows = len(data)
            data = data.dropna()  # Delete rows with NaN
            rows_removed = initial_rows - len(data)
            
            # Warn user if any rows were removed
            if rows_removed > 0:
                print(f"⚠️  Removed {rows_removed} rows with missing data")
            
            # Show download summary
            print(f"✓ Downloaded {len(data)} trading days")
            print(f"  Date Range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"  Columns: {list(data.columns)}")
            
            # Save data to CSV for future use (avoids re-downloading)
            # Filename format: TICKER_YEARSy_INTERVAL.csv (e.g., SPY_10y_1d.csv)
            filepath = self.data_dir / f"{ticker}_{years}y_{interval}.csv"
            data.to_csv(filepath)
            print(f"✓ Saved to: {filepath}")
            
            return data
            
        except Exception as e:
            # Catch any errors (network issues, invalid ticker, etc.)
            print(f" Error downloading data for {ticker}: {e}")
            raise
    
    def load_data(self, ticker, years=10, interval='1d'):
        """
        Load previously downloaded data from CSV.
        
        Args:
            ticker: Stock ticker symbol
            years: Years of data
            interval: Data frequency
            
        Returns:
            DataFrame with OHLCV data
        """
        # Build filepath matching download_data() format
        filepath = self.data_dir / f"{ticker}_{years}y_{interval}.csv"
        
        # Check if file exists before trying to load
        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                f"Run download_data() first."
            )
        
        # Read CSV into DataFrame
        # index_col=0: Use first column (Date) as index
        # parse_dates=True: Convert date strings to datetime objects
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(data)} days from {filepath}")
        
        return data
    
    def get_data(self, ticker, years=10, interval='1d', force_download=False):
        """
        Get data - download if doesn't exist, otherwise load from file.
        Smart function that avoids re-downloading if data already saved.
        
        Args:
            ticker: Stock ticker symbol
            years: Years of data
            interval: Data frequency
            force_download: If True, download even if file exists (gets fresh data)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Build filepath to check if data exists
        filepath = self.data_dir / f"{ticker}_{years}y_{interval}.csv"
        
        # Download if: forced OR file doesn't exist
        if force_download or not filepath.exists():
            return self.download_data(ticker, years, interval)
        else:
            # File exists, load from disk (faster than downloading)
            return self.load_data(ticker, years, interval)
    
    def validate_data(self, data):
        """
        Validate data quality.
        Checks for common data problems that would break backtesting.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            # Total number of rows (trading days)
            'total_rows': len(data),
            
            # Date range of data
            'date_range': f"{data.index[0].date()} to {data.index[-1].date()}",
            
            # Count missing values per column (should be 0 after dropna())
            'missing_values': data.isnull().sum().to_dict(),
            
            # Count negative prices (impossible - indicates data error)
            'negative_prices': (data[['Open', 'High', 'Low', 'Close']] < 0).sum().to_dict(),
            
            # Count days with zero volume (suspicious - market was closed or data error)
            'zero_volume': (data['Volume'] == 0).sum(),
            
            # Check High >= Low (must be true, otherwise data is corrupted)
            'high_low_consistency': ((data['High'] >= data['Low']).all()),
            
            # Check OHLC relationships are valid:
            # - High must be >= Open and Close (highest price of day)
            # - Low must be <= Open and Close (lowest price of day)
            'ohlc_consistency': (
                (data['High'] >= data['Open']).all() and
                (data['High'] >= data['Close']).all() and
                (data['Low'] <= data['Open']).all() and
                (data['Low'] <= data['Close']).all()
            )
        }
        
        # Print validation summary
        print("\n Data Validation:")
        print(f"  Total rows: {validation['total_rows']}")
        print(f"  Date range: {validation['date_range']}")
        print(f"  Missing values: {sum(validation['missing_values'].values())}")
        print(f"  Zero volume days: {validation['zero_volume']}")
        print(f"  OHLC consistency: {'✓ Pass' if validation['ohlc_consistency'] else '✗ Fail'}")
        
        return validation


# Test the data collector (only runs when executing this file directly)
if __name__ == "__main__":
    """
    Test script to download and validate data
    Usage: python src/data_collector.py
    """
    
    # Create data collector instance
    collector = DataCollector()
    
    # Test with S&P 500 ETF (SPY) - most liquid, reliable data
    ticker = "AAPL"
    
    # Download 10 years of daily data
    data = collector.download_data(ticker, years=10, interval='1d')
    
    # Validate data quality (check for errors)
    validation = collector.validate_data(data)
    
    # Show sample of data
    print("\n First 5 rows:")
    print(data.head())
    
    print("\n Last 5 rows:")
    print(data.tail())
    
    # Show statistical summary (min, max, mean, std dev)
    print("\n Basic statistics:")
    print(data.describe())