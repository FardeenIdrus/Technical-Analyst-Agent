"""
Data Collection Module
Downloads historical OHLCV data for backtesting with advanced validation and caching
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import hashlib

class DataCollector:
    """
    Collects historical price data (OHLCV - Open, High, Low, Close, Volume)
    from Yahoo Finance for backtesting technical strategies.

    Features:
    - Parquet format for 5-10x smaller files and 10x faster I/O
    - Advanced validation (gaps, outliers, anomalies)
    - Data quality scoring (0-100)
    - Version tracking and smart caching
    """

    VERSION = "2.0"  # Data format version for cache invalidation

    def __init__(self, data_dir="data/raw", use_parquet=True):
        """
        Initialize data collector.

        Args:
            data_dir: Directory to save downloaded data
            use_parquet: Use Parquet format (faster, smaller) vs CSV (default: True)
        """
        # Create Path object for the data directory
        self.data_dir = Path(data_dir)
        # Create directory if it doesn't exist (parents=True creates parent dirs too)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Use Parquet for better performance, CSV for compatibility
        self.use_parquet = use_parquet
        self.file_extension = ".parquet" if use_parquet else ".csv"

        # Create metadata directory for cache versioning
        self.metadata_dir = self.data_dir / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)

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

            # Save data with chosen format (Parquet or CSV)
            filepath = self._get_filepath(ticker, years, interval)
            self._save_data(data, filepath)

            # Save metadata for version tracking
            self._save_metadata(ticker, years, interval)

            print(f"✓ Saved to: {filepath}")

            return data

        except Exception as e:
            # Catch any errors (network issues, invalid ticker, etc.)
            print(f" Error downloading data for {ticker}: {e}")
            raise

    def load_data(self, ticker, years=10, interval='1d'):
        """
        Load previously downloaded data from file.

        Args:
            ticker: Stock ticker symbol
            years: Years of data
            interval: Data frequency

        Returns:
            DataFrame with OHLCV data
        """
        # Build filepath matching download_data() format
        filepath = self._get_filepath(ticker, years, interval)

        # Check if file exists before trying to load
        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                f"Run download_data() first."
            )

        # Check cache validity
        if not self._is_cache_valid(ticker, years, interval):
            print("⚠️  Cache outdated, recommend re-downloading with force_download=True")

        # Load data based on format
        data = self._load_data(filepath)
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
        filepath = self._get_filepath(ticker, years, interval)

        # Download if: forced OR file doesn't exist OR cache invalid
        if force_download or not filepath.exists() or not self._is_cache_valid(ticker, years, interval):
            return self.download_data(ticker, years, interval)
        else:
            # File exists and is valid, load from disk (faster than downloading)
            return self.load_data(ticker, years, interval)

    def validate_data(self, data, ticker=None):
        """
        Advanced data quality validation with scoring.
        Checks for common data problems that would break backtesting.

        Args:
            data: DataFrame with OHLCV data
            ticker: Optional ticker symbol for context

        Returns:
            Dictionary with validation results including quality score (0-100)
        """
        validation = {
            # Total number of rows (trading days)
            'total_rows': len(data),

            # Date range of data
            'date_range': f"{data.index[0].date()} to {data.index[-1].date()}",

            # Count missing values per column (should be 0 after dropna())
            'missing_values': data.isnull().sum().to_dict(),
            'missing_count': data.isnull().sum().sum(),

            # Count negative prices (impossible - indicates data error)
            'negative_prices': (data[['Open', 'High', 'Low', 'Close']] < 0).sum().to_dict(),
            'negative_count': (data[['Open', 'High', 'Low', 'Close']] < 0).sum().sum(),

            # Count days with zero volume (suspicious - market was closed or data error)
            'zero_volume': int((data['Volume'] == 0).sum()),

            # Check High >= Low (must be true, otherwise data is corrupted)
            'high_low_consistency': bool((data['High'] >= data['Low']).all()),

            # Check OHLC relationships are valid:
            # - High must be >= Open and Close (highest price of day)
            # - Low must be <= Open and Close (lowest price of day)
            'ohlc_consistency': bool(
                (data['High'] >= data['Open']).all() and
                (data['High'] >= data['Close']).all() and
                (data['Low'] <= data['Open']).all() and
                (data['Low'] <= data['Close']).all()
            )
        }

        # Advanced validation features
        validation.update(self._detect_gaps(data))
        validation.update(self._detect_outliers(data))
        validation.update(self._detect_volume_anomalies(data))

        # Calculate overall quality score (0-100)
        validation['quality_score'] = self._calculate_quality_score(validation)

        # Print validation summary
        self._print_validation_summary(validation, ticker)

        return validation

    def _detect_gaps(self, data):
        """
        Detect missing trading days (gaps in date sequence).

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with gap detection results
        """
        # Calculate expected trading days (roughly 252 per year, accounting for weekends)
        total_days = (data.index[-1] - data.index[0]).days
        expected_trading_days = int(total_days * (252/365))  # ~252 trading days per year
        actual_trading_days = len(data)

        # Detect gaps > 5 days (suspiciously long for trading data)
        date_diffs = data.index.to_series().diff()
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=5)]

        return {
            'expected_trading_days': expected_trading_days,
            'actual_trading_days': actual_trading_days,
            'missing_days_estimate': max(0, expected_trading_days - actual_trading_days),
            'large_gaps_count': len(large_gaps),
            'large_gaps': [(gap.date(), days.days) for gap, days in large_gaps.items()] if len(large_gaps) > 0 else []
        }

    def _detect_outliers(self, data):
        """
        Detect outlier price movements (>20% single-day moves).
        These can indicate stock splits, errors, or extreme volatility.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with outlier detection results
        """
        # Calculate daily returns (percentage change)
        daily_returns = data['Close'].pct_change()

        # Flag returns > 20% or < -20% as outliers
        outlier_threshold = 0.20
        outliers = daily_returns[abs(daily_returns) > outlier_threshold]

        # Also check for extreme High-Low ranges (>30% intraday)
        intraday_range = (data['High'] - data['Low']) / data['Low']
        extreme_ranges = intraday_range[intraday_range > 0.30]

        return {
            'outlier_count': len(outliers),
            'outliers': [(date.date(), f"{ret*100:.2f}%") for date, ret in outliers.items()] if len(outliers) > 0 else [],
            'extreme_range_count': len(extreme_ranges),
            'extreme_ranges': [(date.date(), f"{rng*100:.2f}%") for date, rng in extreme_ranges.items()] if len(extreme_ranges) > 0 else []
        }

    def _detect_volume_anomalies(self, data):
        """
        Detect volume anomalies (spikes or unusual patterns).

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with volume anomaly results
        """
        # Calculate volume statistics
        volume_mean = data['Volume'].mean()
        volume_std = data['Volume'].std()

        # Flag volume > 3 standard deviations as anomalies
        volume_threshold = volume_mean + (3 * volume_std)
        volume_spikes = data[data['Volume'] > volume_threshold]

        # Count zero/low volume days
        low_volume_threshold = volume_mean * 0.1  # Less than 10% of average
        low_volume_days = len(data[data['Volume'] < low_volume_threshold])

        return {
            'volume_mean': float(volume_mean),
            'volume_std': float(volume_std),
            'volume_spike_count': len(volume_spikes),
            'volume_spikes': [(date.date(), int(vol)) for date, vol in volume_spikes['Volume'].items()][:10] if len(volume_spikes) > 0 else [],  # Limit to 10
            'low_volume_days': low_volume_days
        }

    def _calculate_quality_score(self, validation):
        """
        Calculate overall data quality score (0-100).

        Scoring criteria:
        - Missing values: -10 points per 1% missing
        - Negative prices: -20 points per occurrence
        - OHLC inconsistencies: -30 points if failed
        - Large gaps: -5 points per gap
        - Outliers: -2 points per outlier (up to -20 max)
        - Volume anomalies: -1 point per spike (up to -10 max)

        Args:
            validation: Dictionary with validation results

        Returns:
            Quality score (0-100)
        """
        score = 100

        # Deduct for missing values
        missing_pct = (validation['missing_count'] / validation['total_rows']) * 100 if validation['total_rows'] > 0 else 0
        score -= min(50, missing_pct * 10)  # Cap at -50

        # Deduct for negative prices (critical error)
        score -= min(40, validation['negative_count'] * 20)  # Cap at -40

        # Deduct for OHLC inconsistencies
        if not validation['ohlc_consistency']:
            score -= 30

        # Deduct for large gaps
        score -= min(20, validation['large_gaps_count'] * 5)  # Cap at -20

        # Deduct for outliers
        score -= min(20, validation['outlier_count'] * 2)  # Cap at -20

        # Deduct for volume anomalies
        score -= min(10, validation['volume_spike_count'])  # Cap at -10

        # Ensure score stays in 0-100 range
        return max(0, min(100, score))

    def _print_validation_summary(self, validation, ticker=None):
        """
        Print a formatted validation summary.

        Args:
            validation: Dictionary with validation results
            ticker: Optional ticker symbol
        """
        title = f"🔍 Data Validation: {ticker}" if ticker else "🔍 Data Validation"
        print(f"\n{title}")
        print("=" * 60)

        # Basic stats
        print(f"  Total rows: {validation['total_rows']}")
        print(f"  Date range: {validation['date_range']}")
        print(f"  Quality Score: {validation['quality_score']}/100", end="")

        # Add quality rating
        if validation['quality_score'] >= 90:
            print(" (Excellent ✓)")
        elif validation['quality_score'] >= 75:
            print(" (Good)")
        elif validation['quality_score'] >= 60:
            print(" (Fair ⚠️)")
        else:
            print(" (Poor)")

        print("\n  Basic Checks:")
        print(f"    Missing values: {validation['missing_count']}")
        print(f"    Negative prices: {validation['negative_count']}")
        print(f"    Zero volume days: {validation['zero_volume']}")
        print(f"    OHLC consistency: {'✓ Pass' if validation['ohlc_consistency'] else '✗ Fail'}")

        print("\n  Advanced Checks:")
        print(f"    Expected trading days: {validation['expected_trading_days']}")
        print(f"    Actual trading days: {validation['actual_trading_days']}")
        print(f"    Missing days estimate: {validation['missing_days_estimate']}")
        print(f"    Large gaps (>5 days): {validation['large_gaps_count']}")
        print(f"    Price outliers (>20%): {validation['outlier_count']}")
        print(f"    Volume spikes: {validation['volume_spike_count']}")

        # Show details for issues if any
        if validation['large_gaps_count'] > 0:
            print("\n  ⚠️  Large Gaps Detected:")
            for date, days in validation['large_gaps'][:5]:  # Show first 5
                print(f"      {date}: {days} day gap")

        if validation['outlier_count'] > 0:
            print("\n  ⚠️  Price Outliers Detected:")
            for date, change in validation['outliers'][:5]:  # Show first 5
                print(f"      {date}: {change} change")

        print("=" * 60)

    def _get_filepath(self, ticker, years, interval):
        """
        Get filepath for data file.

        Args:
            ticker: Stock ticker symbol
            years: Years of data
            interval: Data frequency

        Returns:
            Path object for data file
        """       
        return self.data_dir / f"{ticker}_{years}y_{interval}{self.file_extension}"

    def _save_data(self, data, filepath):
        """
        Save data to file (Parquet or CSV).

        Args:
            data: DataFrame to save
            filepath: Path to save to
        """
        if self.use_parquet:
            data.to_parquet(filepath, compression='snappy')
        else:
            data.to_csv(filepath)

    def _load_data(self, filepath):
        """
        Load data from file (Parquet or CSV).

        Args:
            filepath: Path to load from

        Returns:
            DataFrame with data
        """
        if self.use_parquet:
            return pd.read_parquet(filepath)
        else:
            return pd.read_csv(filepath, index_col=0, parse_dates=True)

    def _save_metadata(self, ticker, years, interval):
        """
        Save metadata for cache versioning.

        Args:
            ticker: Stock ticker symbol
            years: Years of data
            interval: Data frequency
        """
        metadata = {
            'version': self.VERSION,
            'ticker': ticker,
            'years': years,
            'interval': interval,
            'download_date': datetime.now().isoformat(),
            'format': 'parquet' if self.use_parquet else 'csv'
        }

        metadata_file = self.metadata_dir / f"{ticker}_{years}y_{interval}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _is_cache_valid(self, ticker, years, interval):
        """
        Check if cached data is still valid.

        Args:
            ticker: Stock ticker symbol
            years: Years of data
            interval: Data frequency

        Returns:
            True if cache is valid, False otherwise
        """
        metadata_file = self.metadata_dir / f"{ticker}_{years}y_{interval}.json"

        if not metadata_file.exists():
            return False

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check version match
            if metadata.get('version') != self.VERSION:
                return False

            # Check if data is too old (older than 7 days for daily data)
            download_date = datetime.fromisoformat(metadata['download_date'])
            days_old = (datetime.now() - download_date).days

            if interval == '1d' and days_old > 7:
                return False
            elif interval == '1h' and days_old > 1:
                return False

            return True

        except (json.JSONDecodeError, KeyError):
            return False


# Test the data collector (only runs when executing this file directly)
if __name__ == "__main__":
    """
    Test script to download and validate data
    Usage: python src/data_collector.py
    """

    # Create data collector instance with Parquet format
    print("Testing with Parquet format (faster, smaller)...")
    collector = DataCollector(use_parquet=True)

    # Test with Apple stock
    ticker = "AAPL"

    # Download 10 years of daily data
    data = collector.download_data(ticker, years=10, interval='1d')

    # Validate data quality with advanced checks
    validation = collector.validate_data(data, ticker=ticker)

    # Show sample of data
    print("\n First 5 rows:")
    print(data.head())

    print("\n Last 5 rows:")
    print(data.tail())

    # Show statistical summary (min, max, mean, std dev)
    print("\n Basic statistics:")
    print(data.describe())

    # Test caching
    print("\n Testing cache functionality...")
    data_cached = collector.get_data(ticker, years=10, interval='1d')
    print(f"✓ Successfully loaded from cache")

    # Compare file sizes if both formats exist
    parquet_size = (collector.data_dir / f"{ticker}_10y_1d.parquet").stat().st_size if (collector.data_dir / f"{ticker}_10y_1d.parquet").exists() else 0
    csv_size = (collector.data_dir / f"{ticker}_10y_1d.csv").stat().st_size if (collector.data_dir / f"{ticker}_10y_1d.csv").exists() else 0

    if parquet_size > 0 and csv_size > 0:
        print(f"\n File Size Comparison:")
        print(f"  Parquet: {parquet_size / 1024:.1f} KB")
        print(f"  CSV: {csv_size / 1024:.1f} KB")
        print(f"  Savings: {(1 - parquet_size/csv_size) * 100:.1f}%")
