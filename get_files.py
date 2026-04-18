from sec_edgar_downloader import Downloader

# --- Configure these before running ---
YOUR_NAME = "Johann Antisseril"
YOUR_EMAIL = "johannantisseril@gmail.com"
TICKERS = ["AAPL", "NVDA", "GOOGL"]  # List of stock tickers to download filings for
FILING_TYPE = "10-K"
NUM_FILINGS = 3  # How many years back to download per ticker
# --------------------------------------

dl = Downloader(YOUR_NAME, YOUR_EMAIL)
for ticker in TICKERS:
    print(f"Downloading {NUM_FILINGS}x {FILING_TYPE} for {ticker}...")
    dl.get(
        FILING_TYPE, ticker, limit=NUM_FILINGS
    )  # Downloads to ./sec-edgar-filings/ by default
    print(f"  Done: {ticker}")

print("\nAll downloads complete. Run 'python ingest.py' next.")
