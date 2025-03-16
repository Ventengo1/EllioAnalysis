
This project implements basic Elliott Wave theory for stock analysis, providing buy, sell, or hold recommendations. It analyzes historical stock data to identify Elliott Wave patterns and predict potential future movements.

**Disclaimer:** This code currently takes approximately 8 hours to complete analysis with 5 years of data at a granularity of 15 minutes. Performance can vary based on system specifications and network speed. This is also for educational purposes and should not be taken as financial advice.

Features

* Elliott Wave Pattern Recognition: Identifies Elliott Wave patterns in stock data.
* Buy/Sell/Hold Signals: Generates trading signals based on identified patterns.
* Data Visualization: Plots Elliott Wave patterns on stock price charts.
* Divergence Detection: Checks for positive and negative divergence between price and volume.
* Uses yfinance: Uses the latest version of `yfinance` to download stock data.

## Usage

1.  **Run the Python script:**

2.  **Enter the stock symbol:**

    When prompted, enter the stock symbol (exp: SPY) you want to analyze.

3.  **View the results:**

    The script will download stock data, analyze it for Elliott Wave patterns, and display a chart with identified waves. It will also provide a buy, sell, or hold recommendation based on the analysis.

## Dependencies

yfinance: For downloading stock data.
numpy: For numerical computations.
pandas: For data manipulation and analysis.
matplotlib: For data visualization.
scipy: For signal processing.
seaborn: For enhanced data visualization.
ipython: for inline plotting configuration.
