import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plotting style
sns.set(style="whitegrid")

# Load Brent oil price data
data_path = "C:/Users/hp/Desktop/Kifiya AIM/week 10/Technical Content/Workspace/Data/BrentOilPrices.csv"
df = pd.read_csv(data_path)

# Convert Date column to datetime with mixed format
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')

# Check for any invalid dates
if df['Date'].isna().any():
    print("Warning: Some dates could not be parsed:")
    print(df[df['Date'].isna()])

# Basic EDA
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Analyze time series properties
min_price = df['Price'].min()
max_price = df['Price'].max()
min_date = df['Date'][df['Price'].idxmin()].strftime('%d-%b-%Y')
max_date = df['Date'][df['Price'].idxmax()].strftime('%d-%b-%Y')
volatility = df['Price'].std()
log_returns_volatility = df['Price'].apply(lambda x: np.log(x)).diff().std()

time_series_summary = f"""
Time Series Properties:
- Date Range: {df['Date'].min().strftime('%d-%b-%Y')} to {df['Date'].max().strftime('%d-%b-%Y')}
- Price Range: ${min_price:.2f} ({min_date}) to ${max_price:.2f} ({max_date})
- Standard Deviation of Prices: ${volatility:.2f}
- Standard Deviation of Log Returns: {log_returns_volatility:.4f}
- Observations: Prices show long-term trends with spikes (e.g., 2008) and drops (e.g., April 2020). Log returns exhibit volatility clustering during crises (e.g., 2020).
"""

# Plot raw price series
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], label='Brent Oil Price (USD/barrel)')
plt.title('Brent Oil Prices (1987-2022)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.savefig('C:/Users/hp/Desktop/Kifiya AIM/week 10/Technical Content/Workspace/plots/raw_price_series.png')
plt.close()

# Calculate and plot log returns
df['Log_Returns'] = df['Price'].apply(lambda x: np.log(x)).diff()
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Log_Returns'], label='Log Returns')
plt.title('Log Returns of Brent Oil Prices')
plt.xlabel('Date')
plt.ylabel('Log Returns')
plt.legend()
plt.savefig('C:/Users/hp/Desktop/Kifiya AIM/week 10/Technical Content/Workspace/plots/log_returns.png')
plt.close()

# Assumptions and limitations
assumptions_limitations = """
Assumptions:
- Brent oil price data is accurate, though date formats vary (e.g., 20-May-87, Apr 22, 2020).
- Major events (geopolitical, OPEC, economic) drive significant price changes.
- Log returns are stationary, suitable for modeling.
- Bayesian models can detect structural breaks effectively.

Limitations:
- Correlation vs. Causation: Change points aligned with events show correlation, not causation. Other factors (e.g., market sentiment) may contribute.
- Event Data: Incomplete or inaccurate event dates may lead to misattribution.
- Data Quality: Mixed date formats required preprocessing; unparsed dates (if any) may affect results.
- Model Scope: Initial model excludes external variables (e.g., GDP).
"""

# Communication channels
communication_channels = """
Communication Channels:
- Report: PDF or blog post summarizing findings.
- Dashboard: Interactive Flask/React interface for stakeholders.
- Presentations: For Birhan Energies’ clients.
"""

# Data Analysis Workflow
workflow = """
Data Analysis Workflow for Brent Oil Price Change Point Analysis
============================================================
1. Data Preparation:
   - Load Brent oil price data (1987-2022) from data/BrentOilPrices.csv.
   - Convert Date column to datetime, handling mixed formats (e.g., 20-May-87, Apr 22, 2020).
   - Handle missing values and outliers, if any.
2. Exploratory Data Analysis (EDA):
   - Plot raw price series to identify trends and shocks.
   - Compute log returns to analyze volatility and check stationarity.
   - Perform statistical tests (e.g., ADF test) in Task 2.
3. Event Data Compilation:
   - Research 10-15 major events (geopolitical, OPEC, economic).
   - Save in data/events.csv with dates and descriptions.
4. Change Point Modeling (Task 2):
   - Use PyMC to implement Bayesian Change Point model.
   - Define priors for switch points and parameters (mean, volatility).
   - Run MCMC to estimate posterior distributions.
5. Interpretation (Task 2):
   - Align change points with events from events.csv.
   - Quantify price impacts (e.g., percentage change in mean).
6. Visualization and Dashboard (Task 3):
   - Build Flask/React dashboard to display trends and event impacts.
   - Include interactive features (filters, date ranges).
7. Reporting:
   - Summarize findings in a report or blog post for stakeholders.
============================================================
"""

# Save workflow
with open('C:/Users/hp/Desktop/Kifiya AIM/week 10/Technical Content/Workspace/workflow.txt', 'w') as f:
    f.write(workflow)

# Print results for report
print("\nResults for Interim Report:")
print(time_series_summary)
print(assumptions_limitations)
print(communication_channels)
print("Workflow saved to ../workflow.txt")