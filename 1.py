import pandas as pd
import numpy as np
df1 = pd.read_csv('fear_greed_index.csv')
print(df1.head())
df2 = pd.read_csv('historical_data.csv')
print(df2.head())
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df2['Timestamp IST'] = pd.to_datetime(df2['Timestamp IST'], format='%d-%m-%Y %H:%M')
df2['trade_date'] = df2['Timestamp IST'].dt.date
df1['sentiment_date'] = df1['date'].dt.date
sentiment_map = {
    'Extreme Fear': 0,
    'Fear': 1,
    'Neutral': 2,
    'Greed': 3,
    'Extreme Greed': 4
}
df1['sentiment_score'] = df1['classification'].map(sentiment_map)
fear_greed_df = df1[['sentiment_date','sentiment_score', 'classification']]
historical_data_df = df2[['Account','Coin','Execution Price','Size Tokens','Size USD','Side','Closed PnL','Direction','trade_date']]
print(fear_greed_df.head())
print(historical_data_df.head())
merged_df = pd.merge(
    historical_data_df,
    fear_greed_df,
    left_on='trade_date',
    right_on='sentiment_date',
    how='left'
)
print(merged_df.head())
# checking the number of trades with and without sentiment data
total_trades = len(merged_df)
matched_sentiment = merged_df['sentiment_score'].notna().sum()
unmatched_sentiment = total_trades - matched_sentiment
print(f"Total Trades: {total_trades}")
print(f"Matched Sentiment: {matched_sentiment}")
print(f"Unmatched Sentiment: {unmatched_sentiment}")
print(merged_df.head())
merged_df.to_csv('merged_data.csv', index=False)
# Save the merged DataFrame to a CSV file
# Trade level feature 
merged_df['PnL_bin'] = merged_df['Closed PnL'].apply(lambda x: 'Profit' if x > 0 else ('Loss' if x < 0 else 'Zero'))
# PnL bin (profit/loss/zero)
# Convert 'Side' to binary values
merged_df['Side_Binary'] = merged_df['Side'].apply(lambda x : 1 if str(x).upper() == 'BUY' else 0)
# Trade Imapct (proxy for exposure)
merged_df['Trade_Imapct'] = merged_df['Size USD']
# Normalized PnL = PnL/Size USD (risk Adjusted return)
merged_df['Closed_PnL_Numeric'] = historical_data_df['Closed PnL']
merged_df['Normalized_PnL'] = merged_df.apply(
    lambda row : row['Closed_PnL_Numeric'] / row['Size USD'] if row['Size USD'] else 0, axis=1
)
merged_df[['Account', 'trade_date', 'Side', 'Side_Binary', 'Size USD',
           'Closed PnL', 'PnL_bin', 'Trade_Imapct', 'Normalized_PnL','classification']]
print(merged_df.head())

# Account_Level Daily Aggregation

daily_account_stats = merged_df.groupby(['Account', 'trade_date', 'classification', 'sentiment_score']).agg({
    'Closed PnL': ['sum', 'mean'],
    'Size USD': ['sum', 'mean'],
    'Side_Binary': 'mean',  
    'PnL_bin': lambda x: (x == 'Profit').sum(),  
    'Account': 'count'  
}).reset_index()

daily_account_stats.columns = [
    'Account', 'trade_date', 'classification', 'sentiment_score',
    'Total_PnL', 'Avg_PnL',
    'Total_Trade_Size', 'Avg_Trade_Size',
    'Buy_Ratio',
    'Profit_Trades',
    'Total_Trades'
]

daily_account_stats['Profit_Rate'] = daily_account_stats['Profit_Trades'] / daily_account_stats['Total_Trades']
print(daily_account_stats.head())
merged_df['Closed PnL'] = pd.to_numeric(merged_df['Closed PnL'], errors='coerce')

# Group by sentiment and calculate mean PnL
avg_pnl_by_sentiment = merged_df.groupby('classification')['Closed PnL'].mean()
print("Average Closed PnL by Sentiment:")
print(avg_pnl_by_sentiment)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Plot 1: Closed PnL vs. Sentiment Class
plt.figure(figsize=(8, 5))
sns.boxplot(data=merged_df, x='classification', y='Closed PnL')
plt.title("Closed PnL Distribution vs. Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Closed PnL")
plt.show()

# Plot 2: Average PnL vs. Sentiment Class 
plt.figure()
sns.boxplot(data=daily_account_stats, x='classification', y='Avg_PnL', palette='coolwarm')
plt.title('Average PnL vs. Market Sentiment')
plt.ylabel('Average PnL')
plt.xlabel('Market Sentiment')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot 3: Profit Rate vs. Sentiment Class
plt.figure()
sns.boxplot(data=daily_account_stats, x='classification', y='Profit_Rate', palette='viridis')
plt.title('Profit Rate vs. Market Sentiment')
plt.ylabel('Profit Rate')
plt.xlabel('Market Sentiment')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot 4: Buy Ratio vs. Sentiment Class
plt.figure()
sns.boxplot(data=daily_account_stats, x='classification', y='Buy_Ratio', palette='Set2')
plt.title('Buy Ratio vs. Market Sentiment')
plt.ylabel('Buy Ratio')
plt.xlabel('Market Sentiment')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot 5: Number of Trades vs. Sentiment Class
sentiment_counts = daily_account_stats['classification'].value_counts()
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['orange', 'green', 'blue', 'red', 'black'])
plt.title("Number of Trades vs. Market Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Number of Trades")
plt.xticks(rotation=0)
plt.show()

# Plot 6: Win Rate vs. Sentiment Class
merged_df['Is_Profit'] = merged_df['Closed PnL'] > 0
win_rate = merged_df.groupby('classification')['Is_Profit'].mean() * 100
print("Win Rate by Market Sentiment (%):")
print(win_rate)
plt.figure(figsize=(7, 4))
sns.barplot(x=win_rate.index, y=win_rate.values, palette=['orange', 'green', 'blue', 'red', 'black'])
plt.title("Win Rate vs. Market Sentiment")
plt.ylabel("Win Rate (%)")
plt.xlabel("Sentiment")
plt.ylim(0, 100)
plt.show()

# Plot 7: Average Trade Size vs. Sentiment Class
merged_df['Size USD'] = pd.to_numeric(merged_df['Size USD'], errors='coerce')
avg_trade_size = merged_df.groupby('classification')['Size USD'].mean()
print("Average Trade Size (USD) by Sentiment:")
print(avg_trade_size)
plt.figure(figsize=(7, 4))
sns.barplot(x=avg_trade_size.index, y=avg_trade_size.values, palette=['orange', 'green', 'blue', 'red', 'black'])
plt.title("Average Trade Size vs. Market Sentiment")
plt.ylabel("Average Size (USD)")
plt.xlabel("Sentiment")
plt.show()
