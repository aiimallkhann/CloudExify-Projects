# CloudExify Project 1: Sales Data Analysis

## Project Overview
This project focuses on performing an analysis of a corporate retail sales dataset using Python and Jupyter Notebook. The goal is to clean a raw dataset, extract key financial performance indicators, analyze categories, and visualize sales patterns over time and by region.

* **Project Name:** Sales Data Analysis
* **Language:** Python (Jupyter Notebook)
* **Key Libraries:** pandas, numpy, matplotlib
* **Submission Date:** 15 July 2026

---

## Understanding the Dataset
The dataset (`sales_data.csv`) contains **9,994 rows** and **21 columns** of operational data. The key columns used in this analysis include:
* **Date:** The day the transaction occurred (initially loaded as text/object data).
* **Product:** The specific product name or SKU sold.
* **Category:** The macro grouping of the item (`Furniture`, `Office Supplies`, `Technology`).
* **Region:** Geographic market sectors (`North`, `South`, `East`, `West`).
* **Amount:** The monetary value or total revenue generated from the sale.
* **Units:** The total number of items purchased in the transaction.

---

## Code Implementation & Methodology

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Ingestion and Initial Inspection
data = pd.read_csv('sales_data.csv')

print('\n\nFirst 5 Rows\n\n')
print(data.head())

print('\n\nData Shape\n\n')
print(data.shape)

print('\n\nData Info\n\n')
print(data.info())

print('\n\nSatistical Summary\n\n')
print(data.describe())

print('\n\nMissing Values per column:\n', data.isnull().sum())

# 2. Data Cleaning
data = data.dropna()
data = data.drop_duplicates()
data['Date'] = pd.to_datetime(data['Date'])

print('\nCleaned Data Types:\n', data.dtypes)

# 3. Exploratory Analysis
print('Exploratory Analysis Insights\n')
print(f'Total Revenue: Rs {data["Amount"].sum()}')
print(f'Average Sale: Rs {data["Amount"].mean()}')
print(f'Median Sale: Rs {data["Amount"].median()}')
print(f'Standard Deviation: Rs {data["Amount"].std()}')

print('\nProduct Sales Frequency\n')
print(data['Product'].value_counts())

print('Top 10 Products by Revenue')
byProduct = data.groupby('Product')['Amount'].sum().sort_values(ascending=False)
print(byProduct.head(10))

# 4. Visualizations
# Top 10 Products Bar Chart
plt.figure(figsize=(12, 6))
top_products = data.groupby('Product')['Amount'].sum().nlargest(10)
top_products.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Top 10 Products by Revenue', fontsize=14)
plt.xlabel('Product', fontsize=12)
plt.ylabel('Revenue (Rs)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly Sales Trend Line Plot
plt.figure(figsize=(12, 6))
monthly_sales = data.groupby(data['Date'].dt.to_period('M'))['Amount'].sum()
monthly_sales.plot(kind='line', marker='o', color='green', linewidth=2)
plt.title('Monthly Sales Trend', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Revenue (Rs)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Regional Sales Pie Chart
plt.figure(figsize=(8, 8))
by_region = data.groupby('Region')['Amount'].sum()
by_region.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Sales Distribution by Region', fontsize=14)
plt.ylabel('') 
plt.show()

# 5. Category Analysis
by_category = data.groupby('Category').agg({
    'Amount': 'sum',
    'Units': 'sum'
})
print("\n--- Category Comparison ---")
print(by_category)

# 6. Generate Summary Report
print('SALES ANALYSIS REPORT')
print(f"Total Records Analyzed : {len(data)}")
print(f"Date Range             : {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
print(f"Total Revenue Generated: Rs {data['Amount'].sum()}")
print(f"Average Sale Value     : Rs {data['Amount'].mean()}")
print(f"Median Sale Value      : Rs {data['Amount'].median()}")

print('\nTop 5 Products by Revenue')
print(data.groupby('Product')['Amount'].sum().nlargest(5).apply(lambda x: f"Rs {x}"))

print('\nBest Performing Month')
best_month = data.groupby(data['Date'].dt.to_period('M'))['Amount'].sum().idxmax()
best_month_val = data.groupby(data['Date'].dt.to_period('M'))['Amount'].sum().max()
print(f"Month: {best_month} (Revenue: Rs {best_month_val})")
