# CloudExify DS Project 3 — House Price Prediction

Regression task predicting house prices from Area, Bedrooms, Age, and Location.

## Dataset
`house_prices.csv` — 300 rows, features: Area, Bedrooms, Age, Location (North/South/East/West), Price (Rs).
8 rows had missing values, dropped before training (~2.7% of data).

## Approach
- One-hot encoded `Location` via `pd.get_dummies`
- 80/20 train/test split (`random_state=0`)
- Compared Linear Regression vs Random Forest Regressor (`n_estimators=100`)

## Results
| Model | R² (Test) | RMSE |
|---|---|---|
| Linear Regression | 0.987 | Rs 219,263 |
| Random Forest | 0.958 | Rs 396,535 |

Linear Regression outperformed Random Forest. Price in this dataset is close to a linear
function of the features (Area × location rate, plus bedroom/age adjustments), which favors
a linear model — Random Forest has no non-linear structure to exploit here.

## Files
- `CloudExify-Project3.ipynb` — full notebook (EDA, training, evaluation, plots)
- `CloudExify-Project3.py` — script version
- `house_prices.csv` — dataset
