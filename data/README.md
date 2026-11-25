# Dataset documentation

## Included: synthetic_creditcard.csv.gz

Generated locally by `generate_data.py` with NumPy seed 42. Contains 50,000 rows: 49,750 legitimate and 250 simulated fraud (0.5%). No real personal or payment data. Pandas reads the gzip file directly. All 30 features are numeric; Class is the target. No missing values or duplicates in the included data. Labels are assigned first, then overlapping synthetic feature patterns are generated for three fraud subtypes, including a weak-signal subtype and legitimate outliers. This is a simulation, not a sample from the ULB dataset.

| Field | Synthetic meaning | Real ULB benchmark meaning |
|---|---|---|
| Time | Simulated seconds in two-day window | Seconds elapsed since first transaction |
| V1–V28 | Artificial standardized-like numeric variables | Anonymized PCA-transformed variables |
| Amount | Simulated positive transaction amount | Transaction amount |
| Class | 0 legitimate, 1 simulated fraud | 0 legitimate, 1 fraud |

## Optional: creditcard.csv (real data, not included)

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud — research collaboration of Worldline and ULB. The dataset describes 284,807 European card transactions in September 2013 over two days, 492 frauds (about 0.173%) before deduplication. See the source for terms and attribution. The real CSV is not bundled in this project. The original benchmark could not be fetched during project construction; no actual benchmark metrics were computed. Run the real-mode commands in README to obtain independent results. No real dataset is needed for the simulated demo.
