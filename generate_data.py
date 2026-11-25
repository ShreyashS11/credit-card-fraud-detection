"""Generate a reproducible SIMULATED benchmark, not real cardholder data."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

def generate(n=50000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=round(n * .005), replace=False)] = 1
    v = rng.normal(size=(n, 28))
    fraud = y == 1
    # Multiple overlapping fraud subtypes; some have little signal.
    subtype = rng.choice(3, size=fraud.sum(), p=[.55, .30, .15])
    idx = np.flatnonzero(fraud)
    a, b = idx[subtype == 0], idx[subtype == 1]
    v[a, :5] += np.array([-3.0, 2.5, -2.4, 1.8, -1.2])
    v[b, 5:10] += np.array([2.7, -2.5, 2.0, -2.0, 1.6])
    # Shared mild signal, plus legitimate outliers to make the task nontrivial.
    v[fraud, 10] += 1.0
    out = rng.choice(np.flatnonzero(~fraud), size=200, replace=False)
    v[out, :5] += np.array([-1.5, 1.0, -1.2, .9, -.6])
    amount = rng.lognormal(3.5 + .35*y, 1.2, n).clip(0, 20000).round(2)
    time = rng.integers(0, 172800, n)
    df = pd.DataFrame(v.round(5), columns=FEATURES[1:-1])
    df.insert(0, 'Time', time)
    df['Amount'] = amount
    df['Class'] = y
    return df

if __name__ == '__main__':
    path = ROOT / 'data' / 'synthetic_creditcard.csv.gz'
    generate().to_csv(path, index=False, compression={'method':'gzip','mtime':0})
    print(f'Saved simulated dataset to {path}')
