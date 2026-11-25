"""Score anonymized transaction features using the bundled trusted model."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from generate_data import ROOT

def load_model(path):
    # Never deserialize joblib files from unknown or untrusted sources.
    return joblib.load(path)

def score_transactions(frame, bundle):
    features = bundle['features']
    if not set(features).issubset(frame.columns):
        raise ValueError('Required fields: '+', '.join(features))
    x=frame[features].apply(pd.to_numeric,errors='raise')
    if not len(x) or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Provide at least one complete, finite numeric transaction.')
    if (x[['Amount','Time']] < 0).any().any():
        raise ValueError('Amount and Time must be nonnegative.')
    risk=bundle['model'].predict_proba(x)[:,1]
    return pd.DataFrame({'risk_score':risk,'flag_for_review':(risk>=bundle['threshold']).astype(int)},index=frame.index)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',type=Path,default=ROOT/'artifacts/example_transactions.csv'); p.add_argument('--model',type=Path,default=ROOT/'artifacts/model.joblib'); p.add_argument('--output',type=Path)
    args=p.parse_args(); frame=pd.read_csv(args.input); result=score_transactions(frame,load_model(args.model))
    if args.output: result.to_csv(args.output,index=False)
    print(result.to_string(index=False))
