import json
import unittest
import numpy as np
import pandas as pd
from generate_data import ROOT, generate, FEATURES
from train import load_data, metrics, threshold_for_f1
from predict import score_transactions, load_model

class ProjectTests(unittest.TestCase):
    def test_simulation(self):
        df=generate(1000); self.assertEqual(df.shape,(1000,31)); self.assertEqual(df.Class.sum(),5)
        self.assertTrue(df.equals(generate(1000)))
    def test_data(self):
        df,removed=load_data(ROOT/'data/synthetic_creditcard.csv.gz'); self.assertEqual(len(df),50000); self.assertEqual(df.Class.sum(),250); self.assertEqual(removed,0)
    def test_metrics(self):
        y=np.array([0,0,1,1]); s=np.array([.1,.2,.8,.9]); t=threshold_for_f1(y,s)
        self.assertEqual(metrics(y,s,t)['f1'],1)
    def test_bundle_and_predictions(self):
        a=ROOT/'artifacts'; b=load_model(a/'model.joblib'); summary=json.loads((a/'summary.json').read_text()); self.assertEqual(b['model_name'],summary['selected_model']); self.assertEqual(b['dataset_mode'],'synthetic'); self.assertEqual(b['features'],FEATURES)
        x=pd.read_csv(a/'example_transactions.csv'); out=score_transactions(x,b); self.assertEqual(len(out),len(x)); self.assertTrue(out.risk_score.between(0,1).all()); self.assertTrue(set(out.flag_for_review)<= {0,1})
        with self.assertRaises(ValueError): score_transactions(x.drop(columns=['V1']),b)
        x.loc[0,'Amount']=-1
        with self.assertRaises(ValueError): score_transactions(x,b)
    def test_test_accounting(self):
        s=json.loads((ROOT/'artifacts/summary.json').read_text()); m=s['test_metrics']; self.assertEqual(sum(m[k] for k in ['tn','fp','fn','tp']),s['splits']['test']['rows']); self.assertEqual(m['tp']+m['fn'],s['splits']['test']['fraud'])
if __name__=='__main__': unittest.main()
