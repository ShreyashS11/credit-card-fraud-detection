"""Train, choose on validation, and evaluate once on held-out data."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve,
    roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from generate_data import FEATURES, ROOT

SEED = 42
DEFAULT_DATA = ROOT / 'data/synthetic_creditcard.csv.gz'
DEFAULT_OUT = ROOT / 'artifacts'

def load_data(path):
    raw = pd.read_csv(path)
    required = FEATURES + ['Class']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df = raw[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError('Missing or nonfinite values: clean the source before training.')
    if set(df.Class.unique()) != {0, 1}:
        raise ValueError('Class must contain both 0 and 1 only.')
    # Keep one identical record; discard entire feature groups with conflicting labels.
    before = len(df)
    df = df.drop_duplicates()
    df = df.loc[~df.duplicated(FEATURES, keep=False)].reset_index(drop=True)
    if df.Class.value_counts().min() < 20:
        raise ValueError('At least 20 examples of each class are required.')
    return df, before - len(df)

def threshold_for_f1(y, score):
    p, r, thresholds = precision_recall_curve(y, score)
    f = 2*p[:-1]*r[:-1] / np.maximum(p[:-1]+r[:-1], 1e-15)
    return float(thresholds[np.argmax(f)])

def metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred), precision=precision_score(y,pred,zero_division=0),
                recall=recall_score(y,pred,zero_division=0), f1=f1_score(y,pred,zero_division=0),
                average_precision=average_precision_score(y,score), roc_auc=roc_auc_score(y,score),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def run(data_path=DEFAULT_DATA, out=DEFAULT_OUT, mode='synthetic'):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df, dropped = load_data(data_path)
    X, y = df[FEATURES], df.Class.astype(int)
    Xdev, Xtest, ydev, ytest = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    Xtrain, Xval, ytrain, yval = train_test_split(Xdev, ydev, test_size=.25, stratify=ydev, random_state=SEED)
    ratio = float((ytrain==0).sum()/(ytrain==1).sum())
    models = {
      'Majority baseline': DummyClassifier(strategy='prior'),
      'Decision Tree': DecisionTreeClassifier(max_depth=7, min_samples_leaf=10, class_weight='balanced',random_state=SEED),
      'Random Forest': RandomForestClassifier(n_estimators=80,max_depth=12,min_samples_leaf=2,class_weight='balanced_subsample',random_state=SEED,n_jobs=2),
      'XGBoost': XGBClassifier(n_estimators=100,max_depth=4,learning_rate=.08,subsample=.9,colsample_bytree=.85,scale_pos_weight=ratio,eval_metric='logloss',random_state=SEED,n_jobs=2),
      'LightGBM': LGBMClassifier(n_estimators=100,num_leaves=15,learning_rate=.08,scale_pos_weight=ratio,random_state=SEED,n_jobs=2,verbosity=-1),
    }
    rows, val_rows, thresholds, scores = [], [], {}, {}
    for name, model in models.items():
        start = time.perf_counter(); model.fit(Xtrain, ytrain); elapsed = time.perf_counter()-start
        val = model.predict_proba(Xval)[:,1]
        threshold = .5 if name == 'Majority baseline' else threshold_for_f1(yval,val)
        thresholds[name] = threshold
        val_rows.append({'model':name,'threshold':threshold,'fit_seconds':elapsed,**metrics(yval,val,threshold)})
    validation = pd.DataFrame(val_rows)
    best = validation.loc[validation.model != 'Majority baseline'].sort_values('average_precision',ascending=False).iloc[0].model
    # Selection is frozen before accessing test scores.
    for name, model in models.items():
        score = model.predict_proba(Xtest)[:,1]; scores[name] = score
        rows.append({'model':name,'threshold':thresholds[name],**metrics(ytest,score,thresholds[name])})
    results = pd.DataFrame(rows)
    validation.to_csv(out/'validation_metrics.csv',index=False)
    results.to_csv(out/'test_metrics.csv',index=False)
    selected = results.set_index('model').loc[best].to_dict()
    summary = {'dataset_mode':mode,'dataset_file':Path(data_path).name,'rows':len(df),'duplicates_or_conflicts_removed':dropped,
       'fraud_count':int(y.sum()),'fraud_rate':float(y.mean()),'seed':SEED,'train_weight_ratio':ratio,
       'splits':{key:{'rows':len(labels),'fraud':int(labels.sum())} for key,labels in [('train',ytrain),('validation',yval),('test',ytest)]},
       'selected_model':best,'selection_rule':'Highest validation average precision; threshold maximizes validation F1',
       'test_metrics':selected,'features':FEATURES}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    joblib.dump({'model':models[best],'threshold':thresholds[best],'features':FEATURES,'dataset_mode':mode,'model_name':best},out/'model.joblib',compress=3)
    # Demo examples use held-out inputs, not card numbers or personally identifying data.
    examples = Xtest.assign(Class=ytest).groupby('Class',group_keys=False).head(3)
    examples.to_csv(out/'example_transactions.csv',index=False)
    errors = Xtest.assign(actual=ytest,score=scores[best],predicted=(scores[best]>=thresholds[best]).astype(int))
    errors.loc[errors.actual!=errors.predicted].to_csv(out/'test_errors.csv',index=False)
    if hasattr(models[best],'feature_importances_'):
        imp = pd.Series(models[best].feature_importances_,index=FEATURES).sort_values(ascending=False)
        imp.to_csv(out/'feature_importance.csv',header=['importance'])
        fig,ax=plt.subplots(figsize=(7,4)); imp.head(10).sort_values().plot.barh(ax=ax,color='#197c83'); ax.set_title(f'{best}: top feature importances'); fig.tight_layout(); fig.savefig(out/'feature_importance.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.6)); ax.bar(['Legitimate','Fraud'],[(y==0).sum(),(y==1).sum()],color=['#31688e','#dd8452']); ax.set_yscale('log'); ax.set_ylabel('Transactions (log scale)'); ax.set_title(f'Class imbalance — {mode}'); fig.tight_layout(); fig.savefig(out/'class_distribution.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.2))
    for name,score in scores.items():
        p,r,_=precision_recall_curve(ytest,score); ap=average_precision_score(ytest,score); ax.plot(r,p,label=f'{name} ({ap:.3f})')
    ax.set(xlabel='Recall',ylabel='Precision',title='Held-out precision–recall curves',xlim=(0,1),ylim=(0,1.03)); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'precision_recall.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.2))
    for name,score in scores.items():
        fpr,tpr,_=roc_curve(ytest,score); ax.plot(fpr,tpr,label=f'{name} ({roc_auc_score(ytest,score):.3f})')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Held-out ROC curves'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'roc.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay.from_predictions(ytest,(scores[best]>=thresholds[best]).astype(int),display_labels=['Legitimate','Fraud'],cmap='Blues',values_format='d',colorbar=False,ax=ax); ax.set_title(best+' — held-out test'); fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); results.set_index('model')[['precision','recall','f1','average_precision']].plot.bar(ax=ax); ax.set_ylim(0,1.05); ax.set_title('Test metric comparison'); ax.tick_params(axis='x',rotation=18); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'model_comparison.png',dpi=140); plt.close(fig)
    print(results.round(4).to_string(index=False)); print('Selected:',best)
    return summary

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DEFAULT_DATA); p.add_argument('--output',type=Path,default=DEFAULT_OUT); p.add_argument('--mode',choices=['synthetic','real'],default='synthetic')
    args=p.parse_args()
    if args.mode=='real' and (args.data.resolve()==DEFAULT_DATA.resolve() or args.output.resolve()==DEFAULT_OUT.resolve()):
        p.error('Real mode requires a separate real CSV and separate output directory.')
    if args.mode=='synthetic' and args.data.resolve()!=DEFAULT_DATA.resolve():
        p.error('Use --mode real for an independently downloaded benchmark CSV.')
    run(args.data,args.output,args.mode)
