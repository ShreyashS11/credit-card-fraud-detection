"""Fetch the public ULB/Worldline CSV mirror if internet is available."""
from urllib.request import urlretrieve
from generate_data import ROOT
from train import load_data

if __name__=='__main__':
    path=ROOT/'data/creditcard.csv'; temp=path.with_suffix('.part')
    if path.exists(): raise SystemExit(f'{path} already exists; not overwriting.')
    try:
        urlretrieve('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv',temp)
        df,removed=load_data(temp)
        if len(df)<250000 or df.Class.sum()<400: raise ValueError('Unexpected benchmark size.')
        temp.replace(path)
    except Exception as e:
        temp.unlink(missing_ok=True)
        raise SystemExit(f'Download failed: {e}\nDownload manually: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud')
    print(f'Downloaded {path}')
