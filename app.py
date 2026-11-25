"""Local-only browser prototype. Does not collect card details or store requests."""
import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import pandas as pd
from predict import load_model, score_transactions
from generate_data import ROOT

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Fraud Review | Project 19</title><style>
*{box-sizing:border-box}body{font:16px system-ui;background:#eef4f7;color:#142c3b;margin:0}header{background:#12344a;color:white;padding:32px 6%;}h1{margin:0 0 8px;font-size:30px}main{max-width:1000px;margin:24px auto;padding:0 18px}.card{background:white;border:1px solid #d8e3e9;border-radius:12px;padding:24px;margin-bottom:18px}.note{color:#526b79;line-height:1.5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px}label{display:block;font-size:13px;font-weight:600}input,select,button{font:inherit;padding:9px;border:1px solid #b8cbd6;border-radius:6px;width:100%;margin-top:5px}button{background:#147d83;color:white;border:none;font-weight:700;cursor:pointer;max-width:240px;margin:18px 0 0}.result{white-space:pre-wrap;font-size:18px;font-weight:600;line-height:1.5}footer{color:#526b79;font-size:13px;padding:12px}</style>
<header><small>B.TECH MACHINE LEARNING • PROJECT 19</small><h1>Credit Card Fraud Detection</h1><div>Ensemble-based transaction review prototype</div></header><main><section class="card"><strong id="mode"></strong><p class="note">Educational demonstration. Scores are not calibrated fraud probabilities. A flag requests review, not an automatic rejection. Do not enter card numbers, PINs, CVVs or personal information.</p><label for="sample">Load a held-out example</label><select id="sample"></select><p class="note">Time and Amount plus 28 anonymized features. In synthetic mode these are simulated numeric features, not real PCA values.</p><form id="form"><div class="grid" id="fields"></div><button type="submit">Evaluate transaction</button></form></section><section class="card"><h2>Review outcome</h2><div class="result" id="result" aria-live="polite">Choose an example or edit the fields, then evaluate.</div></section><footer>Local prototype · No network service required · Not for payment authorization</footer></main>
<script>let info; const $=id=>document.getElementById(id); async function init(){info=await (await fetch('/info')).json();$('mode').textContent=info.mode.toUpperCase()+' DATA • '+info.model;info.features.forEach(f=>{let l=document.createElement('label');l.textContent=f;let i=document.createElement('input');i.type='number';i.step='any';i.required=true;i.name=f;l.append(i);$('fields').append(l)});info.examples.forEach((e,k)=>{let o=document.createElement('option');o.value=k;o.textContent='Example '+(k+1)+' — actual '+(e.Class?'fraud':'legitimate');$('sample').append(o)});load();}function load(){const e=info.examples[+$('sample').value];info.features.forEach(f=>$('form').elements[f].value=e[f]);$('result').textContent='Example loaded. Evaluate to see the model output.'}$('sample').onchange=load;$('form').onsubmit=async e=>{e.preventDefault();try{const data=Object.fromEntries(new FormData(e.target));const resp=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const r=await resp.json();if(!resp.ok)throw Error(r.error);$('result').textContent=(r.flag_for_review?'FLAG FOR REVIEW':'NOT FLAGGED')+'\\nRisk score: '+r.risk_score.toFixed(4)+' | Review threshold: '+r.threshold.toFixed(4);}catch(err){$('result').textContent=err.message}};init().catch(e=>$('result').textContent=e.message);</script></html>'''

def make_handler(bundle, examples):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, data, status=200, kind='application/json'):
            body=(json.dumps(data) if kind=='application/json' else data).encode()
            self.send_response(status); self.send_header('Content-Type',kind+'; charset=utf-8'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path=='/': self.respond(PAGE,kind='text/html')
            elif self.path=='/info': self.respond({'mode':bundle['dataset_mode'],'model':bundle['model_name'],'features':bundle['features'],'examples':examples})
            else: self.respond({'error':'Not found'},404)
        def do_POST(self):
            if self.path!='/predict': return self.respond({'error':'Not found'},404)
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=16000: raise ValueError('Invalid request length.')
                data=json.loads(self.rfile.read(n)); result=score_transactions(pd.DataFrame([data]),bundle).iloc[0]
                self.respond({'risk_score':float(result.risk_score),'flag_for_review':int(result.flag_for_review),'threshold':bundle['threshold']})
            except (ValueError,TypeError,KeyError) as e: self.respond({'error':str(e)},400)
    return Handler

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--artifacts',type=Path,default=ROOT/'artifacts'); p.add_argument('--port',type=int,default=8000); a=p.parse_args()
    bundle=load_model(a.artifacts/'model.joblib'); examples=pd.read_csv(a.artifacts/'example_transactions.csv').to_dict('records')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),make_handler(bundle,examples)); print(f'Open http://127.0.0.1:{a.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
