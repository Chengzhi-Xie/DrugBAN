#!/usr/bin/env python3
"""Pure-formula DTI scoring for DrugBAN cluster splits.

No optimized classifier or neural network. Source labels are used only to form
class-conditional reference statistics. Target labels are evaluation-only.
"""
from __future__ import annotations
import argparse, hashlib, json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, RDKFingerprint, rdFingerprintGenerator
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score
from scjke_pcm.scjke_pcm import (
    AdaptiveSCJKEPCM, _drug_descriptor, _protein_descriptor,
    _canonical_smiles, _clean_protein, _top_k_mean,
)
EPS=1e-8
GROUPS=("AGV","ILFP","YMTS","HNQW","RK","DE","C")
AA2G={a:str(i) for i,g in enumerate(GROUPS) for a in g}

def prep(df, label=True):
    need={"SMILES","Protein"}|({"Y"} if label else set())
    if need-set(df): raise ValueError(f"missing {need-set(df)}")
    x=df.copy().reset_index(drop=True)
    x.SMILES=x.SMILES.map(_canonical_smiles); x.Protein=x.Protein.map(_clean_protein)
    if label:
        x.Y=pd.to_numeric(x.Y).astype(int)
        bad=x.groupby(["SMILES","Protein"]).Y.nunique()
        if (bad>1).any(): raise ValueError("conflicting labels")
        x=x.drop_duplicates(["SMILES","Protein"]).reset_index(drop=True)
    return x

def bucket(s,m=5): return int(hashlib.sha1(s.encode()).hexdigest()[:12],16)%m

def cold_split(x):
    for m in (5,4,3):
        vd=x.SMILES.map(lambda s:bucket(s,m)==0); vp=x.Protein.map(lambda s:bucket(s,m)==0)
        tr=x[(~vd)&(~vp)].reset_index(drop=True); va=x[vd&vp].reset_index(drop=True)
        if len(va)>=50 and va.Y.nunique()==2 and tr.Y.nunique()==2: return tr,va
    raise RuntimeError("cannot construct source-only cold-both validation")

def norm_rows(a): return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),EPS)
def robust_fit(a):
    c=np.median(a,axis=0); s=np.maximum(np.quantile(a,.9,axis=0)-np.quantile(a,.1,axis=0),1e-6); return c,s
def robust(a,c,s): return np.nan_to_num(np.clip((a-c)/s,-8,8)).astype(np.float32)
def metrics(y,s): return {"auroc":float(roc_auc_score(y,s)),"auprc":float(average_precision_score(y,s))}

def fps(smiles):
    g2=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    g3=rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048)
    out={k:[] for k in ("m2","m3","maccs","rdk")}
    for z in smiles:
        m=Chem.MolFromSmiles(z); out["m2"].append(g2.GetFingerprint(m)); out["m3"].append(g3.GetFingerprint(m))
        out["maccs"].append(MACCSkeys.GenMACCSKeys(m)); out["rdk"].append(RDKFingerprint(m,fpSize=2048))
    return out

def bulk(q,ref): return np.vstack([np.asarray(DataStructs.BulkTanimotoSimilarity(z,ref),np.float32) for z in q])

@dataclass(frozen=True)
class Cfg:
    pk:str; dk:str; joint:str; k:int; frag:float=0.; gauss:float=0.
    @property
    def name(self): return f"P={self.pk}|D={self.dk}|J={self.joint}|k={self.k}|F={self.frag}|G={self.gauss}"

def configs():
    out=[]
    for p,d,j,k in product(("hash","hash_group","hash_group_desc"),("m2","multi","multi_desc"),("prod","sqrt","harm"),(50,100,200,400)):
        out.append(Cfg(p,d,j,k))
    for p,d,j,k in product(("hash_group","hash_group_desc"),("multi","multi_desc"),("sqrt","harm"),(100,200,400)):
        for f,g in ((.25,0),(.5,0),(0,.25),(0,.5),(.25,.25),(.5,.25),(.25,.5)): out.append(Cfg(p,d,j,k,f,g))
    return out

class FormulaModel:
    def fit(self,source):
        self.s=prep(source,True); self.y=self.s.Y.to_numpy(int); self.pos=self.y==1; self.neg=~self.pos
        self.ud=pd.Index(self.s.SMILES.unique()); self.up=pd.Index(self.s.Protein.unique())
        dm={z:i for i,z in enumerate(self.ud)}; pm={z:i for i,z in enumerate(self.up)}
        self.di=np.array([dm[z] for z in self.s.SMILES]); self.pi=np.array([pm[z] for z in self.s.Protein])
        self.sf=fps(self.ud)
        self.h=HashingVectorizer(analyzer="char",ngram_range=(3,3),n_features=32768,alternate_sign=False,norm="l2",lowercase=False)
        self.gh=HashingVectorizer(analyzer="char",ngram_range=(3,3),n_features=2048,alternate_sign=False,norm="l2",lowercase=False)
        self.sh=self.h.transform(self.up); self.sg=self.gh.transform(["".join(AA2G[a] for a in z) for z in self.up])
        self.praw=np.vstack([_protein_descriptor(z) for z in self.up]); self.pc,self.ps=robust_fit(self.praw); self.pd=norm_rows(robust(self.praw,self.pc,self.ps))
        self.draw=np.vstack([_drug_descriptor(z) for z in self.ud]); self.dc,self.ds=robust_fit(self.draw); self.dd=norm_rows(robust(self.draw,self.dc,self.ds))
        self._closed_stats(); return self
    def _closed_stats(self):
        cnt=np.ones((2,2048),float)
        for i,d in enumerate(self.di): cnt[self.y[i],list(self.sf["m2"][d].GetOnBits())]+=1
        den=np.array([self.neg.sum(),self.pos.sum()])[:,None]+2; pr=cnt/den
        self.bitllr=np.log((pr[1]+EPS)/(pr[0]+EPS)).astype(np.float32)
        pair=self.sg[self.pi]; c0=np.asarray(pair[self.neg].sum(0)).ravel()+1; c1=np.asarray(pair[self.pos].sum(0)).ravel()+1
        self.kllr=np.log((c1/c1.sum()+EPS)/(c0/c0.sum()+EPS)).astype(np.float32)
        D=robust(self.draw[self.di],self.dc,self.ds); P=robust(self.praw[self.pi],self.pc,self.ps); X=np.c_[D,P]
        self.mu=np.vstack([X[self.y==c].mean(0) for c in (0,1)]); self.va=np.vstack([X[self.y==c].var(0)+.25 for c in (0,1)])
        raw=self._gauss(X); self.gc=np.median(raw); self.gs=max(np.quantile(raw,.9)-np.quantile(raw,.1),1e-4)
        f=[]
        for d,p in zip(self.di,self.pi):
            b=list(self.sf["m2"][d].GetOnBits()); f.append((self.bitllr[b].mean() if b else 0)+float(self.sg[p]@self.kllr))
        self.fc=np.median(f); self.fs=max(np.quantile(f,.9)-np.quantile(f,.1),1e-4)
    def _gauss(self,X):
        a=[]
        for c in (0,1): a.append(-.5*np.mean(np.log(self.va[c])+(X-self.mu[c])**2/self.va[c],axis=1))
        return a[1]-a[0]
    def query(self,q):
        q=prep(q,False); ud=pd.Index(q.SMILES.unique()); up=pd.Index(q.Protein.unique()); dm={z:i for i,z in enumerate(ud)}; pm={z:i for i,z in enumerate(up)}
        di=np.array([dm[z] for z in q.SMILES]); pi=np.array([pm[z] for z in q.Protein]); qf=fps(ud)
        qh=self.h.transform(up); qg=self.gh.transform(["".join(AA2G[a] for a in z) for z in up])
        praw=np.vstack([_protein_descriptor(z) for z in up]); pd=norm_rows(robust(praw,self.pc,self.ps))
        draw=np.vstack([_drug_descriptor(z) for z in ud]); dd=norm_rows(robust(draw,self.dc,self.ds))
        return q,ud,up,di,pi,qf,qh,qg,praw,pd,draw,dd
    def predict_many(self,q,cfgs):
        q,ud,up,di,pi,qf,qh,qg,praw,pd,draw,dd=self.query(q)
        hs=(qh@self.sh.T).toarray().astype(np.float32); gs=(qg@self.sg.T).toarray().astype(np.float32); ps=np.clip(pd@self.pd.T,0,1)
        pm={"hash":hs,"hash_group":.65*hs+.35*gs,"hash_group_desc":.55*hs+.25*gs+.2*ps}
        m2=bulk(qf["m2"],self.sf["m2"]); multi=(m2+bulk(qf["m3"],self.sf["m3"])+bulk(qf["maccs"],self.sf["maccs"]))/3
        dm={"m2":m2,"multi":multi,"multi_desc":.85*multi+.15*np.clip(dd@self.dd.T,0,1)}
        frag=[]
        for d,p in zip(di,pi):
            b=list(qf["m2"][d].GetOnBits()); frag.append((self.bitllr[b].mean() if b else 0)+float(qg[p]@self.kllr))
        frag=np.clip((np.array(frag)-self.fc)/self.fs,-8,8)
        X=np.c_[robust(draw[di],self.dc,self.ds),robust(praw[pi],self.pc,self.ps)]; gau=np.clip((self._gauss(X)-self.gc)/self.gs,-8,8)
        out={}; cache={}
        for c in cfgs:
            key=(c.pk,c.dk,c.joint,c.k)
            if key not in cache:
                score=np.empty(len(q),np.float32); sup=np.empty(len(q),np.float32); ep=np.empty(len(q),np.float32); en=np.empty(len(q),np.float32)
                for st in range(0,len(q),48):
                    ix=np.arange(st,min(st+48,len(q))); P=pm[c.pk][pi[ix]][:,self.pi]; D=dm[c.dk][di[ix]][:,self.di]
                    J=P*D if c.joint=="prod" else (np.sqrt(P*D) if c.joint=="sqrt" else 2*P*D/(P+D+EPS))
                    a=_top_k_mean(J[:,self.pos],c.k); b=_top_k_mean(J[:,self.neg],c.k); ep[ix]=a; en[ix]=b; score[ix]=np.log((a+EPS)/(b+EPS)); sup[ix]=J.max(1)
                cache[key]=(score,sup,ep,en)
            loc,sup,ep,en=cache[key]; final=loc+c.frag*frag+c.gauss*gau
            out[c.name]={"score":final,"local":loc,"fragment":frag,"gaussian":gau,"support":sup,"ep":ep,"en":en,"cfg":c}
        return out

def choose(source):
    tr,va=cold_split(source); cs=configs(); out=FormulaModel().fit(tr).predict_many(va[["SMILES","Protein"]],cs); y=va.Y.to_numpy(int); base=y.mean(); rows=[]
    for c in cs:
        m=metrics(y,out[c.name]["score"]); obj=.5*m["auroc"]+.5*(m["auprc"]-base)/max(1-base,EPS); rows.append({"config":c.name,"objective":obj,**m})
    tab=pd.DataFrame(rows).sort_values(["objective","auroc","auprc"],ascending=False); name=tab.iloc[0].config
    return next(c for c in cs if c.name==name),tab,{"dev_train_n":len(tr),"dev_val_n":len(va),"dev_val_positive_rate":float(base)}

def run(source_path,target_path,dataset,outdir):
    s=prep(pd.read_csv(source_path),True); raw=pd.read_csv(target_path); t=prep(raw,"Y" in raw); cfg,tab,info=choose(s); z=FormulaModel().fit(s).predict_many(t[["SMILES","Protein"]],[cfg])[cfg.name]
    o=raw.copy(); o["formula_score"]=z["score"]; o["p_formula"]=1/(1+np.exp(-np.clip(z["score"],-30,30))); o["pred_label"]=(z["score"]>=0).astype(int)
    o["local_score"]=z["local"]; o["fragment_score"]=z["fragment"]; o["gaussian_score"]=z["gaussian"]; o["joint_support"]=z["support"]; o["evidence_positive"]=z["ep"]; o["evidence_negative"]=z["en"]
    p=Path(outdir); p.mkdir(parents=True,exist_ok=True); o.to_csv(p/f"{dataset}_cluster_predictions.csv",index=False); tab.head(100).to_csv(p/f"{dataset}_source_validation_top100.csv",index=False)
    m=metrics(t.Y.to_numpy(int),z["score"]) if "Y" in t else None
    r={"dataset":dataset,"method":"PureFormula-MKCE","source_n":len(s),"target_n":len(t),"best_config":cfg.name,"config":cfg.__dict__,"source_selection":info,"target_metrics":m,"no_classifier_optimization":True,"no_neural_network":True,"target_labels_used_in_prediction":False}
    (p/f"{dataset}_cluster_metrics.json").write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2)); return r

def main():
    a=argparse.ArgumentParser(); a.add_argument("--source",required=True); a.add_argument("--target",required=True); a.add_argument("--dataset",required=True); a.add_argument("--output-dir",default="formula_scjke/results"); x=a.parse_args(); run(x.source,x.target,x.dataset,x.output_dir)
if __name__=="__main__": main()
