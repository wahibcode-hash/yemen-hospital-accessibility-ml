"""إعادة تدريب EXP_0008 وEXP_0012 خارج طيات المحافظات؛ غير متصل بالإنترنت.

جميع الأرقام تُحسب من التدريب الحالي وليست مسترجعة من جدول OOF.
هذا تقييم استكشافي لأن طيات المحافظات فُحصت في التجارب السابقة.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit,expit
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor

ROOT=Path(__file__).resolve().parents[1]
EPS=0.0015
ALPHAS=[0.0,.10,.20,.25,.35,.50,.65,.75,1.0]

def model():
    return CatBoostRegressor(loss_function="MAE",iterations=300,depth=4,
                             learning_rate=.05,l2_leaf_reg=5.0,
                             random_seed=2030,verbose=False,thread_count=4,
                             allow_writing_files=False)

def metrics(y,p):
    return {"MAE_pp":float(mean_absolute_error(y,p)),
            "RMSE_pp":float(mean_squared_error(y,p)**.5),
            "R2":float(r2_score(y,p))}

def run():
    frame=pd.read_csv(ROOT/"data/processed/selected_27_features_target_folds.csv",
                      dtype={"ADM2_PCODE":"string","ADM1_PCODE":"string"})
    settings=json.loads((ROOT/"models/inference_settings.json").read_text(encoding="utf-8"))
    features=settings["feature_names"]
    X=frame[features].apply(pd.to_numeric,errors="raise")
    y=frame["hospital_access_pct"].to_numpy(float)
    groups=frame["ADM1_PCODE"].astype(str).to_numpy()
    folds=frame["outer_fold"].to_numpy(int)
    if not (len(X)==335 and len(np.unique(groups))==22 and len(features)==27):
        raise ValueError("البيانات أو مخطط الميزات لا يطابق الإصدار المعتمد")
    if not np.isfinite(X.to_numpy()).all(): raise ValueError("قيم ميزات غير منتهية")
    base=np.full(len(y),np.nan); blend=np.full(len(y),np.nan); fold_rows=[]
    # في كل طية تُحجب محافظات كاملة عن التدريب.
    for fold in range(1,6):
        tr=np.flatnonzero(folds!=fold); te=np.flatnonzero(folds==fold)
        assert not (set(groups[tr])&set(groups[te]))
        imp=SimpleImputer(strategy="median")
        xt=imp.fit_transform(X.iloc[tr]); xv=imp.transform(X.iloc[te])
        baseline=model();baseline.fit(xt,y[tr])
        base[te]=np.clip(baseline.predict(xv),0,100)
        print(f"المرجع — طية {fold}: MAE={mean_absolute_error(y[te],base[te]):.4f} pp")
    for fold in range(1,6):
        tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
        inner_base=np.full(len(tr),np.nan);inner_logit=np.full(len(tr),np.nan)
        for fit_local,val_local in GroupKFold(3).split(X.iloc[tr],y[tr],groups[tr]):
            fit_idx=tr[fit_local];val_idx=tr[val_local]
            assert not (set(groups[fit_idx])&set(groups[val_idx]))
            imp=SimpleImputer(strategy="median"); xf=imp.fit_transform(X.iloc[fit_idx]);xv=imp.transform(X.iloc[val_idx])
            b=model();b.fit(xf,y[fit_idx]);inner_base[val_local]=np.clip(b.predict(xv),0,100)
            z=model();z.fit(xf,logit(np.clip(y[fit_idx]/100,EPS,1-EPS)))
            inner_logit[val_local]=np.clip(100*expit(z.predict(xv)),0,100)
        # هنا فقط: نختار الوزن من بيانات تدريب الطية الخارجية.
        score={a:mean_absolute_error(y[tr],(1-a)*inner_base+a*inner_logit) for a in ALPHAS}
        alpha=min(score,key=score.get)
        imp=SimpleImputer(strategy="median");xf=imp.fit_transform(X.iloc[tr]);xv=imp.transform(X.iloc[te])
        z=model();z.fit(xf,logit(np.clip(y[tr]/100,EPS,1-EPS)))
        logit_te=np.clip(100*expit(z.predict(xv)),0,100)
        blend[te]=np.clip((1-alpha)*base[te]+alpha*logit_te,0,100)
        fold_rows.append({"fold":fold,"selected_alpha":alpha,"n_test":len(te)})
        print(f"التحسين — طية {fold}: alpha={alpha:.2f}, MAE={mean_absolute_error(y[te],blend[te]):.4f} pp")
    if not np.isfinite(base).all() or not np.isfinite(blend).all(): raise RuntimeError("نقص في التنبؤات")
    results={"status":"EXPLORATORY_NOT_INDEPENDENT", "EXP_0008":metrics(y,base),
             "EXP_0012":metrics(y,blend),"folds":fold_rows}
    out=ROOT/"outputs";out.mkdir(exist_ok=True)
    pd.DataFrame({"ADM2_PCODE":frame.ADM2_PCODE,"ADM1_PCODE":frame.ADM1_PCODE,
                  "actual_pct":y,"outer_fold":folds,"EXP_0008_OOF":base,"EXP_0012_OOF":blend}).to_csv(out/"retrained_OOF.csv",index=False)
    (out/"retrained_metrics.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(results,ensure_ascii=False,indent=2))
    return results
if __name__=="__main__":run()
