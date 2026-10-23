"""استدلال مباشر من نموذج CatBoost المحفوظ؛ لا يستخدم التنبؤات التاريخية."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "models" / "inference_settings.json"

def predict_from_dataframe(frame, model_kind="baseline"):
    """حساب قيم جديدة من ميزات المديرية الـ27، دون قراءة عمود الهدف."""
    cfg=json.loads(SETTINGS.read_text(encoding="utf-8"))
    cols=cfg["feature_names"]
    missing=[c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"ميزات ناقصة: {missing}")
    data=frame[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    x=data.fillna(pd.Series(cfg["imputer_medians"],index=cols)).to_numpy(dtype=float)
    model=CatBoostRegressor()
    model.load_model(str(ROOT/"models/EXP_0008_catboost_27features_full.cbm"))
    pred_base=np.clip(model.predict(x),0,100)
    if model_kind == "baseline":
        return pred_base
    if model_kind == "blend":
        logit_model=CatBoostRegressor()
        logit_model.load_model(str(ROOT/"models/EXP_0012_catboost_logit_full.cbm"))
        pred_logit=np.clip(100*expit(logit_model.predict(x)),0,100)
        alpha=float(cfg["final_blend_weight_logit"])
        return np.clip((1-alpha)*pred_base+alpha*pred_logit,0,100)
    raise ValueError("model_kind must be baseline or blend")

def main():
    parser=argparse.ArgumentParser(description="تنبؤ نسبة الوصول إلى المستشفيات من المدخلات")
    parser.add_argument("--input",default=str(ROOT/"data/processed/selected_27_features_target_folds.csv"))
    parser.add_argument("--output",default=str(ROOT/"outputs/predictions.csv"))
    parser.add_argument("--model",choices=["baseline","blend"],default="baseline")
    args=parser.parse_args()
    df=pd.read_csv(args.input,dtype={"ADM2_PCODE":"string","ADM1_PCODE":"string"})
    result=pd.DataFrame(index=df.index)
    for code in ("ADM2_PCODE","ADM1_PCODE"):
        if code in df.columns: result[code]=df[code]
    result["predicted_hospital_access_pct"]=predict_from_dataframe(df,args.model)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(target,index=False)
    print(f"أنشأنا {len(result)} تنبؤًا جديدًا في: {target}")
    print(result.head().to_string(index=False))
if __name__=="__main__": main()
