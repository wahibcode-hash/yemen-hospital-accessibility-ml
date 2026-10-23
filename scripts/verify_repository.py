"""اختبارات سريعة لجودة المستودع واتساق المدخلات والنماذج."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.predict import predict_from_dataframe

def main():
    cfg=json.loads((ROOT/'models/inference_settings.json').read_text(encoding='utf-8'))
    d=pd.read_csv(ROOT/'data/processed/selected_27_features_target_folds.csv',dtype={'ADM2_PCODE':'string','ADM1_PCODE':'string'})
    assert len(d)==335 and d.ADM2_PCODE.nunique()==335 and d.ADM1_PCODE.nunique()==22
    assert len(cfg['feature_names'])==27 and len(cfg['imputer_medians'])==27
    assert not set(cfg['feature_names'])&{'ADM1_PCODE','ADM2_PCODE','outer_fold','hospital_access_pct'}
    assert np.isfinite(d[cfg['feature_names']].to_numpy(float)).all()
    for k in range(1,6):
        tr=d[d.outer_fold!=k];te=d[d.outer_fold==k]
        assert len(te)>0 and not set(tr.ADM1_PCODE)&set(te.ADM1_PCODE)
        folder=ROOT/'data/splits'/f'fold_{k:02}'
        assert len(pd.read_csv(folder/'train_labeled.csv'))==len(tr)
        assert len(pd.read_csv(folder/'test_features_ONLY.csv'))==len(te)
        assert 'hospital_access_pct' not in pd.read_csv(folder/'test_features_ONLY.csv',nrows=1).columns
    baseline=predict_from_dataframe(d,'baseline')
    blended=predict_from_dataframe(d,'blend')
    assert baseline.shape==(335,) and blended.shape==(335,)
    assert np.isfinite(baseline).all() and np.isfinite(blended).all()
    assert ((0<=baseline)&(baseline<=100)).all() and ((0<=blended)&(blended<=100)).all()
    # التدقيق العلمي: نعيد احتساب MAE من OOF المحفوظ، وليس نتائج التدريب الكامل.
    oof=pd.read_csv(ROOT/'reports/out_of_fold_predictions.csv',dtype={'ADM2_PCODE':'string'})
    joined=d[['ADM2_PCODE','hospital_access_pct']].merge(
        oof[['ADM2_PCODE','EXP_0008_OOF_retrained','EXP_0012_OOF_retrained']],on='ADM2_PCODE',validate='one_to_one')
    mae_b=np.abs(joined.hospital_access_pct-joined.EXP_0008_OOF_retrained).mean()
    mae_c=np.abs(joined.hospital_access_pct-joined.EXP_0012_OOF_retrained).mean()
    assert abs(mae_b-13.45094577387978)<1e-7
    assert abs(mae_c-12.888233844497057)<1e-7
    print(f'PASS: 335 districts, 22 governorates, 27 features, 5 isolated folds; OOF MAE {mae_b:.4f} / {mae_c:.4f}')
    print(f'PASS: Saved full-fit CatBoost models loaded and predicted {len(baseline)} rows (not an independent evaluation).')
if __name__=='__main__':main()
