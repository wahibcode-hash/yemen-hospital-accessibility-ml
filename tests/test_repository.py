import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def test_frozen_dataset_and_group_isolation():
    d=pd.read_csv(ROOT/'data/processed/selected_27_features_target_folds.csv')
    assert len(d)==335 and d.ADM2_PCODE.is_unique and d.ADM1_PCODE.nunique()==22
    for fold in range(1,6):
        tr=d[d.outer_fold!=fold];te=d[d.outer_fold==fold]
        assert not set(tr.ADM1_PCODE)&set(te.ADM1_PCODE)

def test_features_are_target_free():
    cfg=json.loads((ROOT/'models/inference_settings.json').read_text(encoding='utf8'))
    assert len(cfg['feature_names'])==27
    assert not {'ADM1_PCODE','ADM2_PCODE','hospital_access_pct','outer_fold'}.intersection(cfg['feature_names'])

def test_data_in_splits_consistent():
    d=pd.read_csv(ROOT/'data/processed/selected_27_features_target_folds.csv')
    for fold in range(1,6):
        te=pd.read_csv(ROOT/'data/splits'/f'fold_{fold:02}'/'test_features_ONLY.csv')
        assert len(te)==(d.outer_fold==fold).sum()
        assert 'hospital_access_pct' not in te.columns
