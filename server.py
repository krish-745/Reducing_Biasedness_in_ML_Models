from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import operator
from typing import Optional, List

from diagnostics import detect_proxy, check_ratio, check_data_desert, check_intersection
from preprocessing.reweighing import reweighing
from preprocessing.disparte_impact_recovery import DisparateImpactRemover
from preprocessing.optimized import OptimizedPreprocessor
from postprocessing.equalized_odds import get_roc_curves, find_fair_operating_point
from postprocessing.reject_option_classification import reject_option_classification
from afs import FairnessAuditor

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = FastAPI(title="Bias Diagnostic Clinic API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  SHARED UTILITIES
# ─────────────────────────────────────────────
ops = {
    '>=': operator.ge, '<=': operator.le,
    '>':  operator.gt, '<':  operator.lt,
    '==': operator.eq, '!=': operator.ne,
}

def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    df_obj = df.select_dtypes(['object'])
    if not df_obj.empty:
        df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    return df

def create_binned_column(df: pd.DataFrame, col: str, priv_val: str, unpriv_val: str) -> str:
    """Converts a raw column into Privileged / Unprivileged / Other via the binner."""
    temp_col = f"{col}_binned"
    df[temp_col] = "Other"
    col_numeric = pd.to_numeric(df[col], errors='coerce')

    def get_mask(condition):
        cond_str = str(condition).strip()
        if not cond_str:
            return pd.Series([False] * len(df), index=df.index)
        for op_str, op_func in ops.items():
            if cond_str.startswith(op_str):
                try:
                    val = float(cond_str[len(op_str):].strip())
                    return op_func(col_numeric, val) & col_numeric.notna()
                except ValueError:
                    pass
        try:
            return col_numeric == float(cond_str)
        except ValueError:
            return df[col].astype(str).str.strip() == cond_str

    df.loc[get_mask(unpriv_val), temp_col] = "Unprivileged"
    df.loc[get_mask(priv_val), temp_col] = "Privileged"
    return temp_col

def encode_for_sklearn(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns for scikit-learn."""
    df_enc = df.copy()
    le = LabelEncoder()
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object':
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc

def compute_afs_from_data(df: pd.DataFrame, target_col: str, binned_prot_col: str) -> dict:
    """
    Trains a Logistic Regression on a 80/20 split and computes the full AFS report.
    Returns the raw report dict from FairnessAuditor.calculate_afs().
    """
    df_enc = encode_for_sklearn(df)
    X = df_enc.drop(columns=[target_col])
    y = df_enc[target_col]
    protected = df_enc[binned_prot_col]

    X_train, X_test, y_train, y_test, attr_train, attr_test = train_test_split(
        X, y, protected, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Priv class = most common value in the protected attribute
    priv_val = int(attr_test.mode().iloc[0])
    unpriv_vals = [v for v in attr_test.unique() if v != priv_val]
    unpriv_val = int(unpriv_vals[0]) if unpriv_vals else priv_val

    auditor = FairnessAuditor(
        y_true=y_test.values,
        y_pred=y_pred,
        protected_attr=attr_test.values,
        priv_class=priv_val,
        unpriv_class=unpriv_val,
        X_features=X_test.values,
    )
    report = auditor.calculate_afs()
    report["accuracy"] = round(accuracy * 100, 1)
    return report

def safe_serialize(obj):
    """Recursively convert numpy types to Python native for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_serialize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ─────────────────────────────────────────────
#  REQUEST MODELS
# ─────────────────────────────────────────────
class DiagnosticRequest(BaseModel):
    csv_data: str
    config: dict

class PreprocessRequest(BaseModel):
    csv_data: str
    config: dict
    algorithm: str                          # 'reweighing' | 'dir' | 'optimized'
    algorithm_params: Optional[dict] = {}

class PostprocessRequest(BaseModel):
    csv_data: str
    config: dict
    algorithm: str                          # 'equalized_odds' | 'reject_option'
    algorithm_params: Optional[dict] = {}


# ─────────────────────────────────────────────
#  ENDPOINT 1: DIAGNOSTICS (unchanged logic)
# ─────────────────────────────────────────────
@app.post("/api/diagnostics")
async def run_diagnostics(req: DiagnosticRequest):
    df = strip_strings(pd.read_csv(io.StringIO(req.csv_data)))
    cfg = req.config
    target = cfg['target_col']
    raw_fav = cfg['fav_outcome']
    fav_outcome = int(raw_fav) if str(raw_fav).isdigit() else str(raw_fav).strip()

    results = {}

    for prot in cfg['prot_cols']:
        mapping = cfg.get('group_mappings', {}).get(prot, {})
        raw_priv   = mapping.get('priv',   '')
        raw_unpriv = mapping.get('unpriv', '')

        binned_prot = create_binned_column(df, prot, raw_priv, raw_unpriv)

        priv_count   = (df[binned_prot] == "Privileged").sum()
        unpriv_count = (df[binned_prot] == "Unprivileged").sum()

        if priv_count == 0 or unpriv_count == 0:
            err = (f"Zero rows matched! Found {priv_count} Privileged and "
                   f"{unpriv_count} Unprivileged. Check your group definitions.")
            results[prot] = {k: {"error": err} for k in
                             ['disparate_impact', 'data_desert', 'proxy', 'intersection']}
            continue

        # 1. Disparate Impact
        di_ratio, is_legal = check_ratio(
            df, binned_prot, target, "Privileged", "Unprivileged",
            fav_outcome, cfg['threshold_45']
        )
        attr_results = {
            'disparate_impact': {
                "ratio":     None if pd.isna(di_ratio) else float(di_ratio),
                "is_legal":  bool(is_legal),
                "threshold": cfg['threshold_45'],
            }
        }

        # 2. Data Desert
        desert_df = check_data_desert(df, binned_prot, cfg['desert_thresh'])
        desert_list = desert_df.reset_index().rename(columns={'index': 'group'}).to_dict('records')
        attr_results['data_desert'] = {
            "threshold": cfg['desert_thresh'],
            "groups": [{"group": row[binned_prot], "proportion": row["proportion"],
                        "is_desert": row["is_desert"]} for row in desert_list],
        }

        # 3. Proxy Radar
        df_proxy = df.drop(columns=[prot]).copy()
        for col in df_proxy.columns:
            if df_proxy[col].dtype == 'object':
                df_proxy[col] = df_proxy[col].fillna("Missing")
            else:
                df_proxy[col] = df_proxy[col].fillna(-9999)

        proxy_df = detect_proxy(df_proxy, binned_prot, cfg['proxy_thresh'], target)
        attr_results['proxy'] = {
            "proxies": proxy_df.reset_index()
                               .rename(columns={'index': 'feature'})
                               .to_dict('records') if not proxy_df.empty else []
        }

        # 4. Intersectional
        inter_cols = list(dict.fromkeys([binned_prot] + cfg.get('sec_prots', [])))
        gap, is_gap, grp = check_intersection(df, inter_cols, target, fav_outcome, threshold=0.2)
        attr_results['intersection'] = {
            "gap":    float(gap),
            "is_gap": bool(is_gap),
            "groups": safe_serialize(grp.to_dict('records')),
        }

        results[prot] = safe_serialize(attr_results)

    return results


# ─────────────────────────────────────────────
#  ENDPOINT 2: PRE-PROCESSING MITIGATION
# ─────────────────────────────────────────────
@app.post("/api/preprocess")
async def run_preprocess(req: PreprocessRequest):
    df = strip_strings(pd.read_csv(io.StringIO(req.csv_data)))
    cfg   = req.config
    target = cfg['target_col']
    prot   = cfg['prot_cols'][0] if cfg.get('prot_cols') else None

    if not prot:
        return {"error": "No protected attribute selected."}

    mapping    = cfg.get('group_mappings', {}).get(prot, {})
    raw_priv   = mapping.get('priv',   '')
    raw_unpriv = mapping.get('unpriv', '')

    df_work   = df.copy()
    binned_col = create_binned_column(df_work, prot, raw_priv, raw_unpriv)

    # ── Baseline AFS ──────────────────────────────
    try:
        baseline = compute_afs_from_data(df_work, target, binned_col)
    except Exception as exc:
        return {"error": f"Baseline AFS computation failed: {exc}"}

    # ── Run selected algorithm ─────────────────────
    algo   = req.algorithm
    params = req.algorithm_params or {}

    numeric_cols = [
        c for c in df_work.select_dtypes(include=[np.number]).columns
        if c != target and c != binned_col
    ]

    try:
        if algo == 'reweighing':
            df_mitigated = reweighing(df_work.copy(), binned_col, target)
            df_for_afs   = df_mitigated.drop(columns=['weight'], errors='ignore')

        elif algo == 'dir':
            repair_level = float(params.get('repair_level', 0.8))
            remover      = DisparateImpactRemover(repair_level=repair_level)
            df_for_afs   = remover.fit_transform(df_work.copy(), binned_col, numeric_cols)

        elif algo == 'optimized':
            e = float(params.get('epsilon', 0.05))
            distortion  = float(params.get('distortion', 3.0))
            bins_count  = int(params.get('bins', 4))
            use_cols = numeric_cols[:2]
            optimizer_cols = list(dict.fromkeys(use_cols + [target, binned_col]))
            df_opt = df_work[optimizer_cols].copy()

            engine = OptimizedPreprocessor(prot=[binned_col], target=target, e=e, distortion=distortion)
            engine.fit(df_opt.copy(), cols=use_cols, bins=bins_count)
            df_for_afs  = engine.transform(df_opt.copy(), cols=use_cols)

        else:
            return {"error": f"Unknown algorithm: '{algo}'"}

        mitigated = compute_afs_from_data(df_for_afs, target, binned_col)

    except Exception as exc:
        return {"error": f"Mitigation failed: {exc}"}

    return safe_serialize({
        "baseline":  {
            "afs":               baseline['Aggregated_Fairness_Score'],
            "accuracy":          baseline.get('accuracy', 0),
            "verdict":           baseline['Verdict'],
            "metrics":           baseline['Metrics'],
            "normalized_scores": baseline['Normalized_Scores'],
        },
        "mitigated": {
            "afs":               mitigated['Aggregated_Fairness_Score'],
            "accuracy":          mitigated.get('accuracy', 0),
            "verdict":           mitigated['Verdict'],
            "metrics":           mitigated['Metrics'],
            "normalized_scores": mitigated['Normalized_Scores'],
        },
        "delta":     mitigated['Aggregated_Fairness_Score'] - baseline['Aggregated_Fairness_Score'],
        "algorithm": algo,
    })


# ─────────────────────────────────────────────
#  ENDPOINT 3: POST-PROCESSING MITIGATION
# ─────────────────────────────────────────────
@app.post("/api/postprocess")
async def run_postprocess(req: PostprocessRequest):
    """
    Expects a CSV that contains:
      - true_label column  (binary: 0 / 1)
      - score column       (probability 0–1)
      - protected attribute column
    """
    df = strip_strings(pd.read_csv(io.StringIO(req.csv_data)))
    cfg        = req.config
    target     = cfg['target_col']       # column of true labels
    score_col  = cfg.get('score_col', 'predicted_score')

    prot = cfg['prot_cols'][0] if cfg.get('prot_cols') else None
    if not prot:
        return {"error": "No protected attribute selected."}

    mapping    = cfg.get('group_mappings', {}).get(prot, {})
    raw_priv   = mapping.get('priv',   '')
    raw_unpriv = mapping.get('unpriv', '')

    df_work    = df.copy()
    binned_col = create_binned_column(df_work, prot, raw_priv, raw_unpriv)

    y_true    = pd.to_numeric(df_work[target],    errors='coerce').fillna(0).astype(int)
    y_scores  = pd.to_numeric(df_work[score_col], errors='coerce').fillna(0.5)
    y_pred    = (y_scores >= 0.5).astype(int)
    protected = df_work[binned_col].map({'Privileged': 1, 'Unprivileged': 0}).fillna(0).astype(int)

    # ── Baseline AFS ──────────────────────────────
    try:
        auditor_base = FairnessAuditor(
            y_true=y_true.values, y_pred=y_pred.values,
            protected_attr=protected.values, priv_class=1, unpriv_class=0,
        )
        baseline_report = auditor_base.calculate_afs()
        baseline_report['accuracy'] = round(accuracy_score(y_true, y_pred) * 100, 1)
    except Exception as exc:
        return {"error": f"Baseline AFS computation failed: {exc}"}

    # ── Run selected algorithm ─────────────────────
    algo   = req.algorithm
    params = req.algorithm_params or {}

    try:
        if algo == 'equalized_odds':
            curves      = get_roc_curves(y_true.values, y_scores.values, protected.values)
            fair_points = find_fair_operating_point(curves)
            y_pred_new  = y_pred.values.copy()
            for g, point in fair_points.items():
                mask = (protected == g).values
                y_pred_new[mask] = (y_scores.values[mask] >= point['threshold']).astype(int)

        elif algo == 'reject_option':
            threshold = float(params.get('threshold', 0.5))
            margin    = float(params.get('margin', 0.15))
            priv_mask   = (protected == 1)
            unpriv_mask = (protected == 0)
            df_roc = df_work.copy()
            df_roc[score_col] = y_scores
            result_df  = reject_option_classification(
                df_roc, score_col, unpriv_mask, priv_mask, threshold, margin
            )
            y_pred_new = result_df['fair_decision'].values

        else:
            return {"error": f"Unknown algorithm: '{algo}'"}

        auditor_new = FairnessAuditor(
            y_true=y_true.values, y_pred=y_pred_new,
            protected_attr=protected.values, priv_class=1, unpriv_class=0,
        )
        mitigated_report = auditor_new.calculate_afs()
        mitigated_report['accuracy'] = round(accuracy_score(y_true, y_pred_new) * 100, 1)

    except Exception as exc:
        return {"error": f"Post-processing failed: {exc}"}

    return safe_serialize({
        "baseline":  {
            "afs":               baseline_report['Aggregated_Fairness_Score'],
            "accuracy":          baseline_report.get('accuracy', 0),
            "verdict":           baseline_report['Verdict'],
            "metrics":           baseline_report['Metrics'],
            "normalized_scores": baseline_report['Normalized_Scores'],
        },
        "mitigated": {
            "afs":               mitigated_report['Aggregated_Fairness_Score'],
            "accuracy":          mitigated_report.get('accuracy', 0),
            "verdict":           mitigated_report['Verdict'],
            "metrics":           mitigated_report['Metrics'],
            "normalized_scores": mitigated_report['Normalized_Scores'],
        },
        "delta":     mitigated_report['Aggregated_Fairness_Score'] - baseline_report['Aggregated_Fairness_Score'],
        "algorithm": algo,
    })


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
