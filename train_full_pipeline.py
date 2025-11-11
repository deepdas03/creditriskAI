import os, sys, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import dump
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

DATA_PATH = "data/bank_loan.csv"
OUT_DIR = "output"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Rows:", len(df), "Columns:", df.shape[1])
print()

if 'loan_status' not in df.columns:
    print("ERROR: expected 'loan_status' column not found. Exiting.")
    sys.exit(1)

print("loan_status dtype:", df['loan_status'].dtype)
print("loan_status value counts:\n", df['loan_status'].value_counts(dropna=False))
df['default_flag'] = df['loan_status'].astype(int)
print("default_flag counts:\n", df['default_flag'].value_counts())
print()

cols = set(df.columns.str.lower())
print("Detected columns (sample 50):", list(df.columns)[:50])
print()

col_map = {
    'age': ['person_age','age','applicant_age'],
    'income': ['person_income','income','applicantincome','monthly_income'],
    'loan_amount': ['loan_amnt','loan_amount','loanamt','loanamount','amount'],
    'loan_int_rate': ['loan_int_rate','int_rate','interest_rate'],
    'loan_percent_income': ['loan_percent_income','loan_percent_income'],
    'credit_history_len':['cb_person_cred_hist_length','cred_hist_length','credit_history_length'],
    'credit_score':['credit_score','fico_score','creditScore'],
    'gender':['person_gender','gender','sex'],
    'education':['person_education','education'],
    'home_ownership':['person_home_ownership','home_ownership','property_area'],
    'loan_purpose':['loan_intent','loan_intent','loan_purpose','purpose','loan_intent'],
    'prev_defaults':['previous_loan_defaults_on_file','previous_defaults','prev_loan_defaults']
}

def find_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in df.columns.str.lower().tolist():
            for real in df.columns:
                if real.lower() == c.lower():
                    return real
    return None

found = {}
for k, lst in col_map.items():
    found[k] = find_col(lst)

print("Mapped columns discovered:")
for k,v in found.items():
    print(f"  {k}: {v}")
print()

if found['loan_amount'] and found['income']:
    df['DTI'] = pd.to_numeric(df[found['loan_amount']], errors='coerce') / (pd.to_numeric(df[found['income']], errors='coerce') + 1)
    print("Created DTI from", found['loan_amount'], "/", found['income'])
else:
    print("DTI not created (loan_amount or income missing)")

if found['loan_amount']:
    df['log_loan_amount'] = np.log1p(pd.to_numeric(df[found['loan_amount']], errors='coerce'))
if found['income']:
    df['log_income'] = np.log1p(pd.to_numeric(df[found['income']], errors='coerce'))

if found['credit_history_len']:
    df['cred_hist_len'] = pd.to_numeric(df[found['credit_history_len']], errors='coerce')

if found['prev_defaults']:
    raw = df[found['prev_defaults']].fillna("").astype(str).str.strip().str.lower()
    truthy = {'yes','y','true','t','1','already','previously','ever','defaulted'}
    falsy  = {'no','n','false','f','0','none','nope','nan','not'}
    def map_prev(x):
        if x in truthy:
            return 1
        if x in falsy or x == "":
            return 0
        try:
            v = float(x)
            return 1 if v != 0 else 0
        except Exception:
            return 0
    df['prev_defaults_flag'] = raw.map(map_prev).astype(int)
    print("Created prev_defaults_flag from", found['prev_defaults'], "value counts:")
    print(df['prev_defaults_flag'].value_counts())

if found['age']:
    df['age'] = pd.to_numeric(df[found['age']], errors='coerce')
    df['age_bucket'] = pd.cut(df['age'], bins=[0,25,35,50,65,120], labels=['<=25','26-35','36-50','51-65','65+'])

if found['loan_int_rate']:
    df['loan_int_rate'] = pd.to_numeric(df[found['loan_int_rate']], errors='coerce')

if found['credit_score']:
    df['credit_score'] = pd.to_numeric(df[found['credit_score']], errors='coerce')

if found['loan_purpose']:
    df['loan_purpose_canon'] = df[found['loan_purpose']].astype(str).str.lower().str.strip()

print("Preview engineered columns (first 5 rows):")
preview_cols = ['DTI','log_loan_amount','log_income','age','age_bucket','loan_int_rate','credit_score','prev_defaults_flag','loan_purpose_canon']
print(df[[c for c in preview_cols if c in df.columns]].head(5).to_string())
print()

numeric_candidates = []
for c in ['age','DTI','log_loan_amount','log_income','loan_int_rate','credit_score','cred_hist_len']:
    if c in df.columns:
        numeric_candidates.append(c)

categorical_candidates = []
for c in ['gender','education','home_ownership','loan_purpose_canon','age_bucket']:
    if c in df.columns:
        categorical_candidates.append(c)

for col in df.columns:
    if col in ('loan_id','loan_status','default_flag'): continue
    if df[col].dtype == object:
        if df[col].nunique() <= 200 and col not in categorical_candidates:
            categorical_candidates.append(col)

num_cols = [c for c in numeric_candidates if c in df.columns]
cat_cols = [c for c in categorical_candidates if c in df.columns]

print("Numeric features used:", num_cols)
print("Categorical features used:", cat_cols)
print()

if len(num_cols) + len(cat_cols) == 0:
    print("ERROR: No usable features detected. Exiting.")
    sys.exit(1)

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])

models = {
    'LogisticRegression': Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'))]),
    'RandomForest': Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))]),
    'GradientBoosting': Pipeline([('pre', preprocessor), ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42))])
}

X = df[num_cols + cat_cols]
y = df['default_flag'].astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running 5-fold CV AUC for candidate models...")
model_scores = {}
for name, pipe in models.items():
    try:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"{name} fold AUCs: {np.round(scores,4)}")
        print(f"{name} mean AUC: {np.nanmean(scores):.4f} ± {np.nanstd(scores):.4f}\n")
        model_scores[name] = (np.nanmean(scores), np.nanstd(scores))
    except Exception as e:
        print(f"{name} failed during CV:", e)
        model_scores[name] = (np.nan, np.nan)

valid = {k:v for k,v in model_scores.items() if not np.isnan(v[0])}
if valid:
    best_name = max(valid.items(), key=lambda kv: kv[1][0])[0]
else:
    best_name = 'RandomForest' if 'RandomForest' in models else list(models.keys())[0]
print("Best model chosen:", best_name)
best_pipe = models[best_name]

print("Fitting best model on full data...")
best_pipe.fit(X, y)

model_path = os.path.join(MODEL_DIR, "pd_model.pkl")
dump(best_pipe, model_path)
print("Saved model to", model_path)

df['PD'] = best_pipe.predict_proba(X)[:,1]

LGD = 0.4
if found['loan_amount']:
    df['EAD'] = pd.to_numeric(df[found['loan_amount']], errors='coerce').fillna(0.0)
else:
    df['EAD'] = 10000.0
df['LGD'] = LGD
df['ECL'] = df['PD'] * df['LGD'] * df['EAD']

out_full = os.path.join(OUT_DIR, "data_with_ecl.csv")
df.to_csv(out_full, index=False)
try:
    df.to_parquet(os.path.join(OUT_DIR, "data_with_ecl.parquet"), index=False)
except Exception:
    pass
print("Saved", out_full)

agg_by = None
for cand in ['loan_purpose_canon', found.get('loan_purpose')]:
    if cand and cand in df.columns:
        agg_by = cand
        break

if agg_by is None:
    if len(cat_cols) > 0:
        agg_by = cat_cols[0]
    else:
        agg_by = None

if agg_by:
    agg = df.groupby(agg_by).agg(total_exposure=('EAD','sum'), avg_PD=('PD','mean'), sum_ECL=('ECL','sum'), count_loans=(found.get('loan_amount') or df.columns[0], 'count')).reset_index().sort_values('sum_ECL', ascending=False)
    out_agg = os.path.join(OUT_DIR, "ecl_by_segment.csv")
    agg.to_csv(out_agg, index=False)
    try:
        agg.to_parquet(os.path.join(OUT_DIR, "ecl_by_segment.parquet"), index=False)
    except Exception:
        pass
    print("Saved aggregated ECL by", agg_by, "to", out_agg)
    print("\nTop 10 segments by ECL:\n", agg.head(10).to_string(index=False))
else:
    print("No categorical column found to aggregate by segments; skipping aggregation file.")

fi = None
if best_name == 'RandomForest':
    try:
        rf = best_pipe.named_steps['clf']
        cat_names = []
        if len(cat_cols) > 0:
            ohe = best_pipe.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
            cat_names = list(ohe.get_feature_names_out(cat_cols))
        feat_names = num_cols + cat_names
        importances = rf.feature_importances_
        fi = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
        fi.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"), index=False)
        # JSON version
        try:
            fi.to_json(os.path.join(OUT_DIR, "feature_importances.json"), orient='records')
        except Exception:
            pass
        print("\nTop feature importances (saved to output/feature_importances.csv):")
        print(fi.head(15).to_string(index=False))
    except Exception as e:
        print("Could not compute RF importances:", e)
else:
    if best_name == 'LogisticRegression':
        try:
            coefs = best_pipe.named_steps['clf'].coef_[0]
            ohe = best_pipe.named_steps['pre'].named_transformers_['cat'].named_steps['ohe'] if len(cat_cols)>0 else None
            cat_names = list(ohe.get_feature_names_out(cat_cols)) if ohe is not None else []
            feat_names = num_cols + cat_names
            fi = pd.DataFrame({'feature': feat_names, 'coef': coefs})
            fi['abs'] = fi['coef'].abs()
            fi = fi.sort_values('abs', ascending=False).drop(columns=['abs'])
            fi.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"), index=False)
            try:
                fi.to_json(os.path.join(OUT_DIR, "feature_importances.json"), orient='records')
            except Exception:
                pass
            print("\nTop logistic coefficients (saved to output/feature_importances.csv):")
            print(fi.head(15).to_string(index=False))
        except Exception as e:
            print("Could not compute logistic coefficients:", e)

def save_report_snapshot(df, agg_df, best_model_name, model_scores, feature_importances_df=None, base_out="output"):
    """
    Saves a timestamped snapshot folder with CSVs and a meta.json for later review.
    Uses existing variables: df (full), agg_df (aggregated), best_model_name, model_scores (dict), feature_importances_df (optional).
    """
    import os, json, shutil
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base_out, "reports", ts)
    os.makedirs(outdir, exist_ok=True)

    data_path = os.path.join(outdir, "data_with_ecl.csv")
    agg_path = os.path.join(outdir, "ecl_by_segment.csv")
    df.to_csv(data_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    meta = {
        "timestamp": ts,
        "model_selected": best_model_name,
        "model_scores": {},
        "rows": int(len(df)),
        "segments": int(len(agg_df)),
    }

    try:
        for k, v in (model_scores.items() if isinstance(model_scores, dict) else []):
            mean_auc = None
            std_auc = None
            try:
                mean_auc = float(v[0]) if v[0] is not None and not (v[0] != v[0]) else None
            except Exception:
                mean_auc = None
            try:
                std_auc = float(v[1]) if v[1] is not None and not (v[1] != v[1]) else None
            except Exception:
                std_auc = None
            meta["model_scores"][k] = {"mean_auc": mean_auc, "std_auc": std_auc}
    except Exception:
        meta["model_scores"] = {}

    if feature_importances_df is not None:
        try:
            fi_csv = os.path.join(outdir, "feature_importances.csv")
            feature_importances_df.to_csv(fi_csv, index=False)
            meta["top_features"] = feature_importances_df['feature'].head(10).tolist()
        except Exception:
            meta["top_features"] = []

    try:
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print("Warning: could not write meta.json:", e)

    print(f"Saved report snapshot to {outdir}")

try:
    save_report_snapshot(df, agg, best_name, model_scores, feature_importances_df=fi if 'fi' in locals() else None)
except Exception as e:
    print("Could not save report snapshot:", e)

try:
    if fi is not None:
        fi.to_json(os.path.join(OUT_DIR, "feature_importances.json"), orient='records')
except Exception:
    pass

manifest = {
    "model_name": best_name,
    "timestamp": datetime.now().isoformat(),
    "rows": int(len(df)),
    "features": num_cols + cat_cols,
    "model_scores": {k: {"mean_auc": (float(v[0]) if v[0] is not None and not (v[0] != v[0]) else None),
                         "std_auc": (float(v[1]) if v[1] is not None and not (v[1] != v[1]) else None)} for k,v in model_scores.items()}
}
try:
    with open(os.path.join(MODEL_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved model manifest to", os.path.join(MODEL_DIR, "manifest.json"))
except Exception as e:
    print("Could not save model manifest:", e)

print("\nPipeline complete. Files saved under ./output and ./models.")
print("Timestamp:", datetime.now().isoformat())

assignments_path = os.path.join(".", "assignments.json")
if not os.path.exists(assignments_path):
    placeholder = {
        "analyst1": ["education","medical"],
        "analyst2": ["personal","debtconsolidation"],
        "cro1": ["*"]
    }
    try:
        with open(assignments_path, "w") as f:
            json.dump(placeholder, f, indent=2)
        print("Created placeholder assignments.json (edit as needed).")
    except Exception as e:
        print("Could not create assignments.json:", e)
