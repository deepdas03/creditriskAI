# streamlit_app.py
"""
Streamlit dashboard for Credit Risk assignment.
- Login + Sign Up (writes users.json)
- Role-based UI: Analyst / CRO (assignments.json enforces allowed segments)
- Dashboard: KPIs, ECL-by-segment, PD distribution, ECL curve
- Report History: browse and download timestamped snapshots (output/reports)
- Assistant: rule-based recommendation + optional HF LLM explanation
- Logs: audit trail at output/chat_logs.csv

Changes included:
- Manual CSV/Parquet upload (sidebar). Upload replaces dataframe in-session and can be saved to output/.
- verify_user now prefers users.json on disk for persistence (st.secrets still supported as fallback).
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import hashlib
import re

# Local helper: snapshot_list.py should export list_snapshots, get_snapshot_files
from snapshot_list import list_snapshots, get_snapshot_files

# Paths and config
PROJECT_ROOT = Path(".")
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"
USERS_FILE = PROJECT_ROOT / "users.json"
ASSIGNMENTS_FILE = PROJECT_ROOT / "assignments.json"
CHAT_LOG = OUTPUT_DIR / "chat_logs.csv"

st.set_page_config(page_title="Credit Risk — ECL Dashboard", layout="wide")

# ----------------- Utility / I/O -----------------
def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(exist_ok=True)
    (OUTPUT_DIR / "plans").mkdir(exist_ok=True)

ensure_dirs()

def log_action(user, role, action_type, details):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user,
        "role": role,
        "action_type": action_type,
        "details": json.dumps(details)
    }
    try:
        if not CHAT_LOG.exists():
            pd.DataFrame([entry]).to_csv(CHAT_LOG, index=False)
        else:
            df = pd.read_csv(CHAT_LOG)
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            df.to_csv(CHAT_LOG, index=False)
    except Exception:
        # don't crash app on logging failure
        pass

# ----------------- Users / Auth (persistent) -----------------
def make_hash(password: str, salt: str):
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def load_users_from_file():
    """
    Always prefer reading users.json from disk so created accounts persist across app restarts.
    """
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_users_to_file(users: dict):
    # write atomically
    tmp = USERS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(USERS_FILE)

def verify_user(username, password):
    """
    Verify user credentials.
    Prioritize users.json on disk (so app-created users persist).
    Fallback to st.secrets['users'] when file not present.
    Returns: (ok:bool, role_or_none)
    """
    # Try disk
    users = load_users_from_file()
    # If no users on disk, fallback to secrets
    if not users:
        try:
            raw = st.secrets.get("users", None)
            if raw:
                if isinstance(raw, dict):
                    users = raw
                else:
                    users = json.loads(raw)
        except Exception:
            users = {}
    if username not in users:
        return False, None
    entry = users[username]
    salt = entry.get("salt")
    expected = entry.get("hash")
    if not salt or not expected:
        return False, None
    return (make_hash(password, salt) == expected), entry.get("role")

def signup_user(username: str, password: str, role: str, auto_assign=True):
    """
    Create user in users.json on disk. Ensures persistence.
    """
    users = load_users_from_file()
    if username in users:
        return False, "username_exists"
    salt = os.urandom(8).hex()
    hashed = make_hash(password, salt)
    users[username] = {"salt": salt, "hash": hashed, "role": role}
    try:
        save_users_to_file(users)
    except Exception as e:
        return False, f"save_failed:{e}"
    # create assignments for analysts/cro
    if role == "Analyst" and auto_assign:
        assign_default_segments_to_user(username)
    elif role == "CRO":
        assign_entire_access(username)
    return True, "created"

# ----------------- Assignments helpers -----------------
def load_assignments():
    if ASSIGNMENTS_FILE.exists():
        try:
            with open(ASSIGNMENTS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {u: [s.lower().strip() for s in segs] if isinstance(segs, list) else [] for u,segs in raw.items()}
        except Exception:
            return {}
    return {}

def save_assignments(assignments: dict):
    tmp = ASSIGNMENTS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(ASSIGNMENTS_FILE)

def assign_entire_access(username):
    assignments = load_assignments()
    assignments[username] = ["*"]
    save_assignments(assignments)

def assign_default_segments_to_user(username, n=2):
    assignments = load_assignments()
    try:
        agg_path = OUTPUT_DIR / "ecl_by_segment.csv"
        if agg_path.exists():
            agg = pd.read_csv(agg_path)
            seg_col = agg.columns[0]
            top_segments = [str(s).lower().strip() for s in agg.head(n)[seg_col].astype(str).tolist()]
            if len(top_segments) == 0:
                top_segments = ["education","medical"]
        else:
            top_segments = ["education","medical"]
    except Exception:
        top_segments = ["education","medical"]
    assignments[username] = top_segments
    save_assignments(assignments)

# ----------------- Data loading / Upload -----------------
@st.cache_data(ttl=600)
def _load_disk_data():
    df_path = OUTPUT_DIR / "data_with_ecl.parquet"
    csv_path = OUTPUT_DIR / "data_with_ecl.csv"
    agg_par = OUTPUT_DIR / "ecl_by_segment.parquet"
    agg_csv = OUTPUT_DIR / "ecl_by_segment.csv"

    if df_path.exists():
        df = pd.read_parquet(df_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return None, None

    if agg_par.exists():
        agg = pd.read_parquet(agg_par)
    elif agg_csv.exists():
        agg = pd.read_csv(agg_csv)
    else:
        seg_col = detect_segment_col(df)
        agg = df.groupby(seg_col).agg(total_exposure=('EAD','sum'), avg_PD=('PD','mean'), sum_ECL=('ECL','sum'), count_loans=('PD','count')).reset_index().sort_values('sum_ECL', ascending=False)
    return df, agg

def load_data(prefer_uploaded=True):
    """
    Load dataset. Behavior:
    - If user uploaded a dataset during session (st.session_state['uploaded_df']) and prefer_uploaded True, use that.
    - Else, load from output/data_with_ecl.csv (or parquet).
    Returns (df, agg)
    """
    # Use in-session uploaded dataset if present
    if prefer_uploaded and st.session_state.get("uploaded_df") is not None:
        df = st.session_state["uploaded_df"]
        # attempt to compute agg if uploaded_agg present else create
        if st.session_state.get("uploaded_agg") is not None:
            agg = st.session_state["uploaded_agg"]
            return df, agg
        # create agg if possible
        try:
            seg_col = detect_segment_col(df)
            if "EAD" in df.columns and "ECL" in df.columns and "PD" in df.columns:
                agg = df.groupby(seg_col).agg(total_exposure=('EAD','sum'), avg_PD=('PD','mean'), sum_ECL=('ECL','sum'), count_loans=('PD','count')).reset_index().sort_values('sum_ECL', ascending=False)
            else:
                # try to create EAD/ECL columns if loan amount present
                agg = pd.DataFrame()
        except Exception:
            agg = pd.DataFrame()
        return df, agg

    # else load from disk (cached)
    df, agg = _load_disk_data()
    if df is None:
        st.error("Missing data outputs. Run the training pipeline first (creates output/data_with_ecl.csv and output/ecl_by_segment.csv), or upload a CSV/Parquet on the sidebar.")
        st.stop()
    return df, agg

def detect_segment_col(df):
    for c in ["loan_purpose_canon","loan_intent","loan_purpose","purpose"]:
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            if df[c].dtype == object and 2 <= df[c].nunique() <= 200:
                return c
        except Exception:
            continue
    return df.columns[0]

def try_load_uploaded_file(uploaded_file):
    """
    Accept uploaded_file from st.file_uploader. Returns (df, agg, error)
    """
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            return None, None, f"Unsupported file type: {uploaded_file.name}"
    except Exception as e:
        return None, None, f"Could not read uploaded file: {e}"
    # attempt to compute ECL/PD/EAD if already present
    agg = None
    try:
        seg_col = detect_segment_col(df)
        if all(c in df.columns for c in ["EAD","ECL","PD"]):
            agg = df.groupby(seg_col).agg(total_exposure=('EAD','sum'), avg_PD=('PD','mean'), sum_ECL=('ECL','sum'), count_loans=('PD','count')).reset_index().sort_values('sum_ECL', ascending=False)
        else:
            # Not enough columns to create full agg; leave agg empty (caller can warn)
            agg = pd.DataFrame()
    except Exception:
        agg = pd.DataFrame()
    return df, agg, None

# ----------------- Analytics helpers -----------------
def build_ecl_curve(df_segment):
    df2 = df_segment.copy()
    if "ECL" not in df2.columns or "EAD" not in df2.columns:
        return pd.DataFrame(columns=["cum_exposure_pct","cum_ecl_pct"])
    df2 = df2.sort_values("ECL", ascending=False).reset_index(drop=True)
    df2["cum_exposure"] = df2["EAD"].cumsum()
    df2["cum_ecl"] = df2["ECL"].cumsum()
    total_ead = df2["EAD"].sum() + 1e-9
    total_ecl = df2["ECL"].sum() + 1e-9
    df2["cum_exposure_pct"] = df2["cum_exposure"] / total_ead
    df2["cum_ecl_pct"] = df2["cum_ecl"] / total_ecl
    return df2

def recommend_by_segment(total_exposure, sum_ecl, avg_pd, count_loans, high_thresh=0.025, med_thresh=0.01, min_loans=50):
    metrics = {"total_exposure": float(total_exposure), "sum_ECL": float(sum_ecl), "avg_PD": float(avg_pd), "count_loans": int(count_loans)}
    metrics["risk_ratio"] = metrics["sum_ECL"] / (metrics["total_exposure"] + 1e-9)
    if metrics["count_loans"] < min_loans:
        return {"action":"Monitor (small sample)","risk_level":"small_sample","rationale":f"Only {metrics['count_loans']} loans (<{min_loans}); manual review recommended.","metrics":metrics}
    r = metrics["risk_ratio"]
    if r > high_thresh:
        return {"action":"Reduce disbursement and increase interest","risk_level":"high","rationale":f"High ECL/exposure ({r:.4%}). Reduce new exposure and increase pricing.","metrics":metrics}
    if r > med_thresh:
        return {"action":"Tighten underwriting and consider pricing uplift","risk_level":"medium","rationale":f"Moderate ECL/exposure ({r:.4%}). Tighten underwriting or modest pricing uplift.","metrics":metrics}
    return {"action":"Maintain or expand (low risk)","risk_level":"low","rationale":f"Low ECL/exposure ({r:.4%}). Maintain current policies.","metrics":metrics}

# ----------------- Hugging Face LLM (Router) -----------------
def get_hf_token_and_model():
    token = None
    model = None
    try:
        token = st.secrets.get("HF_TOKEN", None)
        model = st.secrets.get("HF_MODEL", None)
    except Exception:
        token = None
        model = None
    if not token:
        token = os.getenv("HF_TOKEN")
    if not model:
        model = os.getenv("HF_MODEL")
    if not model:
        model = "google/flan-t5-large"
    return token, model

def call_hf_inference(prompt: str, model: str = None, token: str = None, temperature: float = 0.0, max_new_tokens: int = 300):
    """
    Use Hugging Face Router endpoint.
    Returns (text, error).
    """
    if token is None or model is None:
        tkn, mdl = get_hf_token_and_model()
        token = token or tkn
        model = model or mdl

    if not token:
        return None, "no_hf_token"

    base_url = "https://router.huggingface.co/hf-inference"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    candidates = [model, "gpt2", "distilgpt2"]
    last_err = None

    for m in candidates:
        if not m:
            continue
        payload = {
            "model": m,
            "inputs": prompt,
            "parameters": {
                "temperature": float(temperature),
                "max_new_tokens": int(max_new_tokens),
                "return_full_text": False
            },
            "options": {"wait_for_model": True}
        }
        try:
            r = requests.post(base_url, headers=headers, json=payload, timeout=60)
            if r.status_code == 410:
                last_err = f"410: model {m} gone"
                continue
            if r.status_code == 401:
                return None, "unauthorized: invalid HF_TOKEN"
            if r.status_code == 403:
                return None, f"forbidden: token lacks inference permission for model {m}"
            if not r.ok:
                last_err = f"{r.status_code}: {r.text[:400]}"
                continue

            out = r.json()
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"], None
                if isinstance(first, str):
                    return first, None
                return json.dumps(out), None
            if isinstance(out, dict):
                if "generated_text" in out:
                    return out["generated_text"], None
                if "outputs" in out and isinstance(out["outputs"], list):
                    pieces = []
                    for o in out["outputs"]:
                        if isinstance(o, dict) and "generated_text" in o:
                            pieces.append(o["generated_text"])
                        elif isinstance(o, str):
                            pieces.append(o)
                    if pieces:
                        return "\n".join(pieces), None
                if out.get("error"):
                    last_err = out.get("error")
                    continue
                if "text" in out and isinstance(out["text"], str):
                    return out["text"], None
                return json.dumps(out), None
            return str(out), None
        except Exception as e:
            last_err = str(e)
            continue

    return None, f"All tried models failed. Last error: {last_err}"

# ----------------- Local Assistant -----------------
def build_local_action_plan(metrics):
    rr = metrics.get("risk_ratio", 0.0)
    count = metrics.get("count_loans", 0)
    plan = {"verdict":"", "actions":[], "monitoring":[], "notes":[]}
    if count < 50:
        plan["verdict"] = "Small sample — monitor and collect more data"
        plan["actions"] = [
            "Increase sampling audits for this segment",
            "Collect additional borrower attributes",
            "Delay policy changes until sample grows"
        ]
        plan["monitoring"] = [
            "Monthly sample size growth",
            "Manual review rate (weekly)",
            "Avg PD trend (monthly)"
        ]
        return plan
    if rr > 0.025:
        plan["verdict"] = "High risk — reduce disbursement and increase pricing"
        plan["actions"] = [
            "Pause approvals for top-risk subsegments",
            "Increase interest rates for new loans (200-400bps)",
            "Require additional collateral for borderline cases"
        ]
        plan["monitoring"] = [
            "Weekly PD trend",
            "Daily ECL / exposure ratio",
            "Approval-to-default ratio (weekly)"
        ]
    elif rr > 0.01:
        plan["verdict"] = "Medium risk — tighten underwriting and consider pricing uplift"
        plan["actions"] = [
            "Raise credit score floor for approvals",
            "Require income verification for borderline cases",
            "Introduce modest pricing uplift (50-150bps)"
        ]
        plan["monitoring"] = [
            "Bi-weekly PD trend",
            "Approval vs default (monthly)",
            "Avg DTI of new approvals (weekly)"
        ]
    else:
        plan["verdict"] = "Low risk — maintain or selectively expand"
        plan["actions"] = [
            "Maintain pricing and limits",
            "Target marketing to low-risk profiles",
            "Monitor concentration by borrower"
        ]
        plan["monitoring"] = [
            "Monthly PD stability check",
            "Quarterly ECL vs exposure review",
            "Top-10 borrower concentration (monthly)"
        ]
    if metrics.get("avg_PD", 0) > 0.2:
        plan["notes"].append("Avg PD > 20% — urgent underwriting review advised")
    return plan

# ----------------- UI -----------------
st.title("📊 Credit Risk — ECL Dashboard & Assistant (with manual upload + persistent users)")

# ----------------- Sidebar: Upload + Auth -----------------
st.sidebar.header("Dataset & User")

# Manual upload widget (allows switching dataset for analysis)
uploaded_file = st.sidebar.file_uploader("Upload dataset (.csv or .parquet) to analyze (optional)", type=["csv","parquet"])
if uploaded_file:
    df_up, agg_up, err = try_load_uploaded_file(uploaded_file)
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success(f"Loaded uploaded file: {uploaded_file.name} ({len(df_up):,} rows)")
        # store in session so other pages use it
        st.session_state["uploaded_df"] = df_up
        st.session_state["uploaded_agg"] = agg_up
        # option to save uploaded dataset to output/ (persists)
        if st.sidebar.checkbox("Save uploaded dataset to output/ (replace disk dataset)", value=False):
            try:
                # attempt to save CSV; if parquet original, keep parquet too
                out_csv = OUTPUT_DIR / "data_with_ecl.csv"
                df_up.to_csv(out_csv, index=False)
                # if we computed agg_up, save it
                if agg_up is not None and not agg_up.empty:
                    agg_up.to_csv(OUTPUT_DIR / "ecl_by_segment.csv", index=False)
                st.sidebar.success("Saved uploaded dataset to output/ (data_with_ecl.csv).")
                log_action("system", "system", "save_uploaded_dataset", {"file": uploaded_file.name})
                # invalidate cached disk load
                _load_disk_data.clear()
            except Exception as e:
                st.sidebar.error(f"Could not save uploaded file: {e}")

# ----------------- Authentication (SignUp/Login) -----------------
auth_mode = st.sidebar.radio("Auth", ["Login", "Sign Up"], index=0)

if auth_mode == "Sign Up":
    st.sidebar.subheader("Create a new account")
    new_user = st.sidebar.text_input("Choose username", key="su_user")
    new_pass = st.sidebar.text_input("Choose password", type="password", key="su_pass")
    new_role = st.sidebar.selectbox("Role", ["Analyst", "CRO"], key="su_role")
    if st.sidebar.button("Create account"):
        if not new_user or not new_pass:
            st.sidebar.warning("Provide username and password")
        else:
            ok, msg = signup_user(new_user.strip(), new_pass, new_role)
            if ok:
                st.sidebar.success(f"User {new_user} created. Signed in.")
                st.session_state["user"] = new_user.strip()
                st.session_state["role"] = new_role
                log_action(new_user.strip(), new_role, "signup", {"status":"success"})
                st.experimental_rerun()
            else:
                if msg == "username_exists":
                    st.sidebar.error("Username already exists. Pick another.")
                else:
                    st.sidebar.error(f"Could not create user: {msg}")

elif auth_mode == "Login":
    st.sidebar.subheader("Sign in")
    username = st.sidebar.text_input("Username", key="login_user")
    password = st.sidebar.text_input("Password", type="password", key="login_pass")
    if st.sidebar.button("Sign in"):
        ok, role = verify_user(username.strip(), password)
        if ok:
            st.session_state["user"] = username.strip()
            st.session_state["role"] = role
            st.sidebar.success(f"Signed in as {username} ({role})")
            log_action(username.strip(), role, "login", {"status":"success"})
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid credentials")

# Require authentication
if "user" not in st.session_state:
    st.info("Please sign in or sign up (sidebar).")
    st.stop()

# ----------------- Page selector / Controls -----------------
page = st.sidebar.selectbox("Page", ["Dashboard", "Report History", "Assistant", "Logs"])

# Load data (uploaded preferred)
df, agg = load_data(prefer_uploaded=True)
if agg is None:
    agg = pd.DataFrame()

# Segment column detection
segment_col = agg.columns[0] if (not agg.empty) else (detect_segment_col(df) if df is not None else None)

# Enforce per-user assignments
assignments = load_assignments()
user = st.session_state["user"]
role = st.session_state.get("role", "Analyst")
user_allowed = assignments.get(user, [])

# normalize names for matching
if segment_col:
    agg[segment_col] = agg[segment_col].astype(str).str.lower().str.strip()
    df[segment_col] = df[segment_col].astype(str).str.lower().str.strip()

if user_allowed and "*" not in [s.strip() for s in user_allowed]:
    allowed_set = set([s.lower().strip() for s in user_allowed])
    try:
        agg = agg[agg[segment_col].isin(allowed_set)].reset_index(drop=True)
        df = df[df[segment_col].isin(allowed_set)].reset_index(drop=True)
        st.sidebar.info(f"Access restricted to: {', '.join(sorted(list(allowed_set)))}")
    except Exception:
        # if normalization fails, skip filtering but warn
        st.sidebar.warning("Could not enforce assignment filtering due to unexpected segment column structure.")
else:
    pass

# Sidebar controls
st.sidebar.markdown("---")
high_thresh = st.sidebar.number_input("High risk threshold (ECL/exposure)", value=0.025, step=0.005, format="%.4f")
med_thresh = st.sidebar.number_input("Medium risk threshold (ECL/exposure)", value=0.01, step=0.005, format="%.4f")
min_loans = st.sidebar.number_input("Min loans for confident recommendation", value=50, step=10)
use_llm = st.sidebar.checkbox("Enable LLM explanations (Hugging Face)", value=False)
st.sidebar.markdown("---")
st.sidebar.write(f"Data source: `{('uploaded session' if st.session_state.get('uploaded_df') is not None else str(OUTPUT_DIR / 'data_with_ecl.csv'))}`")

# ----------------- Dashboard -----------------
if page == "Dashboard":
    st.header("Portfolio summary")
    try:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total loans", f"{len(df):,}")
        c2.metric("Average PD", f"{df['PD'].mean():.3f}")
        c3.metric("Total ECL", f"{df['ECL'].sum():,.0f}")
        c4.metric("Avg LGD (assumed)", f"{df['LGD'].mean() if 'LGD' in df.columns else 0.4:.2f}")
    except Exception:
        st.error("Error computing portfolio KPIs. Check data columns (PD, ECL, EAD).")

    st.subheader("ECL by segment")
    if agg.empty:
        st.info("No segments available for your user. Upload a dataset with ECL/EAD/PD or run the training pipeline to generate output files.")
    else:
        fig = px.bar(agg.sort_values("sum_ECL", ascending=False), x=segment_col, y="sum_ECL", hover_data=["total_exposure","avg_PD","count_loans"])
        fig.update_layout(xaxis_tickangle=-45, height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("PD distribution")
    try:
        fig_pd = px.histogram(df, x="PD", nbins=60, title="PD distribution")
        st.plotly_chart(fig_pd, use_container_width=True)
    except Exception:
        st.info("PD distribution not available (missing PD column).")

    st.subheader("Segment drill-down")
    if agg.empty:
        st.warning("No segments to drill down.")
    else:
        seg_choice = st.selectbox("Choose segment", options=agg[segment_col].astype(str).tolist())
        if seg_choice:
            seg_row = agg[agg[segment_col] == str(seg_choice).lower().strip()].iloc[0]
            st.markdown(f"**Segment:** {seg_choice} — loans: {int(seg_row['count_loans'])}, total exposure: {seg_row['total_exposure']:.0f}, avg PD: {seg_row['avg_PD']:.3f}")
            rec = recommend_by_segment(seg_row['total_exposure'], seg_row['sum_ECL'], seg_row.get('avg_PD',0.0), seg_row['count_loans'], high_thresh, med_thresh, min_loans)
            st.markdown("### Rule-based recommendation")
            st.write(f"**Action:** {rec['action']}")
            st.write(f"**Rationale:** {rec['rationale']}")
            st.json(rec["metrics"])

            # HF LLM block if enabled
            if use_llm:
                try:
                    metrics = rec["metrics"]
                    prompt = (
                        f"You are a pragmatic credit risk analyst. Given aggregated metrics:\n"
                        f"- Total exposure: {metrics['total_exposure']:.0f}\n"
                        f"- Sum ECL: {metrics['sum_ECL']:.0f}\n"
                        f"- Average PD: {metrics['avg_PD']:.3f}\n"
                        f"- Count of loans: {metrics['count_loans']}\n"
                        f"- Risk ratio (ECL/exposure): {metrics['risk_ratio']:.4f}\n\n"
                        f"The rule-based recommendation: {rec['action']}\n\n"
                        f"Task: 1) Confirm recommendation in one sentence. 2) Give 3 short, concrete actions. 3) Give 3 monitoring items. Use numbers from above and be concise. Output JSON with keys: verdict, actions (list), monitoring (list), rationale."
                    )
                    token, model = get_hf_token_and_model()
                    if not token:
                        st.warning("HF_TOKEN not found in secrets or environment. Add it to `.streamlit/secrets.toml` or environment.")
                    else:
                        with st.spinner("Calling Hugging Face Inference API..."):
                            text, err = call_hf_inference(prompt, model=model, token=token, temperature=0.0, max_new_tokens=300)
                        if err:
                            st.error(f"Hugging Face call failed: {err}")
                            log_action(user, role, "assistant_hf_failed", {"segment": seg_choice, "error": err})
                        else:
                            parsed = None
                            try:
                                parsed = json.loads(text)
                            except Exception:
                                m = re.search(r'\{.*\}', text, re.DOTALL)
                                if m:
                                    try:
                                        parsed = json.loads(m.group(0))
                                    except Exception:
                                        parsed = None
                            if parsed is None:
                                st.warning("Model did not return clean JSON. Showing raw output:")
                                st.code(text)
                                log_action(user, role, "assistant_hf_raw", {"segment": seg_choice, "raw": text})
                            else:
                                st.subheader("LLM Action Plan (HF)")
                                st.write("**Verdict:**", parsed.get("verdict", "—"))
                                st.write("**Rationale:**", parsed.get("rationale", "—"))
                                st.write("**Actions:**")
                                for a in parsed.get("actions", []):
                                    st.write("-", a)
                                st.write("**Monitoring items:**")
                                for m in parsed.get("monitoring", []):
                                    st.write("-", m)
                                log_action(user, role, "assistant_hf", {"segment": seg_choice, "parsed": parsed})
                except Exception as e:
                    st.error(f"HF LLM block failed: {e}")
            else:
                st.info("LLM is disabled. Toggle 'Enable LLM explanations (Hugging Face)' in the sidebar to call HF.")

            if role == "Analyst":
                if st.button("Apply recommendation (simulate)"):
                    log_action(user, role, "apply_recommendation", {"segment": seg_choice, "action": rec["action"], "metrics": rec["metrics"]})
                    st.success("Recommendation applied and logged.")

            seg_df = df[df[segment_col] == str(seg_choice).lower().strip()]
            st.markdown("#### PD vs EAD (sample)")
            if not seg_df.empty:
                samp = seg_df.sample(min(len(seg_df), 2000), random_state=42)
                fig2 = px.scatter(samp, x="EAD", y="PD", size="ECL", hover_data=[c for c in ['loan_amnt','loan_amount'] if c in samp.columns])
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("#### Cumulative ECL vs Cumulative Exposure (ECL curve)")
                ecl_df = build_ecl_curve(seg_df)
                fig3 = px.line(ecl_df, x="cum_exposure_pct", y="cum_ecl_pct", title="ECL curve")
                fig3.update_layout(xaxis_title="Cumulative exposure (%)", yaxis_title="Cumulative ECL (%)", height=450)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No loan rows for this segment.")

# ----------------- Assistant (dedicated page) -----------------
elif page == "Assistant":
    st.header("Action Assistant — ECL curve + advice")

    st.caption("Debug: data sizes and sample segments (for troubleshooting).")
    try:
        st.write("Rows in full data:", len(df))
        st.write("Rows in aggregated segments (agg):", len(agg))
        sample_segments = agg[segment_col].astype(str).unique().tolist()[:20] if not agg.empty else []
        st.write("Sample segments (first 20):", sample_segments)
    except Exception as e:
        st.error(f"Error reading debug info: {e}")

    if agg is None or agg.empty:
        st.warning("No segments available to show in Assistant. Possible causes:\n"
                   "• output/ecl_by_segment.csv is missing or empty (run the training pipeline),\n"
                   "• assignments.json filtered out all segments for your user.\n\n"
                   "Upload a dataset with EAD/ECL/PD or open Report History to load a snapshot.")
        if st.button("Open Report History"):
            st.experimental_set_query_params(page="Report History")
        st.stop()

    options = sorted(list(dict.fromkeys([str(s).lower().strip() for s in agg[segment_col].astype(str).tolist()])))
    if not options:
        st.warning("No segment options available after normalization. Check assignments.json and segment names.")
        st.stop()

    seg_choice = st.selectbox("Choose segment", options=options)
    if not seg_choice:
        st.info("No segment selected.")
        st.stop()

    seg_choice_norm = str(seg_choice).lower().strip()

    seg_row = None
    try:
        mask = agg[segment_col].astype(str).str.lower().str.strip() == seg_choice_norm
        if mask.any():
            seg_row = agg[mask].iloc[0]
        else:
            candidates = agg[segment_col].astype(str).tolist()
            match = next((s for s in candidates if seg_choice_norm in str(s).lower()), None)
            if match:
                seg_row = agg[agg[segment_col].astype(str) == match].iloc[0]
    except Exception:
        seg_row = None

    if seg_row is None:
        st.error("Selected segment not found in aggregated data. Available segments shown above.")
        st.stop()

    st.subheader(f"Segment summary: {seg_choice_norm}")
    st.write(f"Loans: **{int(seg_row.get('count_loans',0)):,}**, Total exposure: **{seg_row.get('total_exposure',0):.0f}**, Avg PD: **{seg_row.get('avg_PD',0):.4f}**, Sum ECL: **{seg_row.get('sum_ECL',0):.0f}**")

    rec = recommend_by_segment(seg_row['total_exposure'], seg_row['sum_ECL'], seg_row.get('avg_PD',0.0), seg_row['count_loans'], high_thresh, med_thresh, min_loans)
    st.markdown("### Rule-based recommendation")
    st.write(f"**Action:** {rec['action']}")
    st.write(f"**Rationale:** {rec['rationale']}")
    st.json(rec["metrics"])

    st.markdown("### ECL curve controls")
    n_top = st.slider("Number of top loans to show on curve (sorted by ECL)", min_value=100, max_value=5000, value=2000, step=100)
    seg_df = df[df[segment_col].astype(str).str.lower().str.strip() == seg_choice_norm].copy()
    if seg_df.empty:
        st.info("No loan rows for this segment in the loaded dataset.")
    else:
        ecl_df = build_ecl_curve(seg_df)
        plot_df = ecl_df.head(n_top)
        fig = px.line(plot_df, x="cum_exposure_pct", y="cum_ecl_pct", title=f"ECL Curve — {seg_choice_norm}", labels={"cum_exposure_pct":"Cumulative exposure (%)", "cum_ecl_pct":"Cumulative ECL (%)"})
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50% exposure", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True, height=480)

        st.markdown("Top loan contributions to ECL (top 50 by ECL)")
        top50 = seg_df.sort_values("ECL", ascending=False).head(50)
        if not top50.empty:
            fig2 = px.bar(top50.reset_index(drop=True).reset_index().rename(columns={"index":"rank"}), x="rank", y="ECL", hover_data=["EAD","PD"])
            st.plotly_chart(fig2, use_container_width=True, height=320)

    st.markdown("### Action plan (LLM or local) ")
    st.write("Use the local deterministic assistant (no external calls) or toggle Hugging Face LLM.")

    use_hf = use_llm
    metrics = rec["metrics"]

    if use_hf:
        token, model = get_hf_token_and_model()
        if not token:
            st.warning("HF_TOKEN not found — falling back to local assistant. Add HF_TOKEN to `.streamlit/secrets.toml` to enable HF calls.")
            plan = build_local_action_plan(metrics)
        else:
            prompt = (
                f"You are a pragmatic credit risk analyst. Given aggregated metrics:\n"
                f"- Total exposure: {metrics['total_exposure']:.0f}\n"
                f"- Sum ECL: {metrics['sum_ECL']:.0f}\n"
                f"- Average PD: {metrics['avg_PD']:.3f}\n"
                f"- Count of loans: {metrics['count_loans']}\n"
                f"- Risk ratio (ECL/exposure): {metrics['risk_ratio']:.4f}\n\n"
                f"The rule-based recommendation: {rec['action']}\n\n"
                f"Task: 1) Confirm recommendation in one sentence. 2) Give 3 short, concrete actions. 3) Give 3 monitoring items. Output JSON with keys: verdict, actions (list), monitoring (list), rationale."
            )
            with st.spinner("Calling Hugging Face Inference..."):
                text, err = call_hf_inference(prompt, model=model, token=token, temperature=0.0, max_new_tokens=300)
            if err:
                st.error(f"Hugging Face call failed: {err}")
                log_action(user, role, "assistant_hf_failed", {"segment": seg_choice_norm, "error": err})
                plan = build_local_action_plan(metrics)
            else:
                parsed = None
                try:
                    parsed = json.loads(text)
                except Exception:
                    m = re.search(r'\{.*\}', text, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group(0))
                        except Exception:
                            parsed = None
                if parsed is None:
                    st.warning("Model did not return clean JSON. Showing raw output:")
                    st.code(text)
                    log_action(user, role, "assistant_hf_raw", {"segment": seg_choice_norm, "raw": text})
                    plan = build_local_action_plan(metrics)
                else:
                    plan = {
                        "verdict": parsed.get("verdict", ""),
                        "actions": parsed.get("actions", []),
                        "monitoring": parsed.get("monitoring", []),
                        "notes": [parsed.get("rationale","")]
                    }
                    log_action(user, role, "assistant_hf", {"segment": seg_choice_norm, "parsed": parsed})
    else:
        plan = build_local_action_plan(metrics)

    st.markdown("#### Verdict")
    st.write(plan.get("verdict", "—"))
    if plan.get("notes"):
        for note in plan.get("notes"):
            st.warning(note)
    st.markdown("#### Actions")
    for i,a in enumerate(plan.get("actions", []), start=1):
        st.write(f"{i}. {a}")
    st.markdown("#### Monitoring")
    for i,m in enumerate(plan.get("monitoring", []), start=1):
        st.write(f"{i}. {m}")

    if st.button("Save action plan as JSON"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = OUTPUT_DIR / "plans" / f"plan_{user}_{seg_choice_norm}_{ts}.json"
        try:
            with open(fname, "w", encoding="utf-8") as fh:
                json.dump({"user":user, "role":role, "segment":seg_choice_norm, "plan":plan, "metrics":metrics, "timestamp": datetime.now().isoformat()}, fh, indent=2)
            log_action(user, role, "save_plan", {"file": str(fname), "segment": seg_choice_norm})
            st.success(f"Saved plan to {fname}")
        except Exception as e:
            st.error(f"Could not save plan: {e}")

# ----------------- Report History -----------------
elif page == "Report History":
    st.header("Report History (snapshots)")
    snaps = list_snapshots()
    if not snaps:
        st.info("No snapshots found under output/reports/. Run training pipeline to create snapshots.")
    else:
        rows = []
        for s in snaps:
            meta = s.get("meta", {})
            rows.append({"timestamp": s["timestamp"], "friendly": s.get("friendly_ts"), "path": s["path"], "model": meta.get("model_selected","-"), "rows": meta.get("rows","-"), "segments": meta.get("segments","-")})
        df_snaps = pd.DataFrame(rows)
        st.dataframe(df_snaps, use_container_width=True)
        sel = st.selectbox("Select snapshot to inspect", df_snaps["timestamp"].tolist())
        if sel:
            info = next((s for s in snaps if s["timestamp"] == sel), None)
            if info:
                st.subheader(f"Snapshot: {sel}")
                st.json(info.get("meta", {}))
                files = get_snapshot_files(info["path"])
                if files.get("error"):
                    st.error("Snapshot folder not found.")
                else:
                    for f in files["files"]:
                        p = Path(files["path"]) / f["name"]
                        st.write(f["name"], f"({f['size_bytes']} bytes)")
                        try:
                            with open(p, "rb") as fh:
                                st.download_button(label=f"Download {f['name']}", data=fh, file_name=f["name"])
                        except Exception:
                            st.info("Could not provide download for this file.")
                    if st.button("Load snapshot into dashboard (preview)"):
                        try:
                            df_preview = pd.read_csv(Path(info["path"]) / "data_with_ecl.csv")
                            agg_preview = pd.read_csv(Path(info["path"]) / "ecl_by_segment.csv")
                            st.markdown("### Snapshot agg preview")
                            st.dataframe(agg_preview.head(200))
                            st.markdown("### Snapshot sample rows")
                            st.dataframe(df_preview.head(200))
                        except Exception as e:
                            st.error(f"Could not load CSVs: {e}")

# ----------------- Logs -----------------
elif page == "Logs":
    st.header("Audit logs")
    if CHAT_LOG.exists():
        try:
            logs = pd.read_csv(CHAT_LOG).sort_values("timestamp", ascending=False)
            if role == "CRO":
                st.dataframe(logs)
            else:
                st.dataframe(logs[logs["user"] == user])
        except Exception as e:
            st.error(f"Could not read logs: {e}")
    else:
        st.info("No logs found yet.")