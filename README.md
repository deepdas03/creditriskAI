# 💳 Credit Risk AI — Intelligent Loan Risk Analyzer and ECL Dashboard  

A modern **Streamlit-based AI application** for financial risk management, this project helps banking analysts visualize credit risk exposure, estimate Expected Credit Loss (ECL), and receive AI-driven recommendations on adjusting lending or interest strategies.  

---

## 🌟 Overview

Banks and NBFCs often face the challenge of identifying and managing high-risk portfolios while maintaining profitability.  
This project — **Credit Risk AI** — bridges that gap by providing a data-driven, interactive dashboard powered by **machine learning** and **AI-based explanations**.

It enables:
- Dynamic analysis of loan portfolios.  
- Visual ECL curve generation.  
- Role-based user dashboards (Analyst / CRO).  
- AI assistant that provides recommendations on **interest rate adjustments** or **loan disbursement reduction** based on ECL patterns.

---

## 🧠 Key Features

### 🔹 **1. Interactive Risk Analysis**
- Upload or analyze built-in portfolio datasets.
- Visualize risk categories and loan distribution.
- Compute Expected Credit Loss (ECL) and compare across sectors.

### 🔹 **2. AI Decision Support**
- Integrated **ChatGPT-like Assistant** gives suggestions:
  - Whether to **increase interest rate** or **reduce disbursement**.
  - Explains the rationale using ECL curve trends.

### 🔹 **3. Role-Based Access**
- Supports multiple user types:
  - **Analyst**: can analyze and filter sectors (e.g., Education, Medical, Personal Loans).
  - **CRO (Chief Risk Officer)**: can access full portfolio (`*` wildcard access).
- Secure login system using hashed credentials (`users.json` and `assignments.json`).

### 🔹 **4. ECL Visualization**
- Plots ECL curve for multiple segments.
- Highlights risk thresholds and policy boundaries.
- Interactive threshold sliders let users dynamically simulate policy changes.

### 🔹 **5. Upload Custom Datasets**
- Users can upload `.csv` datasets to analyze new portfolios in real time.
- The app automatically computes risk variations, recalculates ECL, and generates updated recommendations.

### 🔹 **6. Adjustable Threshold Controls**
- Modify **high-risk** and **medium-risk** thresholds.
- Observe how verdicts change (e.g., "Increase interest" vs "Reduce disbursement").

---

## 🏗️ Project Architecture

```bash
credit-risk-ai/
├── streamlit_app.py                # Main Streamlit application
├── model/                          # ML models and ECL computation logic
│   ├── ecl_model.pkl               # Pretrained or saved model
│   └── risk_analysis.py            # Core functions
├── data/
│   ├── labeled_c_dataset.csv       # Default dataset
│   └── sample_upload.csv           # Example file for testing upload
├── .streamlit/
│   └── secrets.toml                # API keys and secrets (HF_TOKEN, etc.)
├── users.json                      # Encrypted user credentials
├── assignments.json                # Role-based access definition
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python runtime version (e.g., python-3.10.12)
├── README.md                       # You are here!
└── report/
    ├── Credit_Risk_AI_Report.docx  # Detailed methodology report
    └── Credit_Risk_AI_Presentation.pptx  # Project presentation slides
```


```bash
| Component                | Technology                          |
| ------------------------ | ----------------------------------- |
| **Frontend / Dashboard** | Streamlit (v1.26.1)                 |
| **Backend / ML Engine**  | Python 3.10+                        |
| **Visualization**        | Plotly                              |
| **Data Handling**        | Pandas, NumPy                       |
| **Machine Learning**     | Scikit-learn                        |
| **AI Assistant**         | Hugging Face Transformers / Flan-T5 |
| **Storage**              | JSON (user roles), CSV (datasets)   |
| **Deployment**           | Streamlit Cloud (Community)         |
```


### 4️⃣ **User Roles**
- **Analyst**: Restricted access to assigned loan sectors.  
- **CRO (Chief Risk Officer)**: Full access to all datasets (`*` wildcard).  
- Role-based login with encrypted credentials.

### 5️⃣ **Upload and Analyze**
- Users can upload new `.csv` files.
- The app automatically recalculates metrics and regenerates graphs.

### 6️⃣ **Threshold Simulation**
- Adjust high and medium-risk thresholds dynamically.  
- View how different risk ratios affect policy recommendations.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend / Dashboard** | Streamlit (v1.26.1) |
| **Backend / Engine** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Machine Learning** | Scikit-learn |
| **Authentication** | Hashed credentials (users.json) |
| **Deployment** | Streamlit Cloud |
| **Storage** | CSV, JSON |

---

## 🧩 Methodology

### 1️⃣ Dataset Preparation  
- The base dataset (`labeled_c_dataset.csv`) includes:
- Loan amount, purpose, PD, EAD, LGD, credit score, etc.
- Preprocessing steps:
- Cleaning missing data.
- Encoding categorical variables.
- Normalizing numerical features.

### 2️⃣ ECL Computation  
**Expected Credit Loss (ECL)** is calculated using:  
\[
ECL = EAD × PD × LGD
\]

Where:
- **EAD (Exposure at Default)** → Amount exposed if the borrower defaults.  
- **PD (Probability of Default)** → Likelihood that the borrower defaults.  
- **LGD (Loss Given Default)** → % of loss after recovery efforts.  

Each segment’s ECL is aggregated to estimate **total portfolio loss**.

---

### 3️⃣ Risk Segmentation

| Risk Level | Condition | Recommended Action |
|-------------|------------|--------------------|
| High Risk | ECL/Exposure > 0.025 | Reduce disbursement, increase pricing |
| Medium Risk | 0.01 < ECL/Exposure ≤ 0.025 | Tighten underwriting |
| Low Risk | ECL/Exposure ≤ 0.01 | Maintain or expand lending |

The app recalculates these ratios dynamically when the user adjusts thresholds in the sidebar.

---

### 4️⃣ Visualization Workflow

| Visualization | Description |
|----------------|--------------|
| **ECL by Segment** | Bar chart showing ECL distribution across loan types. |
| **PD Distribution** | Histogram for probability of default across borrowers. |
| **ECL Curve** | Line chart: cumulative ECL vs cumulative exposure. |
| **Segment Drilldown** | Scatter plot: PD vs EAD with bubble size = ECL. |

🌐 **Live App:** [https://creditriskaii.streamlit.app/](https://creditriskaii.streamlit.app/)
