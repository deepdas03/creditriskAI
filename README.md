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
