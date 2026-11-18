# 📊 Data Science & Product Decisions Portfolio

Welcome to my Data Science Portfolio. This repository hosts advanced end-to-end analytics and predictive projects built to drive business growth, optimize product features, and translate statistical findings into actionable business ROI.

---

## 🚀 Projects Included

### 1. [AeroTest: Enterprise A/B Testing & Statistical Engine](ab-testing-platform)
* **Live Demo**: [https://charan-reddy-dev.github.io/Data-Science-Projects/ab-testing-platform/](https://charan-reddy-dev.github.io/Data-Science-Projects/ab-testing-platform/)
* **Core Jupyter Notebook**: [ab_testing_analysis.ipynb](ab-testing-platform/notebooks/ab_testing_analysis.ipynb)
* **Domain**: Product Analytics, Growth Data Science, Statistics.
* **Key Components**:
  * Pre-experiment sample size calculator (power analysis).
  * Hypothesis testing (Two-Proportions Z-Test).
  * Sample Ratio Mismatch (SRM) Chi-Square Goodness-of-Fit detector to prevent data tracking bias.
  * Payback duration and visual business ROI projections.

### 2. [ChurnIntel: Schema-Grounded Text-to-SQL Copilot & SHAP Churn Engine](churn-predictive-agent)
* **Live Demo** (client-side illustration): [https://charan-reddy-dev.github.io/Data-Science-Projects/churn-predictive-agent/](https://charan-reddy-dev.github.io/Data-Science-Projects/churn-predictive-agent/)
* **Engine & results**: [churn-predictive-agent/README.md](churn-predictive-agent/README.md), [benchmarks/results.json](churn-predictive-agent/benchmarks/results.json)
* **Domain**: Text-to-SQL, SQL safety, Classification, Explainable ML.
* **Key Components**:
  * Text-to-SQL copilot grounded in live SQLite schema introspection: Gemini / OpenAI-compatible providers with a rule-based fallback, bound parameters only.
  * Three-layer read-only sandbox (sqlparse validator, SQLite authorizer, read-only connection): 22/22 attack queries blocked, 8/8 legitimate controls allowed.
  * LightGBM churn model (test ROC-AUC 0.8271 on synthetic data) with per-customer SHAP drivers written back for the copilot to query.

### 3. [SentinelShield: Real-Time Financial Fraud Detection Engine & Drift Tracker](realtime-fraud-detector)
* **Live Demo**: [https://charan-reddy-dev.github.io/Data-Science-Projects/realtime-fraud-detector/](https://charan-reddy-dev.github.io/Data-Science-Projects/realtime-fraud-detector/)
* **Core Jupyter Notebook**: [fraud_detection_streaming.ipynb](realtime-fraud-detector/notebooks/fraud_detection_streaming.ipynb)
* **Domain**: Streaming Event Processing, Cybersecurity, Outlier Isolation.
* **Key Components**:
  * Real-time transaction outlier scoring using a client-side Isolation Forest path length simulation.
  * Kolmogorov-Smirnov (KS) statistical test distribution modeling for active feature drift detection.
  * Interactive SMOTE calibration panel over-sampling minor fraud events to balance model boundaries.

### 4. [LendingClub: Loan Portfolio Risk & Default Prediction](loan_portfolio)
* **Live Demo**: [https://charan-reddy-dev.github.io/Data-Science-Projects/loan_portfolio/](https://charan-reddy-dev.github.io/Data-Science-Projects/loan_portfolio/)
* **Core Jupyter Notebooks**: [data_analytics_capstone.ipynb](loan_portfolio/notebooks/data_analytics_capstone.ipynb), [loan_portfolio_detailed_self.ipynb](loan_portfolio/notebooks/loan_portfolio_detailed_self.ipynb)
* **Domain**: Credit Risk Modeling, Default Prediction, SHAP Explainability.
* **Key Components**:
  * Exploratory Data Analysis (EDA) on loan distributions, interest rate trends, and borrower metrics.
  * Supervised binary classification modeling to differentiate between fully paid and charged-off loans.
  * Model interpretability with SHAP summary, beeswarm plots, and Partial Dependence Plots (PDP).
  * Portfolio return optimization and loan selection strategies for investors.

---

## 🛠️ Technology Stack & Libraries Used
* **Languages**: Python, JavaScript (ES6+), SQL, HTML5/CSS3.
* **ML & Statistics**: NumPy, Pandas, Scikit-learn (Ensemble methods, Logistic Regression), Scipy, Statsmodels.
* **Data Visualization**: Matplotlib, Seaborn, Chart.js, SVG Animations.
* **In-Memory Querying**: AlaSQL (in-browser database execution engine).
