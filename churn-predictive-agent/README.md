# ChurnIntel - Predictive Customer Intelligence Platform & GenAI SQL Copilot

ChurnIntel is an interactive customer intelligence platform built to illustrate supervised machine learning prediction pipelines and conversational AI agent concepts. It bridges the gap between predictive modeling and prescriptive retention marketing.

### 🌐 Live Demo & Repository
- **Interactive Live Dashboard**: Open `index.html` directly in any browser (deployable on GitHub Pages).
- **Core Modeling Notebook**: [Jupyter Notebook: ML Modeling & Business Impact](notebooks/churn_modeling.ipynb)

---

## 🚀 Key Features

### 1. Real-time Churn Forecasting Engine
- Adjust subscriber attributes (tenure, monthly charges, customer support interactions, contracts) to get real-time churn predictions.
- Powered by a client-side decision threshold model calibrated against Gradient Boosting outputs.
- Displayed using an interactive **Radial Gauge** that dynamically updates colors based on risk severity (Low, Moderate, High).

### 2. explainable AI (SHAP-like Contribution List)
- Provides immediate transparency into *why* the model assigned a risk percentage.
- Breaks down feature impacts: displays positive factors (e.g. Month-to-Month contracts, low tenure, high support calls) and negative factors (e.g. two-year contracts, long tenure) that pull the risk score.

### 3. GenAI SQL Copilot (Conversational Agent Simulator)
- Translates natural language business questions into precise SQL queries.
- Executes queries against an in-memory database of 100 customer records.
- Displays the compiled SQL code, standard table outputs, and an analytical summary recommending retention actions.

### 4. Proactive Retention Campaign Simulator
- Simulates the cost-benefit analysis of sending incentives (e.g. voucher discounts) to identified high-risk customer segments.
- Estimates recovered customer lifetime value (LTV), campaign expenses, net savings, and Return on Investment (ROI).

---

## 📊 Classification Model & scoring Formula

The prediction probability is calculated using an optimized logistic mapping function modeling customer risks:
$$Z = -0.05 \cdot \text{Tenure} + 0.025 \cdot \text{Charges} + 0.65 \cdot \text{Calls} + 1.3 \cdot I_{\text{MonthToMonth}} - 0.9 \cdot I_{\text{TwoYear}} - 1.8$$
$$\text{Probability} = \frac{1}{1 + e^{-Z}}$$

### Feature Weights and Interpretations:
- **Contract Type**: Month-to-Month contracts have a large positive weight ($+1.3$), indicating high instability, whereas Two-Year contracts have a negative weight ($-0.9$), showing high retention.
- **Customer Support Calls**: High weights ($+0.65$ per call), showing that poor customer service experience is a massive churn driver.
- **Tenure**: A negative multiplier ($-0.05$ per month), showing customer loyalty grows over time.

---

## 🛠️ Technology Stack
- **Core UI**: HTML5 / JavaScript (ES6+).
- **Styling**: Cyberpunk-inspired modern dark theme (CSS3 Grid, SVG Radial animations, responsive chat bubbles).
- **Data Science / Python**: Pandas, NumPy, Scikit-learn, Ensemble Methods (Random Forest, Gradient Boosting), Matplotlib, Seaborn.
