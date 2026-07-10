# SentinelShield - Real-Time Financial Fraud Detection Engine & Drift Tracker

SentinelShield is an advanced real-time financial fraud detection simulation and statistical monitoring platform. It demonstrates how to handle extreme class imbalance (99% normal, 1% fraud), process streaming event pipelines, and detect active feature drift in production data.

### 🌐 Live Demo & Repository
- **Interactive Live Dashboard**: Open `index.html` directly in any browser (hosted on GitHub Pages).
- **Core Streaming Notebook**: [Jupyter Notebook: PySpark & Isolation Forest Pipeline](notebooks/fraud_detection_streaming.ipynb)

---

## 🚀 Key Features

### 1. Simulated Live Streaming Ingestion (Apache Kafka Representation)
* Simulates a continuous stream of financial transactions (Card ID, Amount, Location, Time, Category, and Risk Score).
* Displays live Kafka ingestion metrics: events per second, pipeline latency (ms), and consumer group offset lag.

### 2. Anomaly Detection Engine (Isolation Forest & SMOTE)
* **Outlier Isolation**: Uses a simulated client-side Isolation Forest scoring algorithm to flag anomalies.
* **Imbalance Mitigation**: Shows how **SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic fraud data to train robust decision boundaries under extreme imbalance.
* **Decision Path Visualization**: Displays path lengths for isolating transactions (short paths imply anomalous/fraudulent activity).

### 3. Active Feature Drift Tracker (Evidently AI Representation)
* Tracks statistical changes in feature distributions in real-time.
* Uses the **Kolmogorov-Smirnov (KS) Test** ($\alpha = 0.05$) to compare the distribution of the last 100 transactions against a baseline reference dataset.
* Interactive line and area charts display drifting feature distributions and trigger drift alerts immediately if a user injects a "high-value fraud spike."

---

## 📊 Mathematical & Statistical Methodology

### 1. Isolation Forest Anomaly Scoring
The anomaly score $s$ for an instance $x$ is defined as:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
* $E(h(x))$ is the average path length of $x$ across a collection of isolation trees.
* $c(n)$ is the average path length of an unsuccessful search in a Binary Search Tree (BST) with $n$ nodes:
$$c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n-1)}{n}$$
* **Score Interpretation**:
  * If $s \to 1$: Average path length is very short $\to$ easily isolated $\to$ **High anomaly/fraud risk**.
  * If $s \to 0$: Average path length is long $\to$ deep in trees $\to$ **Normal transaction**.

### 2. Kolmogorov-Smirnov (KS) Test for Feature Drift
To detect if the production feature distribution has drifted from baseline, we calculate the KS statistic $D$:
$$D = \sup_{u} |F_{\text{baseline}}(u) - F_{\text{production}}(u)|$$
* If the p-value associated with $D$ is less than the significance level ($\alpha = 0.05$), we reject the null hypothesis $H_0$ (which states that the two samples come from the same distribution), flagging active **feature drift**.

---

## 🛠️ Technology Stack
* **Dashboard Front-end**: HTML5, Modern CSS3 (Cyberpunk Glassmorphism), JavaScript (ES6+).
* **Charts & Plots**: Chart.js (Real-time distribution area plots & Precision-Recall curves).
* **Data Science / Python**: PySpark Streaming, Apache Kafka, Scikit-Learn (Isolation Forest), Imbalanced-Learn (SMOTE), Evidently AI.
