# AeroTest - Enterprise A/B Testing & Statistical Decision Platform

AeroTest is a statistical decision engine designed for product analysts and growth data scientists. It translates raw experiment metrics into reliable, statistically sound business recommendations, helping companies make product decisions backed by data.

### 🌐 Live Demo & Repository
- **Interactive Live Dashboard**: Open `index.html` directly in any browser (deployable on GitHub Pages).
- **Core Analysis Notebook**: [Jupyter Notebook: Statistical Analysis](notebooks/ab_testing_analysis.ipynb)

---

## 🚀 Key Features

### 1. Sample Size Calculator (Pre-Experiment Design)
- Calculates required sample size per variant using power analysis formulas.
- Prevents under-powered tests (which lead to false negatives) and waste of experimentation traffic.
- Displays an interactive **Required Sample Size vs. Minimum Detectable Effect (MDE)** curve.

### 2. Hypothesis Z-Test & Sample Ratio Mismatch (SRM) Engine
- **Two-Proportions Z-Test**: Checks whether the difference in conversion rates between Control and Treatment is statistically significant.
- **Sample Ratio Mismatch (SRM) Guardrail**: Employs a Chi-Square Goodness-of-Fit test ($\alpha = 0.001$) to detect distribution anomalies, indicating potential randomization issues or tracking bugs.
- **Dynamic Probability Density Functions (PDF)**: Chart.js visualization of the normal distribution curves for both variants, showing overlap and significance levels.

### 3. Business ROI Projections ("Now What?" Analysis)
- Bridges the gap between statistical significance and business outcomes.
- Calculates projected incremental monthly/annual conversions, revenue gains, and Return on Investment (ROI).
- Generates payback timelines and business performance tables.

---

## 📊 Statistical Methodology & Formulas

### 1. Sample Size Calculation
$$n = \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot (p_1(1-p_1) + p_2(1-p_2))}{(p_1 - p_2)^2}$$
- $p_1$ = Baseline conversion rate
- $p_2$ = Target treatment conversion rate (based on absolute or relative MDE)
- $Z_{\alpha/2}$ = Critical value for significance level $\alpha$ (e.g., $1.96$ for $\alpha = 0.05$, two-sided)
- $Z_{\beta}$ = Critical value for power $1-\beta$ (e.g., $0.84$ for $80\%$ power)

### 2. Two-Proportion Z-Test
$$Z = \frac{p_B - p_A}{\sqrt{P_{\text{pooled}}(1 - P_{\text{pooled}})(\frac{1}{n_A} + \frac{1}{n_B})}}$$
- $P_{\text{pooled}} = \frac{c_A + c_B}{n_A + n_B}$ (pooled conversion rate)
- $p_A, p_B$ = Control and Treatment conversion rates
- $c_A, c_B$ = Conversions in control and treatment
- $n_A, n_B$ = Sample sizes of control and treatment

### 3. Sample Ratio Mismatch (SRM) Chi-Square Test
$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i} = \frac{(n_A - E)^2}{E} + \frac{(n_B - E)^2}{E}$$
- $O_i$ = Observed counts ($n_A$ and $n_B$)
- $E_i$ = Expected count ($E = \frac{n_A + n_B}{2}$ for a 50/50 target split)
- Degree of Freedom (df) = 1. p-value is calculated using standard normal distribution symmetry: $p_{\text{SRM}} = 2 \cdot (1 - \Phi(\sqrt{\chi^2}))$. A warning triggers if $p_{\text{SRM}} < 0.001$.

---

## 🛠️ Technology Stack
- **Core**: Vanilla HTML5 / JavaScript (ES6+).
- **Styling**: Modern CSS3 (Dark theme, responsive flex layouts, custom visual indicators).
- **Visualization**: Chart.js (Interactive line charts).
- **Data Science / Python**: NumPy, Pandas, Scipy, Statsmodels, Matplotlib, Seaborn.
