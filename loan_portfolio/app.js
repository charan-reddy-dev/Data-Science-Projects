/**
 * LoanIQ Portfolio Engine - Client-side Analytics & ML Underwriting
 * Powered by Balanced Logistic Regression & SHAP Explainability logic from LendingClub Capstone
 */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initUnderwriter();
  initPortfolioOptimizer();
  initStressTester();
  initEdaCharts();
});

/* ==========================================================================
   TAB NAVIGATION SYSTEM
   ========================================================================== */
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panels = document.querySelectorAll('.tab-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const activePanel = document.getElementById(`panel-${target}`);
      if (activePanel) {
        activePanel.classList.add('active');
      }

      // Trigger resize for Chart.js instances inside newly visible tabs
      window.dispatchEvent(new Event('resize'));
    });
  });
}

/* ==========================================================================
   TAB 1: UNDERWRITING RISK CLASSIFIER & SHAP EXPLAINER
   ========================================================================== */
let shapChart = null;
let currentTerm = 36;

function initUnderwriter() {
  const inputAmount = document.getElementById('input-amount');
  const inputGrade = document.getElementById('input-grade');
  const inputIntRate = document.getElementById('input-int-rate');
  const inputIncome = document.getElementById('input-income');
  const inputDti = document.getElementById('input-dti');
  const inputUtil = document.getElementById('input-util');
  const inputHome = document.getElementById('input-home');
  const inputBk = document.getElementById('input-bk');
  const inputPurpose = document.getElementById('input-purpose');

  const displayAmount = document.getElementById('display-amount');
  const displayGrade = document.getElementById('display-grade');
  const displayIntRate = document.getElementById('display-int-rate');
  const displayIncome = document.getElementById('display-income');
  const displayDti = document.getElementById('display-dti');
  const displayUtil = document.getElementById('display-util');

  const term36 = document.getElementById('term-36');
  const term60 = document.getElementById('term-60');

  // Term Button Toggles
  term36.addEventListener('click', () => {
    currentTerm = 36;
    term36.classList.add('active');
    term60.classList.remove('active');
    calculateRisk();
  });

  term60.addEventListener('click', () => {
    currentTerm = 60;
    term60.classList.add('active');
    term36.classList.remove('active');
    calculateRisk();
  });

  // Grade selection auto-adjusts suggested benchmark interest rate
  const gradeRateMap = {
    'A': 7.5,
    'B': 11.0,
    'C': 14.5,
    'D': 17.8,
    'E': 20.5,
    'F': 23.2,
    'G': 25.5
  };

  inputGrade.addEventListener('change', () => {
    const selectedGrade = inputGrade.value;
    const suggestedRate = gradeRateMap[selectedGrade] || 11.0;
    inputIntRate.value = suggestedRate;
    displayIntRate.textContent = `${suggestedRate.toFixed(1)}%`;
    displayGrade.textContent = `Grade ${selectedGrade}`;
    calculateRisk();
  });

  // Slider inputs & displays
  inputAmount.addEventListener('input', () => {
    displayAmount.textContent = `$${parseInt(inputAmount.value).toLocaleString()}`;
    calculateRisk();
  });

  inputIntRate.addEventListener('input', () => {
    displayIntRate.textContent = `${parseFloat(inputIntRate.value).toFixed(1)}%`;
    calculateRisk();
  });

  inputIncome.addEventListener('input', () => {
    displayIncome.textContent = `$${parseInt(inputIncome.value).toLocaleString()}`;
    calculateRisk();
  });

  inputDti.addEventListener('input', () => {
    displayDti.textContent = `${parseFloat(inputDti.value).toFixed(1)}%`;
    calculateRisk();
  });

  inputUtil.addEventListener('input', () => {
    displayUtil.textContent = `${parseFloat(inputUtil.value).toFixed(1)}%`;
    calculateRisk();
  });

  inputHome.addEventListener('change', calculateRisk);
  inputBk.addEventListener('change', calculateRisk);
  inputPurpose.addEventListener('change', calculateRisk);

  // Preset buttons
  document.getElementById('preset-prime').addEventListener('click', function() {
    setPresetActive(this);
    setApplicantValues({
      amount: 15000,
      term: 36,
      grade: 'A',
      rate: 7.5,
      income: 110000,
      dti: 9.5,
      util: 18,
      home: 'MORTGAGE',
      bk: '0'
    });
  });

  document.getElementById('preset-balanced').addEventListener('click', function() {
    setPresetActive(this);
    setApplicantValues({
      amount: 18000,
      term: 36,
      grade: 'C',
      rate: 14.5,
      income: 62000,
      dti: 19.5,
      util: 52,
      home: 'RENT',
      bk: '0'
    });
  });

  document.getElementById('preset-subprime').addEventListener('click', function() {
    setPresetActive(this);
    setApplicantValues({
      amount: 28000,
      term: 60,
      grade: 'F',
      rate: 23.5,
      income: 42000,
      dti: 34.0,
      util: 88,
      home: 'RENT',
      bk: '1'
    });
  });

  function setPresetActive(btn) {
    document.querySelectorAll('.preset-pill-group .btn-chip').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }

  function setApplicantValues(vals) {
    inputAmount.value = vals.amount;
    displayAmount.textContent = `$${vals.amount.toLocaleString()}`;

    currentTerm = vals.term;
    if (vals.term === 36) {
      term36.classList.add('active');
      term60.classList.remove('active');
    } else {
      term60.classList.add('active');
      term36.classList.remove('active');
    }

    inputGrade.value = vals.grade;
    displayGrade.textContent = `Grade ${vals.grade}`;

    inputIntRate.value = vals.rate;
    displayIntRate.textContent = `${vals.rate.toFixed(1)}%`;

    inputIncome.value = vals.income;
    displayIncome.textContent = `$${vals.income.toLocaleString()}`;

    inputDti.value = vals.dti;
    displayDti.textContent = `${vals.dti.toFixed(1)}%`;

    inputUtil.value = vals.util;
    displayUtil.textContent = `${vals.util.toFixed(1)}%`;

    inputHome.value = vals.home;
    inputBk.value = vals.bk;

    calculateRisk();
  }

  // Initial Calculation
  calculateRisk();
}

/**
 * Balanced Logistic Regression Inference & SHAP Decomposition
 */
function calculateRisk() {
  const amount = parseFloat(document.getElementById('input-amount').value);
  const term = currentTerm;
  const grade = document.getElementById('input-grade').value;
  const rate = parseFloat(document.getElementById('input-int-rate').value);
  const income = parseFloat(document.getElementById('input-income').value);
  const dti = parseFloat(document.getElementById('input-dti').value);
  const util = parseFloat(document.getElementById('input-util').value);
  const home = document.getElementById('input-home').value;
  const bk = parseInt(document.getElementById('input-bk').value);

  // Grade index: A=0, B=1, ... G=6
  const gradeMap = { 'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6 };
  const gradeIdx = gradeMap[grade] || 2;

  // Logistic Regression Base Log-Odds for Payoff (Class 1) vs Default (Class 0)
  // Calibrated with balanced weights from Capstone Notebook
  let logOdds = 1.95; // Prior favoring payoff (baseline ~85%)

  // SHAP Feature Impact deltas (in probability space)
  const shapDeltas = [];

  // 1. Annual Income Impact
  // Normalized around median LC income $65,000
  const incomeLogRatio = Math.log(income / 65000);
  const incomeImpactOdds = incomeLogRatio * 0.72;
  logOdds += incomeImpactOdds;
  shapDeltas.push({
    name: 'Annual Income',
    desc: `$${income.toLocaleString()} (${income >= 65000 ? 'Above' : 'Below'} Median)`,
    impact: (incomeImpactOdds * 4.2), // Approx % shift
  });

  // 2. Debt to Income (DTI)
  // Median DTI ~ 15%
  const dtiDelta = (15.0 - dti) / 10.0;
  const dtiImpactOdds = dtiDelta * 0.65;
  logOdds += dtiImpactOdds;
  shapDeltas.push({
    name: 'Debt-to-Income (DTI)',
    desc: `${dti.toFixed(1)}% (Threshold: 20%)`,
    impact: (dtiImpactOdds * 4.0),
  });

  // 3. Revolving Credit Line Utilization
  // Median LC Revol Util ~ 48%
  const utilDelta = (48.0 - util) / 25.0;
  const utilImpactOdds = utilDelta * 0.58;
  logOdds += utilImpactOdds;
  shapDeltas.push({
    name: 'Revol. Line Utilization',
    desc: `${util.toFixed(1)}% Credit Line Usage`,
    impact: (utilImpactOdds * 3.8),
  });

  // 4. Grade & Interest Rate Risk Burden
  // Higher interest rates and lower grades represent steep default penalties
  const gradePenalty = -(gradeIdx * 0.48) - ((rate - 11.5) * 0.08);
  logOdds += gradePenalty;
  shapDeltas.push({
    name: 'Loan Grade & Interest',
    desc: `Grade ${grade} @ ${rate.toFixed(1)}%`,
    impact: (gradePenalty * 4.4),
  });

  // 5. Loan Amount Scale Risk
  // Median loan ~ $12,000
  const amountPenalty = -((amount - 12000) / 10000) * 0.32;
  logOdds += amountPenalty;
  shapDeltas.push({
    name: 'Principal Exposure',
    desc: `$${amount.toLocaleString()} Asked Principal`,
    impact: (amountPenalty * 3.5),
  });

  // 6. Term Duration (36 mo vs 60 mo)
  // Notebook finding: 36m has 77% of payoffs and lower delinquency
  const termEffect = term === 36 ? 0.35 : -0.45;
  logOdds += termEffect;
  shapDeltas.push({
    name: 'Loan Term Tenure',
    desc: `${term} Months (${term === 36 ? 'Lower Volatility' : 'Extended Duration'})`,
    impact: (termEffect * 4.5),
  });

  // 7. Public Record Bankruptcies
  const bkPenalty = -(bk * 0.85);
  logOdds += bkPenalty;
  shapDeltas.push({
    name: 'Bankruptcy Public Records',
    desc: bk === 0 ? 'Clean Record (0 Filings)' : `${bk} Prior Filing(s)`,
    impact: (bkPenalty * 4.8),
  });

  // Convert Log-Odds to Probability
  const payoffProb = 1 / (1 + Math.exp(-logOdds));
  const defaultProb = 1 - payoffProb;

  // Synthetic Credit Rating Score (300 to 850 scale)
  const creditScore = Math.round(320 + (payoffProb * 510) - (bk * 45));
  const clampedScore = Math.max(340, Math.min(845, creditScore));

  // Update UI KPI Scores
  const defaultProbPct = (defaultProb * 100).toFixed(1);
  const payoffProbPct = (payoffProb * 100).toFixed(1);

  document.getElementById('kpi-default-prob').textContent = `${defaultProbPct}%`;
  document.getElementById('kpi-payoff-prob').textContent = `${payoffProbPct}%`;
  document.getElementById('kpi-credit-score').textContent = clampedScore;

  // Score meter widths
  const meterDefault = document.getElementById('meter-default');
  const meterPayoff = document.getElementById('meter-payoff');
  const meterScore = document.getElementById('meter-score');

  meterDefault.style.width = `${Math.min(100, defaultProb * 100)}%`;
  meterPayoff.style.width = `${Math.min(100, payoffProb * 100)}%`;
  meterScore.style.width = `${((clampedScore - 300) / 550) * 100}%`;

  // Decision Verdict Badge and Recommendation
  const decisionBadge = document.getElementById('decision-badge');
  const decisionText = document.getElementById('decision-text');
  const calloutTitle = document.getElementById('callout-title');
  const calloutDesc = document.getElementById('callout-desc');
  const scoreTier = document.getElementById('kpi-score-tier');

  decisionBadge.className = 'decision-badge';

  if (defaultProb < 0.08) {
    decisionBadge.classList.add('badge-approve');
    decisionBadge.querySelector('.badge-icon').textContent = '✓';
    decisionText.textContent = 'APPROVED FOR FUNDING';
    scoreTier.textContent = 'Prime Tier (Low Risk)';
    meterDefault.className = 'meter-bar emerald-fill';
    calloutTitle.textContent = 'Highly Recommended Note';
    calloutDesc.textContent = `Default probability of only ${defaultProbPct}% meets stringent institutional underwriting benchmarks. Projected to return full principal plus ${rate.toFixed(1)}% coupon interest.`;
  } else if (defaultProb < 0.18) {
    decisionBadge.classList.add('badge-conditional');
    decisionBadge.querySelector('.badge-icon').textContent = '⚠️';
    decisionText.textContent = 'CONDITIONAL APPROVAL';
    scoreTier.textContent = 'Moderate Quality Tier';
    meterDefault.className = 'meter-bar amber-fill';
    calloutTitle.textContent = 'Acceptable with Yield Cushion';
    calloutDesc.textContent = `Moderate default risk (${defaultProbPct}%). Yield of ${rate.toFixed(1)}% provides sufficient spread to absorb potential volatility, provided portfolio is adequately diversified.`;
  } else {
    decisionBadge.classList.add('badge-decline');
    decisionBadge.querySelector('.badge-icon').textContent = '✕';
    decisionText.textContent = 'HIGH RISK (DECLINED)';
    scoreTier.textContent = 'Subprime / Speculative Tier';
    meterDefault.className = 'meter-bar rose-fill';
    calloutTitle.textContent = 'High Default Exposure';
    calloutDesc.textContent = `The balanced classifier flags this loan with an elevated ${defaultProbPct}% risk of charge-off. Filtered out by the ML allocation pipeline to preserve investor capital.`;
  }

  // Render SHAP Waterfall Chart & Breakdown
  renderShapWaterfall(shapDeltas);
}

/**
 * Render SHAP Feature Impact Chart
 */
function renderShapWaterfall(deltas) {
  // Sort deltas by magnitude of absolute impact
  deltas.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

  const labels = deltas.map(d => d.name);
  const values = deltas.map(d => parseFloat(d.impact.toFixed(1)));
  const backgroundColors = values.map(v => v >= 0 ? '#10b981' : '#f43f5e');
  const borderColors = values.map(v => v >= 0 ? '#34d399' : '#fb7185');

  const ctx = document.getElementById('shapWaterfallChart').getContext('2d');

  if (shapChart) {
    shapChart.data.labels = labels;
    shapChart.data.datasets[0].data = values;
    shapChart.data.datasets[0].backgroundColor = backgroundColors;
    shapChart.data.datasets[0].borderColor = borderColors;
    shapChart.update();
  } else {
    shapChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'SHAP Feature Contribution (%)',
          data: values,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                const val = context.raw;
                return val >= 0 ? `+${val}% Towards Payoff` : `${val}% Risk Penalty`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            ticks: {
              color: '#94a3b8',
              font: { family: 'Outfit', size: 11 },
              callback: val => (val > 0 ? `+${val}%` : `${val}%`)
            }
          },
          y: {
            grid: { display: false },
            ticks: {
              color: '#f8fafc',
              font: { family: 'Outfit', size: 11, weight: '500' }
            }
          }
        }
      }
    });
  }

  // Update text item breakdown
  const breakdownList = document.getElementById('shap-breakdown-list');
  breakdownList.innerHTML = '';

  deltas.slice(0, 4).forEach(item => {
    const isPositive = item.impact >= 0;
    const row = document.createElement('div');
    row.className = 'shap-item';
    row.innerHTML = `
      <span class="shap-feat-name">${item.name}: <span style="color: #64748b;">${item.desc}</span></span>
      <span class="shap-impact-val ${isPositive ? 'emerald-text' : 'rose-text'}">
        ${isPositive ? '+' : ''}${item.impact.toFixed(1)}%
      </span>
    `;
    breakdownList.appendChild(row);
  });
}

/* ==========================================================================
   TAB 2: PORTFOLIO YIELD OPTIMIZER
   ========================================================================== */
let growthChart = null;
let donutChart = null;

function initPortfolioOptimizer() {
  const capitalSlider = document.getElementById('port-capital-slider');
  const displayCapital = document.getElementById('port-display-capital');

  const allocA = document.getElementById('alloc-a');
  const allocB = document.getElementById('alloc-b');
  const allocC = document.getElementById('alloc-c');
  const allocD = document.getElementById('alloc-d');

  const allocValA = document.getElementById('alloc-val-a');
  const allocValB = document.getElementById('alloc-val-b');
  const allocValC = document.getElementById('alloc-val-c');
  const allocValD = document.getElementById('alloc-val-d');

  const filterMl = document.getElementById('filter-ml-class1');
  const filterNoBk = document.getElementById('filter-no-bk');
  const filterTerm = document.getElementById('filter-term-tilt');

  capitalSlider.addEventListener('input', () => {
    const cap = parseInt(capitalSlider.value);
    displayCapital.textContent = `$${cap.toLocaleString()}`;
    recalculatePortfolio();
  });

  const sliders = [allocA, allocB, allocC, allocD];
  sliders.forEach(s => {
    s.addEventListener('input', () => {
      allocValA.textContent = `${allocA.value}%`;
      allocValB.textContent = `${allocB.value}%`;
      allocValC.textContent = `${allocC.value}%`;
      allocValD.textContent = `${allocD.value}%`;
      recalculatePortfolio();
    });
  });

  [filterMl, filterNoBk, filterTerm].forEach(chk => {
    chk.addEventListener('change', recalculatePortfolio);
  });

  // Preset investment profiles
  document.getElementById('strat-conservative').addEventListener('click', function() {
    setStratActive(this);
    setAllocations(55, 35, 10, 0);
  });

  document.getElementById('strat-balanced').addEventListener('click', function() {
    setStratActive(this);
    setAllocations(40, 35, 20, 5);
  });

  document.getElementById('strat-yield').addEventListener('click', function() {
    setStratActive(this);
    setAllocations(15, 35, 35, 15);
  });

  function setStratActive(btn) {
    document.querySelectorAll('.strategy-pill-group .btn-chip').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }

  function setAllocations(a, b, c, d) {
    allocA.value = a; allocValA.textContent = `${a}%`;
    allocB.value = b; allocValB.textContent = `${b}%`;
    allocC.value = c; allocValC.textContent = `${c}%`;
    allocD.value = d; allocValD.textContent = `${d}%`;
    recalculatePortfolio();
  }

  recalculatePortfolio();
}

function recalculatePortfolio() {
  const capital = parseInt(document.getElementById('port-capital-slider').value);

  // Raw slider weights
  let wA = parseFloat(document.getElementById('alloc-a').value);
  let wB = parseFloat(document.getElementById('alloc-b').value);
  let wC = parseFloat(document.getElementById('alloc-c').value);
  let wD = parseFloat(document.getElementById('alloc-d').value);

  const totalW = (wA + wB + wC + wD) || 100;
  // Normalized percentages
  const pA = wA / totalW;
  const pB = wB / totalW;
  const pC = wC / totalW;
  const pD = wD / totalW;

  const filterMl = document.getElementById('filter-ml-class1').checked;
  const filterNoBk = document.getElementById('filter-no-bk').checked;
  const filterTerm = document.getElementById('filter-term-tilt').checked;

  // Historical contract coupons by grade
  const couponA = 0.075;
  const couponB = 0.110;
  const couponC = 0.145;
  const couponD = 0.205; // Weighted blend of D through G

  // Baseline LendingClub raw default rates by grade
  const baseDefA = 0.058;
  const baseDefB = 0.118;
  const baseDefC = 0.174;
  const baseDefD = 0.285;

  const grossCoupon = (pA * couponA) + (pB * couponB) + (pC * couponC) + (pD * couponD);
  const baselineDefault = (pA * baseDefA) + (pB * baseDefB) + (pC * baseDefC) + (pD * baseDefD);

  // Machine Learning Filter Impact:
  // Balanced Logistic Regression eliminates 87% of defaults (Class 0 recall)
  let mlDefaultRate = baselineDefault * (filterMl ? 0.124 : 1.0);

  // Bankruptcy filter adds +0.10% IRR boost by pruning tail defaults
  if (filterNoBk) {
    mlDefaultRate *= 0.92;
  }
  // Term tilt favors 36m lower volatility
  if (filterTerm) {
    mlDefaultRate *= 0.94;
  }

  // Expected charge-off loss (assuming ~30% recovery rate on defaulted principal)
  const lossRateMl = mlDefaultRate * 0.70;
  const lossRateBase = baselineDefault * 0.70;

  const servicingFee = 0.005; // 0.50% LendingClub standard investor servicing fee

  const netIrrMl = grossCoupon - lossRateMl - servicingFee;
  const netIrrBase = grossCoupon - lossRateBase - servicingFee;

  const alpha = netIrrMl - netIrrBase;
  const annualSavingsDollars = capital * (lossRateBase - lossRateMl);

  // Update Top KPIs
  document.getElementById('port-val-capital').textContent = `$${capital.toLocaleString()}`;
  document.getElementById('port-val-base-default').textContent = `${(baselineDefault * 100).toFixed(2)}%`;
  document.getElementById('port-val-ml-default').textContent = `${(mlDefaultRate * 100).toFixed(2)}%`;
  document.getElementById('port-val-net-irr').textContent = `${(netIrrMl * 100).toFixed(2)}%`;
  document.getElementById('port-val-alpha').textContent = `+${(alpha * 100).toFixed(2)}% Alpha vs Baseline`;

  // Summary Card
  document.getElementById('summary-gross').textContent = `${(grossCoupon * 100).toFixed(2)}%`;
  document.getElementById('summary-loss').textContent = `-${(lossRateMl * 100).toFixed(2)}%`;
  document.getElementById('summary-net').textContent = `${(netIrrMl * 100).toFixed(2)}%`;
  document.getElementById('summary-savings').textContent = `$${Math.round(annualSavingsDollars).toLocaleString()}`;

  // Update Charts
  renderPortfolioCharts(capital, netIrrMl, netIrrBase, [pA, pB, pC, pD]);
}

function renderPortfolioCharts(capital, netIrrMl, netIrrBase, gradePcts) {
  // 36 Months Cumulative Growth Simulation
  const months = [0, 6, 12, 18, 24, 30, 36];
  const mlCurve = months.map(m => Math.round(capital * Math.pow(1 + netIrrMl, m / 12)));
  const baseCurve = months.map(m => Math.round(capital * Math.pow(1 + netIrrBase, m / 12)));

  const ctxGrowth = document.getElementById('portfolioGrowthChart').getContext('2d');
  if (growthChart) {
    growthChart.data.datasets[0].data = mlCurve;
    growthChart.data.datasets[1].data = baseCurve;
    growthChart.update();
  } else {
    growthChart = new Chart(ctxGrowth, {
      type: 'line',
      data: {
        labels: ['Month 0', 'Month 6', 'Month 12', 'Month 18', 'Month 24', 'Month 30', 'Month 36'],
        datasets: [
          {
            label: 'ML-Filtered Portfolio',
            data: mlCurve,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.12)',
            fill: true,
            tension: 0.35,
            borderWidth: 2.5,
            pointBackgroundColor: '#10b981',
            pointRadius: 4
          },
          {
            label: 'Unfiltered LendingClub Market',
            data: baseCurve,
            borderColor: '#f43f5e',
            backgroundColor: 'transparent',
            borderDash: [5, 5],
            tension: 0.35,
            borderWidth: 2,
            pointBackgroundColor: '#f43f5e',
            pointRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            labels: { color: '#94a3b8', font: { family: 'Outfit', size: 12 } }
          },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: $${ctx.raw.toLocaleString()}`
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              color: '#94a3b8',
              font: { family: 'Outfit', size: 11 },
              callback: v => `$${(v / 1000).toFixed(0)}k`
            }
          }
        }
      }
    });
  }

  // Donut Allocation Chart
  const ctxDonut = document.getElementById('portfolioDonutChart').getContext('2d');
  const donutData = gradePcts.map(p => Math.round(p * 100));

  if (donutChart) {
    donutChart.data.datasets[0].data = donutData;
    donutChart.update();
  } else {
    donutChart = new Chart(ctxDonut, {
      type: 'doughnut',
      data: {
        labels: ['Grade A', 'Grade B', 'Grade C', 'Grade D-G'],
        datasets: [{
          data: donutData,
          backgroundColor: ['#10b981', '#06b6d4', '#f59e0b', '#f43f5e'],
          borderWidth: 2,
          borderColor: '#0f172a'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
          }
        },
        cutout: '70%'
      }
    });
  }
}

/* ==========================================================================
   TAB 3: MACROECONOMIC STRESS TESTING
   ========================================================================== */
let stressChart = null;

const stressScenarios = {
  baseline: {
    name: 'Normal Expansion',
    desc: 'Stable GDP growth, 4.0% unemployment rate, neutral credit conditions',
    multipliers: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    defaultRate: 2.1,
    drawdown: -1.2,
    netReturn: '+9.8%'
  },
  rateHike: {
    name: 'Fed Rate Hike (+250 bps)',
    desc: 'Tighter consumer borrowing costs, variable debt burden strains DTI',
    multipliers: [1.15, 1.25, 1.45, 1.65, 1.85, 2.05, 2.30],
    defaultRate: 3.4,
    drawdown: -2.8,
    netReturn: '+8.4%'
  },
  mildRecession: {
    name: 'Mild Economic Downturn',
    desc: 'Unemployment rises to 6.2%, discretionary debt charge-offs increase',
    multipliers: [1.30, 1.50, 1.85, 2.20, 2.65, 3.10, 3.60],
    defaultRate: 4.8,
    drawdown: -4.5,
    netReturn: '+6.9%'
  },
  gfcShock: {
    name: '2008 Great Financial Crisis Shock',
    desc: 'Unemployment spikes to 9.5%, severe consumer credit market freeze',
    multipliers: [1.60, 1.95, 2.60, 3.40, 4.20, 5.00, 5.80],
    defaultRate: 7.9,
    drawdown: -8.2,
    netReturn: '+3.5%'
  }
};

function initStressTester() {
  const pBaseline = document.getElementById('shock-baseline');
  const pRateHike = document.getElementById('shock-rate-hike');
  const pMild = document.getElementById('shock-mild-recession');
  const pGfc = document.getElementById('shock-2008-crisis');

  const buttons = [pBaseline, pRateHike, pMild, pGfc];

  function activate(btn, key) {
    buttons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyStressScenario(key);
  }

  pBaseline.addEventListener('click', () => activate(pBaseline, 'baseline'));
  pRateHike.addEventListener('click', () => activate(pRateHike, 'rateHike'));
  pMild.addEventListener('click', () => activate(pMild, 'mildRecession'));
  pGfc.addEventListener('click', () => activate(pGfc, 'gfcShock'));

  applyStressScenario('baseline');
}

function applyStressScenario(scenarioKey) {
  const sc = stressScenarios[scenarioKey];
  document.getElementById('stress-scenario-name').textContent = sc.name;
  document.getElementById('stress-scenario-desc').textContent = sc.desc;
  document.getElementById('stress-shock-default').textContent = `${sc.defaultRate.toFixed(1)}%`;
  document.getElementById('stress-drawdown').textContent = `${sc.drawdown.toFixed(1)}%`;
  document.getElementById('stress-post-return').textContent = sc.netReturn;

  const baselineGradeDefaults = [5.8, 11.8, 17.4, 24.5, 28.2, 33.1, 38.4];
  const shockedGradeDefaults = baselineGradeDefaults.map((d, i) => parseFloat((d * sc.multipliers[i]).toFixed(1)));

  const ctx = document.getElementById('stressBarChart').getContext('2d');
  const grades = ['Grade A', 'Grade B', 'Grade C', 'Grade D', 'Grade E', 'Grade F', 'Grade G'];

  if (stressChart) {
    stressChart.data.datasets[1].data = shockedGradeDefaults;
    stressChart.update();
  } else {
    stressChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: grades,
        datasets: [
          {
            label: 'Normal Baseline Default (%)',
            data: baselineGradeDefaults,
            backgroundColor: 'rgba(99, 102, 241, 0.45)',
            borderColor: '#818cf8',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'Shocked Stress Default (%)',
            data: shockedGradeDefaults,
            backgroundColor: 'rgba(244, 63, 94, 0.65)',
            borderColor: '#fb7185',
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } } },
          tooltip: {
            callbacks: {
              label: c => `${c.dataset.label}: ${c.raw}% default`
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              color: '#94a3b8',
              font: { family: 'Outfit', size: 11 },
              callback: v => `${v}%`
            }
          }
        }
      }
    });
  }
}

/* ==========================================================================
   TAB 4: EXPLORATORY DATA ANALYSIS CHARTS (FROM CAPSTONE NOTEBOOKS)
   ========================================================================== */
function initEdaCharts() {
  // Chart 1: Grade vs Interest Rate & Payoff Rate
  const ctxGrade = document.getElementById('edaGradeChart').getContext('2d');
  new Chart(ctxGrade, {
    type: 'bar',
    data: {
      labels: ['Grade A', 'Grade B', 'Grade C', 'Grade D', 'Grade E', 'Grade F', 'Grade G'],
      datasets: [
        {
          type: 'line',
          label: 'Avg Interest Rate (%)',
          data: [7.3, 11.0, 14.5, 17.8, 20.5, 23.2, 25.4],
          borderColor: '#38bdf8',
          backgroundColor: '#38bdf8',
          borderWidth: 2.5,
          pointRadius: 4,
          yAxisID: 'y'
        },
        {
          type: 'bar',
          label: 'Payoff Rate (%)',
          data: [94.2, 88.2, 82.6, 75.5, 71.8, 66.9, 61.6],
          backgroundColor: 'rgba(16, 185, 129, 0.35)',
          borderColor: '#10b981',
          borderWidth: 1,
          borderRadius: 4,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } } }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
        },
        y: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'Interest Rate (%)', color: '#38bdf8', font: { size: 11 } },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#94a3b8', font: { size: 11 } }
        },
        y1: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: 'Payoff Rate (%)', color: '#10b981', font: { size: 11 } },
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 11 } },
          min: 50,
          max: 100
        }
      }
    }
  });

  // Chart 2: Term Distribution (36 vs 60 mo)
  const ctxTerm = document.getElementById('edaTermChart').getContext('2d');
  new Chart(ctxTerm, {
    type: 'bar',
    data: {
      labels: ['36 Months (3-Year)', '60 Months (5-Year)'],
      datasets: [
        {
          label: 'Loan Origination Share (%)',
          data: [76.5, 23.5],
          backgroundColor: ['rgba(99, 102, 241, 0.65)', 'rgba(6, 182, 212, 0.65)'],
          borderColor: ['#818cf8', '#38bdf8'],
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'Payoff Rate (%)',
          data: [87.8, 74.2],
          backgroundColor: ['rgba(16, 185, 129, 0.65)', 'rgba(245, 158, 11, 0.65)'],
          borderColor: ['#10b981', '#fbbf24'],
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } } }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#94a3b8',
            font: { family: 'Outfit', size: 11 },
            callback: v => `${v}%`
          }
        }
      }
    }
  });

  // Chart 3: Geographic Distribution
  const ctxGeo = document.getElementById('edaGeoChart').getContext('2d');
  new Chart(ctxGeo, {
    type: 'doughnut',
    data: {
      labels: ['California (CA)', 'New York (NY)', 'Florida (FL)', 'Texas (TX)', 'New Jersey (NJ)', 'Illinois (IL)'],
      datasets: [{
        data: [35.7, 18.2, 14.5, 12.8, 10.1, 8.7],
        backgroundColor: ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6'],
        borderColor: '#0f172a',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
        },
        tooltip: {
          callbacks: {
            label: c => `${c.label}: ${c.raw}% of originations`
          }
        }
      },
      cutout: '65%'
    }
  });
}
