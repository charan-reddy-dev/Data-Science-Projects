// ChurnIntel - Application Script Logic

document.addEventListener('DOMContentLoaded', () => {
    // Navigation handling
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = item.getAttribute('data-tab');
            
            navItems.forEach(nav => nav.classList.remove('active'));
            tabContents.forEach(tab => tab.classList.remove('active'));

            item.classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
        });
    });

    // --- Tab 1: Churn Predictor Logic ---
    const tenureSlider = document.getElementById('tenure');
    const chargesSlider = document.getElementById('monthly-charges');
    const callsSlider = document.getElementById('support-calls');
    const contractSelect = document.getElementById('contract');
    
    const tenureVal = document.getElementById('tenure-val');
    const chargesVal = document.getElementById('charges-val');
    const callsVal = document.getElementById('calls-val');

    const progressCircle = document.getElementById('progress-circle');
    const riskScorePct = document.getElementById('risk-score-pct');
    const riskStatus = document.getElementById('risk-status');
    const shapContributions = document.getElementById('shap-contributions');

    function updateInference() {
        const tenure = parseInt(tenureSlider.value);
        const charges = parseFloat(chargesSlider.value);
        const calls = parseInt(callsSlider.value);
        const contract = contractSelect.value;

        // Sync slider labels
        tenureVal.innerText = tenure;
        chargesVal.innerText = charges.toFixed(2);
        callsVal.innerText = calls;

        // Model weights derived from gradient boosting / logistic regression coefficients
        const isMonthToMonth = contract === 'Month-to-Month' ? 1 : 0;
        const isTwoYear = contract === 'Two-Year' ? 1 : 0;

        const Z = -0.05 * tenure + 0.025 * charges + 0.65 * calls + 1.3 * isMonthToMonth - 0.9 * isTwoYear - 1.8;
        const prob = 1 / (1 + Math.exp(-Z));
        const pct = Math.round(prob * 100);

        // Update radial progress bar
        const r = 65;
        const circ = 2 * Math.PI * r; // ~ 408.4
        progressCircle.style.strokeDasharray = circ;
        progressCircle.style.strokeDashoffset = circ - (prob * circ);

        // Set gauge color and label
        let color = '#00ff88'; // success
        let label = 'Low Churn Risk';
        if (pct >= 30 && pct < 60) {
            color = '#ffaa00'; // warning
            label = 'Moderate Churn Risk';
        } else if (pct >= 60) {
            color = '#ff0055'; // danger
            label = 'High Churn Risk';
        }

        progressCircle.style.stroke = color;
        riskScorePct.innerText = pct + '%';
        riskScorePct.style.color = color;
        riskStatus.innerText = label;
        riskStatus.style.color = color;

        // --- SHAP Explainability Breakdown ---
        const shap = [];

        // Contract impact
        if (contract === 'Month-to-Month') {
            shap.push({ factor: 'Month-to-Month Contract', impact: '+25%', positive: true });
        } else if (contract === 'Two-Year') {
            shap.push({ factor: 'Two-Year Commitment', impact: '-18%', positive: false });
        }

        // Support calls impact
        if (calls > 2) {
            shap.push({ factor: `Excessive Support Calls (${calls})`, impact: `+${(calls * 12)}%`, positive: true });
        } else if (calls === 0) {
            shap.push({ factor: 'Zero Support Calls', impact: '-8%', positive: false });
        }

        // Tenure impact
        if (tenure <= 6) {
            shap.push({ factor: 'Early Lifecycle Stage (<= 6m)', impact: '+15%', positive: true });
        } else if (tenure > 36) {
            shap.push({ factor: `Long Tenure Subscription (${tenure}m)`, impact: `-${Math.min(25, Math.round(tenure * 0.4))}%`, positive: false });
        }

        // Charges impact
        if (charges > 90) {
            shap.push({ factor: `High Premium Billing ($${charges})`, impact: '+10%', positive: true });
        } else if (charges < 40) {
            shap.push({ factor: `Low Plan Billing ($${charges})`, impact: '-6%', positive: false });
        }

        // Render SHAP UI list
        shapContributions.innerHTML = '';
        if (shap.length === 0) {
            shapContributions.innerHTML = '<div class="explain-item"><span class="explain-factor">Baseline risk profile active</span></div>';
        } else {
            shap.forEach(item => {
                const icon = item.positive ? '▲' : '▼';
                const cls = item.positive ? 'impact-increase' : 'impact-decrease';
                shapContributions.innerHTML += `
                    <div class="explain-item">
                        <span class="explain-factor">${item.factor}</span>
                        <span class="explain-impact ${cls}">${icon} ${item.impact} Risk</span>
                    </div>
                `;
            });
        }
    }

    [tenureSlider, chargesSlider, callsSlider, contractSelect].forEach(elem => {
        elem.addEventListener('input', updateInference);
    });


    // --- Database Initialization (100 synthetic customer records) ---
    const db = [];
    for (let i = 1; i <= 100; i++) {
        const tenure = (i * 7) % 71 + 1; // 1 to 72
        const monthly_charges = 20 + ((i * 13) % 101); // 20 to 120
        const support_calls = (i * 3) % 7; // 0 to 6
        
        let contract = "Month-to-Month";
        if (i % 3 === 1) contract = "One-Year";
        else if (i % 3 === 2) contract = "Two-Year";
        
        const isMonthToMonth = contract === "Month-to-Month" ? 1 : 0;
        const isTwoYear = contract === "Two-Year" ? 1 : 0;
        
        // Match prediction formula
        const Z = -0.05 * tenure + 0.025 * monthly_charges + 0.65 * support_calls + 1.3 * isMonthToMonth - 0.9 * isTwoYear - 1.8;
        const prob = 1 / (1 + Math.exp(-Z));
        const churn = prob > 0.5 ? 1 : 0;
        
        db.push({
            customer_id: `CUST-${String(i).padStart(4, '0')}`,
            tenure: tenure,
            monthly_charges: monthly_charges,
            support_calls: support_calls,
            contract: contract,
            churn: churn
        });
    }

    // --- Tab 2: GenAI SQL Copilot Logic ---
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('chat-user-input');
    const btnSend = document.getElementById('btn-chat-send');
    const promptChips = document.querySelectorAll('.prompt-chip');

    function appendMessage(sender, text, isHtml = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        if (isHtml) {
            msgDiv.innerHTML = text;
        } else {
            msgDiv.innerText = text;
        }
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function processAgentQuery(queryText) {
        // Parse Query using simple NLP rules
        const text = queryText.toLowerCase();
        let sql = "SELECT customer_id, tenure, contract, monthly_charges, support_calls, churn FROM customers";
        let filters = [];

        // Check Churn filter
        if (text.includes("high risk") || text.includes("churn risk") || text.includes("churn")) {
            filters.push("churn = 1");
        }

        // Check Contract filters
        if (text.includes("month-to-month") || text.includes("month to month")) {
            filters.push("contract = 'Month-to-Month'");
        } else if (text.includes("two-year") || text.includes("two year")) {
            filters.push("contract = 'Two-Year'");
        } else if (text.includes("one-year") || text.includes("one year")) {
            filters.push("contract = 'One-Year'");
        }

        // Check charges filters
        const chargesRegex = /(?:charges|billing)\s+(?:over|greater than|>)\s+(\d+)/;
        const matchCharges = text.match(chargesRegex);
        if (matchCharges) {
            filters.push(`monthly_charges > ${matchCharges[1]}`);
        }

        // Check tenure filters
        const tenureRegex = /(?:tenure)\s+(?:under|less than|<)\s+(\d+)/;
        const matchTenure = text.match(tenureRegex);
        if (matchTenure) {
            filters.push(`tenure < ${matchTenure[1]}`);
        }

        // Check support calls filters
        const callsRegex = /(?:more than|>)\s+(\d+)\s+support\s+calls/;
        const matchCalls = text.match(callsRegex);
        if (matchCalls) {
            filters.push(`support_calls > ${matchCalls[1]}`);
        }

        if (filters.length > 0) {
            sql += " WHERE " + filters.join(" AND ");
        }

        sql += " LIMIT 10;";

        // Execute local filter on the memory db
        const results = db.filter(c => {
            let pass = true;
            if (filters.includes("churn = 1") && c.churn !== 1) pass = false;
            if (filters.includes("contract = 'Month-to-Month'") && c.contract !== 'Month-to-Month') pass = false;
            if (filters.includes("contract = 'Two-Year'") && c.contract !== 'Two-Year') pass = false;
            if (filters.includes("contract = 'One-Year'") && c.contract !== 'One-Year') pass = false;
            
            // Monthly charges matches
            if (matchCharges) {
                const limit = parseInt(matchCharges[1]);
                if (c.monthly_charges <= limit) pass = false;
            }

            // Tenure matches
            if (matchTenure) {
                const limit = parseInt(matchTenure[1]);
                if (c.tenure >= limit) pass = false;
            }

            // Support calls matches
            if (matchCalls) {
                const limit = parseInt(matchCalls[1]);
                if (c.support_calls <= limit) pass = false;
            }
            return pass;
        });

        // Format conversational agent response
        let tableHtml = `
            <table class="sql-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Tenure</th>
                        <th>Contract</th>
                        <th>Charges</th>
                        <th>Calls</th>
                        <th>Risk</th>
                    </tr>
                </thead>
                <tbody>
        `;

        const displayRows = results.slice(0, 6);
        if (displayRows.length === 0) {
            tableHtml += `<tr><td colspan="6" style="text-align:center;">No records matched standard criteria.</td></tr>`;
        } else {
            displayRows.forEach(c => {
                const riskSpan = c.churn === 1 ? '<span style="color:var(--accent-red); font-weight:600;">High</span>' : '<span style="color:var(--success)">Low</span>';
                tableHtml += `
                    <tr>
                        <td>${c.customer_id}</td>
                        <td>${c.tenure}m</td>
                        <td>${c.contract}</td>
                        <td>$${c.monthly_charges.toFixed(2)}</td>
                        <td>${c.support_calls}</td>
                        <td>${riskSpan}</td>
                    </tr>
                `;
            });
        }
        tableHtml += `</tbody></table>`;

        const responseText = `
            <div>I've interpreted your request and written a SQL search query.</div>
            <div class="sql-query-block">${sql}</div>
            <div>Query executed successfully. Matched **${results.length}** records in database. Displaying top results:</div>
            <div class="sql-table-container">${tableHtml}</div>
            <div style="margin-top:0.75rem; font-size:0.85rem; color:var(--text-secondary)">
                <strong>Insight Summary:</strong> This targeted segment has an average tenure of ${results.length > 0 ? Math.round(results.reduce((acc, c) => acc + c.tenure, 0) / results.length) : 0} months, 
                and represents $${results.reduce((acc, c) => acc + c.monthly_charges, 0).toLocaleString(undefined, {maximumFractionDigits: 2})} in monthly subscription billing. 
                I recommend adding them to a high-priority retention contact group.
            </div>
        `;

        setTimeout(() => {
            appendMessage('agent', responseText, true);
        }, 600);
    }

    function handleChatSend() {
        const text = userInput.value.trim();
        if (!text) return;
        
        appendMessage('user', text);
        userInput.value = '';
        processAgentQuery(text);
    }

    btnSend.addEventListener('click', handleChatSend);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleChatSend();
    });

    promptChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query');
            userInput.value = query;
            handleChatSend();
        });
    });


    // --- Tab 3: Campaign Retention Simulator Logic ---
    const simSizeInput = document.getElementById('sim-target-size');
    const simDiscountInput = document.getElementById('sim-discount');
    const simSuccessInput = document.getElementById('sim-success-rate');
    const simLtvInput = document.getElementById('sim-ltv');

    const simSavings = document.getElementById('sim-savings');
    const simSavedRev = document.getElementById('sim-saved-revenue');
    const simCost = document.getElementById('sim-cost');
    const simSavedCount = document.getElementById('sim-saved-count');
    const simRoiVal = document.getElementById('sim-roi-val');

    const tableCohort = document.getElementById('table-cohort');
    const tableSpend = document.getElementById('table-spend');
    const tableRetainedVal = document.getElementById('table-retained-val');

    function calculateSim() {
        const cohortSize = parseInt(simSizeInput.value);
        const discount = parseFloat(simDiscountInput.value);
        const recoveryRate = parseFloat(simSuccessInput.value) / 100;
        const ltv = parseFloat(simLtvInput.value);

        if (cohortSize <= 0 || discount < 0 || recoveryRate < 0 || ltv <= 0) return;

        // Model projections
        const cost = cohortSize * discount;
        const savedCustomers = Math.round(cohortSize * recoveryRate);
        const recoveredRev = savedCustomers * ltv;
        const netSavings = recoveredRev - cost;
        const roi = cost > 0 ? (netSavings / cost) * 100 : 0;

        // Update displays
        simSavings.innerText = '$' + Math.round(netSavings).toLocaleString();
        simSavedRev.innerText = '$' + Math.round(recoveredRev).toLocaleString();
        simCost.innerText = '$' + Math.round(cost).toLocaleString();
        simSavedCount.innerText = savedCustomers.toLocaleString();
        simRoiVal.innerText = Math.round(roi).toLocaleString() + '%';

        // Update table details
        tableCohort.innerText = cohortSize.toLocaleString() + ' Customers';
        tableSpend.innerText = '-$' + Math.round(cost).toLocaleString();
        tableRetainedVal.innerText = '+$' + Math.round(recoveredRev).toLocaleString();

        const highlightCard = document.getElementById('sim-savings').parentElement;
        if (netSavings < 0) {
            highlightCard.className = "metric-highlight danger";
        } else {
            highlightCard.className = "metric-highlight success";
        }
    }

    [simSizeInput, simDiscountInput, simSuccessInput, simLtvInput].forEach(elem => {
        elem.addEventListener('input', calculateSim);
    });


    // --- Page Initialization ---
    updateInference();
    calculateSim();
});
