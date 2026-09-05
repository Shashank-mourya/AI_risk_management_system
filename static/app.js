/*
 * Client-side logic: tab switching, scoring via /api/score, cost-model
 * recalculation via /api/cost-optimal, the Chart.js charts, and the held-out
 * metrics from /api/meta.
 */

// --- state
let META = null;          // model metadata from /api/meta
let SWEEP = null;         // threshold sweep from /api/sweep
let costChart = null;     // Chart.js instance for cost tab
let contribChart = null;  // Chart.js instance for contributions

// The FX rate is NOT a constant here. It lives in cost_model.py and arrives on
// /api/meta; a hardcoded copy in this file would silently mis-convert every
// order value the moment the one in Python moved, and the score would move
// with it. gbpToInr() throws rather than guessing.
function gbpToInr() {
    if (!META || typeof META.gbp_to_inr !== 'number') {
        throw new Error('Model metadata has not loaded, so no currency rate is known.');
    }
    return META.gbp_to_inr;
}

// --- init
document.addEventListener('DOMContentLoaded', async () => {
    initTabs();
    initRangeInputs();
    initScoreButton();
    initCostButton();
    initToggles();
    initColdStartWatch();

    try {
        const [metaRes, sweepRes] = await Promise.all([
            fetch('/api/meta'),
            fetch('/api/sweep')
        ]);
        // A failed /api/meta answers with {error}, not with metadata. Assigning
        // that to META left every later reader silently reading undefined - a
        // header with no model name, bands with no bounds - instead of saying
        // the model had not loaded. The API's reason is the useful message.
        const metaBody = await metaRes.json();
        if (!metaRes.ok) throw new Error(metaBody.error || `/api/meta returned ${metaRes.status}`);
        META = metaBody;
        if (sweepRes.ok) SWEEP = await sweepRes.json();

        renderHeader();
        renderHoldoutTab();
        initCostDefaults();
        renderInitialCostChart();
    } catch (e) {
        console.error('Failed to load metadata:', e);
        document.getElementById('header-subtitle').textContent =
            'Model metadata unavailable — ' + e.message;
    }
});


// --- tabs
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
        });
    });
}

// --- range inputs
function initRangeInputs() {
    document.querySelectorAll('input[type="range"]').forEach(input => {
        const display = document.getElementById('rv-' + input.id.replace('f-', '').replace('c-', ''));
        if (display) {
            const updateDisplay = () => {
                const v = parseFloat(input.value);
                display.textContent = input.step === '1' ? v.toString() : v.toFixed(input.id.includes('hour') || input.id.includes('day_of_week') ? 0 : (v >= 1 ? 1 : 3));
            };
            input.addEventListener('input', updateDisplay);
            updateDisplay();
        }
    });

    // Fix range value display for cost tab sliders
    ['c-recovery', 'c-prevention', 'c-abandon', 'c-margin'].forEach(id => {
        const input = document.getElementById(id);
        const display = document.getElementById('rv-' + id);
        if (input && display) {
            const updateDisplay = () => {
                display.textContent = parseFloat(input.value).toFixed(2);
            };
            input.addEventListener('input', updateDisplay);
            updateDisplay();
        }
    });
}


// --- header
function renderHeader() {
    if (!META) return;
    const sub = document.getElementById('header-subtitle');
    sub.textContent = `UCI Online Retail II · ${META.winner} · operating threshold ${META.threshold} · amounts in INR at an assumed ${META.gbp_to_inr}/GBP`;

    if (META.data_is_synthetic) {
        document.getElementById('synthetic-badge').style.display = 'inline-flex';
    }
}


// --- score tab
function initScoreButton() {
    document.getElementById('score-btn').addEventListener('click', scoreOrder);
}

function initColdStartWatch() {
    const priorInput = document.getElementById('f-customer_prior_orders');
    const note = document.getElementById('cold-start-note');
    priorInput.addEventListener('input', () => {
        note.style.display = parseInt(priorInput.value) === 0 ? 'flex' : 'none';
    });
}

async function scoreOrder() {
    const btn = document.getElementById('score-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Scoring…';

    let rate;
    try {
        rate = gbpToInr();
    } catch (e) {
        showScoreError(e.message);
        btn.disabled = false;
        btn.innerHTML = 'Score this order';
        return;
    }

    const priorOrders = parseInt(getVal('f-customer_prior_orders'));
    // customer_prior_returns counts prior ORDERS observed returned, so it
    // cannot exceed the prior order count - build_features.py asserts exactly
    // that, and the model has never seen a row where it does. Clamping only
    // the ratio, as this did, still sent the impossible count itself.
    const rawReturns = parseInt(getVal('f-customer_prior_returns'));
    const priorReturns = Math.min(rawReturns, priorOrders);
    if (rawReturns > priorOrders) {
        document.getElementById('f-customer_prior_returns').value = priorReturns;
    }
    const orderValueINR = parseFloat(getVal('f-order_value'));
    const orderValueGBP = orderValueINR / rate;

    const isNew = priorOrders === 0 ? 1 : 0;
    const returnRate = priorOrders === 0 ? -1.0 : priorReturns / priorOrders;

    const order = {
        order_value: orderValueGBP,
        log_order_value: Math.log1p(Math.max(orderValueGBP, 0)),
        n_lines: parseInt(getVal('f-n_lines')),
        total_quantity: parseInt(getVal('f-total_quantity')),
        // Unit prices are entered in INR like every other amount on the page;
        // the features themselves are denominated in GBP.
        mean_unit_price: parseFloat(getVal('f-mean_unit_price')) / rate,
        max_unit_price: parseFloat(getVal('f-max_unit_price')) / rate,
        price_vs_sku_mean: parseFloat(getVal('f-price_vs_sku_mean')),
        hour_of_day: parseInt(document.getElementById('f-hour_of_day').value),
        day_of_week: parseInt(document.getElementById('f-day_of_week').value),
        is_uk: document.getElementById('f-is_uk').checked ? 1 : 0,
        customer_prior_orders: priorOrders,
        customer_prior_returns: priorReturns,
        customer_prior_return_rate: returnRate,
        customer_tenure_days: parseFloat(getVal('f-customer_tenure_days')),
        is_new_customer: isNew,
        basket_sku_return_rate: parseFloat(document.getElementById('f-basket_sku_return_rate').value),
        basket_max_sku_return_rate: parseFloat(document.getElementById('f-basket_max_sku_return_rate').value),
    };

    try {
        const res = await fetch('/api/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(order)
        });
        const data = await res.json();

        if (data.error) {
            showScoreError(data.error);
        } else {
            renderScoreResult(data);
        }
    } catch (e) {
        showScoreError(e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Score this order';
    }
}

function getVal(id) {
    return document.getElementById(id).value;
}

function showScoreError(msg) {
    const panel = document.getElementById('score-result');
    panel.innerHTML = `<div class="callout callout-error"><span class="icon">❌</span><span>${escHtml(msg)}</span></div>`;
    panel.classList.add('visible');
    document.getElementById('score-placeholder').style.display = 'none';
}

function renderScoreResult(data) {
    const placeholder = document.getElementById('score-placeholder');
    placeholder.style.display = 'none';

    const panel = document.getElementById('score-result');
    const recLabel = data.recommendation.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

    const calloutMap = {
        allow: { cls: 'callout-success', icon: '✅' },
        manual_review: { cls: 'callout-warning', icon: '⚠️' },
        hold_payout: { cls: 'callout-error', icon: '⛔' },
    };
    const co = calloutMap[data.recommendation] || calloutMap.manual_review;

    const bounds = META ? { medium: META.band_bounds.medium, high: META.band_bounds.high }
                       : { medium: data.threshold_applied, high: Math.min(data.threshold_applied * 2, 1) };

    const baseRate = META?.holdout?.base_rate_test;

    panel.innerHTML = `
        <h2>Decision</h2>
        <div class="four-col" style="grid-template-columns: 1fr 1fr 1fr; margin:16px 0;">
            <div class="metric">
                <div class="metric-label">Return probability</div>
                <div class="metric-value">${(data.risk_probability * 100).toFixed(1)}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Risk band</div>
                <div class="metric-value">${data.risk_band.toUpperCase()}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Threshold</div>
                <div class="metric-value">${data.threshold_applied.toFixed(2)}</div>
            </div>
        </div>
        <div class="callout ${co.cls}">
            <span class="icon">${co.icon}</span>
            <span><strong>${recLabel}</strong> — a recommendation, not an action. Acting on it requires a named reviewer.</span>
        </div>
        <p class="caption">
            Bands sit at the threshold (${bounds.medium.toFixed(2)}) and at twice it (${bounds.high.toFixed(2)}),
            so they move when the operating point does.${baseRate ? ` Base rate is ${(baseRate * 100).toFixed(1)}%.` : ''}
        </p>

        <h2 style="margin-top:24px;">What drove this score</h2>
        ${data.top_features && data.top_features.length > 0
            ? `<div class="chart-container" style="height:260px;"><canvas id="contrib-chart"></canvas></div>
               <p class="caption">
                   For logistic regression this is exact — coefficient × standardised value, the additive
                   terms of the log-odds. Not a post-hoc approximation.
               </p>`
            : `<div class="callout callout-info"><span class="icon">ℹ️</span>
               <span>The shipped model has no linear decomposition, so no exact per-feature contribution is shown rather than an invented one.</span></div>`
        }
    `;

    panel.classList.add('visible');

    // Draw contribution chart
    if (data.top_features && data.top_features.length > 0) {
        drawContribChart(data.top_features);
    }
}

function drawContribChart(features) {
    if (contribChart) contribChart.destroy();

    const reversed = [...features].reverse();
    const labels = reversed.map(f => f.feature);
    const values = reversed.map(f => f.contribution);
    const colors = values.map(v => v > 0 ? '#e66767' : '#3987e5');

    const ctx = document.getElementById('contrib-chart').getContext('2d');
    contribChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1,
                borderRadius: 3,
                barPercentage: 0.65,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#222221',
                    borderColor: '#3a3a37',
                    borderWidth: 1,
                    titleFont: { family: 'Inter' },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    callbacks: {
                        label: ctx => `contribution: ${ctx.raw > 0 ? '+' : ''}${ctx.raw.toFixed(3)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(58,58,55,0.3)' },
                    ticks: { color: '#8a8980', font: { size: 10, family: 'JetBrains Mono' } },
                    title: {
                        display: true,
                        text: 'contribution to log-odds  ← lowers risk · raises risk →',
                        color: '#8a8980',
                        font: { size: 11, family: 'Inter' }
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#c3c2b7', font: { size: 10, family: 'JetBrains Mono' } }
                }
            }
        }
    });
}


// --- cost tab
function initCostDefaults() {
    if (!META) return;
    const gbp = META.gbp_to_inr;
    const cm = META.cost_model;
    if (cm && cm.measured) {
        document.getElementById('c-v_ret').value = Math.round(cm.measured.returned_order_value_pence / 100 * gbp);
        document.getElementById('c-v_kept').value = Math.round(cm.measured.kept_order_value_pence / 100 * gbp);
    }
    if (cm && cm.assumed) {
        document.getElementById('c-recovery').value = cm.assumed.goods_recovery_rate;
        document.getElementById('rv-c-recovery').textContent = cm.assumed.goods_recovery_rate.toFixed(2);
        document.getElementById('c-logistics').value = Math.round(cm.assumed.return_logistics_pence / 100 * gbp);
        document.getElementById('c-prevention').value = cm.assumed.prevention_rate;
        document.getElementById('rv-c-prevention').textContent = cm.assumed.prevention_rate.toFixed(2);
        document.getElementById('c-review').value = Math.round(cm.assumed.cost_review_pence / 100 * gbp);
        document.getElementById('c-abandon').value = cm.assumed.abandon_rate;
        document.getElementById('rv-c-abandon').textContent = cm.assumed.abandon_rate.toFixed(2);
        document.getElementById('c-margin').value = cm.assumed.contribution_margin_rate;
        document.getElementById('rv-c-margin').textContent = cm.assumed.contribution_margin_rate.toFixed(2);
    }
}

function initCostButton() {
    document.getElementById('recalc-cost-btn').addEventListener('click', recalcCost);
}

function renderInitialCostChart() {
    if (!SWEEP || !SWEEP.sweep || !SWEEP.sweep.length || !META) return;

    const sweep = SWEEP.sweep;
    const shipped = META.threshold;

    // Best point on the SHIPPED model's curve. /api/sweep filters the CSV to
    // the shipped model; it holds a second candidate's sweep too, and reading
    // it unfiltered used to report that other model's optimum here.
    let best = sweep[0];
    sweep.forEach(s => { if (s.total_cost_inr < best.total_cost_inr) best = s; });

    // Cost of flagging nothing, priced server-side by cost_model.py. This was
    // a fourth transcription of the cost model, complete with a hardcoded
    // fallback for the cost of a return.
    const costNoneINR = SWEEP.cost_flag_nothing_inr;

    renderCostMetrics(best.threshold, META.analytic_break_even, best.total_cost_inr, costNoneINR, shipped);
    drawCostChart(sweep.map(s => s.threshold), sweep.map(s => s.total_cost_inr), best, shipped, costNoneINR);

    const formula = META.cost_model?.formula || '';
    document.getElementById('cost-formula-caption').innerHTML =
        `Formula: <code>${escHtml(formula)}</code>. Reviewing a flagged order costs money whether or not the flag was right, ` +
        `and a caught return is only prevented ${((META.cost_model?.assumed?.prevention_rate || 0.3) * 100).toFixed(0)}% of the time.`;
}

async function recalcCost() {
    const btn = document.getElementById('recalc-cost-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Recalculating…';

    const body = {
        v_ret_inr: parseFloat(document.getElementById('c-v_ret').value),
        v_kept_inr: parseFloat(document.getElementById('c-v_kept').value),
        recovery: parseFloat(document.getElementById('c-recovery').value),
        logistics_inr: parseFloat(document.getElementById('c-logistics').value),
        prevention: parseFloat(document.getElementById('c-prevention').value),
        review_inr: parseFloat(document.getElementById('c-review').value),
        abandon: parseFloat(document.getElementById('c-abandon').value),
        margin: parseFloat(document.getElementById('c-margin').value),
    };

    try {
        const res = await fetch('/api/cost-optimal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        const shipped = data.shipped_threshold;
        renderCostMetrics(data.best_threshold, data.p_star, data.best_cost_inr, data.cost_flag_nothing_inr, shipped);

        if (data.recosted_sweep && data.recosted_sweep.length > 0) {
            const thresholds = data.recosted_sweep.map(r => r.threshold);
            const costs = data.recosted_sweep.map(r => r.total_cost_inr);
            const best = { threshold: data.best_threshold, total_cost_inr: data.best_cost_inr };
            drawCostChart(thresholds, costs, best, shipped, data.cost_flag_nothing_inr);
        }

        // Drift callout
        const drift = document.getElementById('cost-drift-callout');
        if (Math.abs(data.best_threshold - shipped) > 0.02) {
            drift.innerHTML = `
                <div class="callout callout-info" style="margin-top:16px;">
                    <span class="icon">ℹ️</span>
                    <span>Under <strong>these</strong> assumptions the optimum is ${data.best_threshold.toFixed(2)},
                    but the shipped model still operates at ${shipped.toFixed(2)}.
                    Moving a slider does not silently re-deploy anything — changing the operating point is
                    a decision someone signs off on.</span>
                </div>`;
            drift.style.display = 'block';
        } else {
            drift.style.display = 'none';
        }

        // Re-run sensitivity
        runSensitivity(body);

    } catch (e) {
        console.error('Cost recalc error:', e);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Recalculate optimal threshold';
    }
}

function renderCostMetrics(bestThr, pStar, bestCostINR, costNoneINR, shipped) {
    const savedPct = costNoneINR > 0 ? ((costNoneINR - bestCostINR) / costNoneINR * 100).toFixed(1) : '—';
    document.getElementById('cost-metrics').innerHTML = `
        <div class="metric">
            <div class="metric-label">Cost-optimal threshold</div>
            <div class="metric-value">${bestThr != null ? bestThr.toFixed(2) : '—'}</div>
            <div class="metric-delta">${bestThr != null ? (bestThr - shipped > 0 ? '+' : '') + (bestThr - shipped).toFixed(2) + ' vs shipped' : ''}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Analytic break-even</div>
            <div class="metric-value">${pStar != null ? pStar.toFixed(3) : '—'}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Cost at that point</div>
            <div class="metric-value">₹${bestCostINR != null ? fmtNum(bestCostINR) : '—'}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Saved vs flagging nothing</div>
            <div class="metric-value">${savedPct}%</div>
        </div>
    `;
}

function drawCostChart(thresholds, costsINR, best, shipped, costNoneINR) {
    if (costChart) costChart.destroy();
    const ctx = document.getElementById('cost-chart').getContext('2d');

    // Index of the best threshold. Nearest match, not an exact one: a tolerance
    // test that finds nothing silently left the marker sitting on the first
    // point of the curve, labelled with the right number in the wrong place.
    let bestIdx = 0;
    thresholds.forEach((t, i) => {
        if (Math.abs(t - best.threshold) < Math.abs(thresholds[bestIdx] - best.threshold)) bestIdx = i;
    });

    // Plugin for annotations (shipped line + flag-nothing line)
    const annotationPlugin = {
        id: 'customAnnotations',
        afterDraw(chart) {
            const { ctx, scales: { x, y } } = chart;
            ctx.save();

            // Shipped threshold vertical line
            const xShipped = x.getPixelForValue(shipped);
            ctx.beginPath();
            ctx.strokeStyle = '#8a8980';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.moveTo(xShipped, y.top);
            ctx.lineTo(xShipped, y.bottom);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#8a8980';
            ctx.font = '10px Inter';
            ctx.textAlign = 'right';
            ctx.save();
            ctx.translate(xShipped - 6, y.top + 14);
            ctx.fillText(`shipped ${shipped.toFixed(2)}`, 0, 0);
            ctx.restore();

            // Flag nothing horizontal line
            if (costNoneINR) {
                const yNone = y.getPixelForValue(costNoneINR);
                ctx.beginPath();
                ctx.strokeStyle = '#8a8980';
                ctx.lineWidth = 1;
                ctx.setLineDash([6, 3]);
                ctx.moveTo(x.left, yNone);
                ctx.lineTo(x.right, yNone);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = '#8a8980';
                ctx.font = '10px Inter';
                ctx.textAlign = 'right';
                ctx.fillText('flag nothing', x.right, yNone - 4);
            }

            ctx.restore();
        }
    };

    costChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: thresholds,
            datasets: [{
                data: costsINR,
                borderColor: '#3987e5',
                backgroundColor: 'rgba(57,135,229,0.06)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHitRadius: 8,
                pointHoverRadius: 4,
                pointHoverBackgroundColor: '#3987e5',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#222221',
                    borderColor: '#3a3a37',
                    borderWidth: 1,
                    titleFont: { family: 'JetBrains Mono', size: 11 },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    callbacks: {
                        title: items => `threshold ${items[0].label}`,
                        label: ctx => `₹${fmtNum(ctx.raw)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(58,58,55,0.25)' },
                    ticks: {
                        color: '#8a8980',
                        font: { size: 10, family: 'JetBrains Mono' },
                        maxTicksLimit: 15,
                        callback: (val, idx) => thresholds[idx] % 0.1 < 0.015 ? thresholds[idx].toFixed(1) : ''
                    },
                    title: { display: true, text: 'decision threshold', color: '#8a8980', font: { size: 11, family: 'Inter' } }
                },
                y: {
                    grid: { color: 'rgba(58,58,55,0.25)' },
                    ticks: {
                        color: '#8a8980',
                        font: { size: 10, family: 'JetBrains Mono' },
                        callback: val => `₹${fmtNum(val)}`
                    },
                    title: { display: true, text: 'total cost on held-out orders (INR)', color: '#8a8980', font: { size: 11, family: 'Inter' } }
                }
            }
        },
        plugins: [annotationPlugin, {
            // Draw the best-point marker
            id: 'bestPoint',
            afterDraw(chart) {
                const ds = chart.data.datasets[0];
                const meta = chart.getDatasetMeta(0);
                const point = meta.data[bestIdx];
                if (!point) return;
                const { ctx } = chart;
                ctx.save();
                ctx.beginPath();
                ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
                ctx.fillStyle = '#3987e5';
                ctx.fill();
                ctx.strokeStyle = '#1a1a19';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Label
                ctx.fillStyle = '#3987e5';
                ctx.font = '11px JetBrains Mono';
                ctx.textAlign = 'left';
                ctx.fillText(`optimum ${best.threshold.toFixed(2)}`, point.x + 14, point.y - 6);
                ctx.fillText(`₹${fmtNum(best.total_cost_inr)}`, point.x + 14, point.y + 8);
                ctx.restore();
            }
        }]
    });
}

// Sensitivity analysis (client-side, using sweep data)
async function runSensitivity(baseInputs) {
    // Each scenario scales a finished composite cost, which is what the row
    // label claims. Halving the cost of a return by doubling the recovery
    // shortfall is a different question: recovery only touches one of the three
    // terms in c_return, so those rows disagreed with the same-named rows in
    // the Streamlit app for identical inputs.
    const scenarios = [
        { name: 'cost of a return ×0.5', overrides: { c_return_mult: 0.5 } },
        { name: 'cost of a return ×2', overrides: { c_return_mult: 2 } },
        { name: 'prevention 10%', overrides: { prevention: 0.10 } },
        { name: 'prevention 90%', overrides: { prevention: 0.90 } },
        { name: 'review cost ×5', overrides: { c_review_mult: 5 } },
    ];

    const tbody = document.getElementById('sensitivity-tbody');
    tbody.innerHTML = '';

    for (const sc of scenarios) {
        const body = { ...baseInputs, ...sc.overrides };
        try {
            const res = await fetch('/api/cost-optimal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            if (data.error) continue;

            // Find the sweep row at the best threshold to get flag_rate and recall
            let flagRate = '—', recall = '—';
            if (SWEEP && SWEEP.sweep) {
                const row = SWEEP.sweep.find(s => Math.abs(s.threshold - data.best_threshold) < 0.005);
                if (row) {
                    flagRate = row.flag_rate.toFixed(3);
                    recall = (row.tp / (row.tp + row.fn)).toFixed(3);
                }
            }

            const tr = document.createElement('tr');
            tr.innerHTML = `<td style="font-family:Inter; color:var(--ink-2);">${escHtml(sc.name)}</td>
                            <td>${data.best_threshold != null ? data.best_threshold.toFixed(2) : '—'}</td>
                            <td>${flagRate}</td>
                            <td>${recall}</td>`;
            tbody.appendChild(tr);
        } catch (e) { /* skip */ }
    }
}


// --- holdout tab
function renderHoldoutTab() {
    if (!META || !META.holdout) return;
    const h = META.holdout;
    const baseRate = h.base_rate_test;

    document.getElementById('holdout-intro').textContent =
        `${h.n_test.toLocaleString()} orders after ${h.split_date}, never seen in training. ` +
        `The split is chronological — every training order precedes every test order.`;

    document.getElementById('holdout-metrics').innerHTML = `
        <div class="metric">
            <div class="metric-label">Precision</div>
            <div class="metric-value">${h.precision.toFixed(3)}</div>
            <div class="metric-delta">${(h.precision / baseRate).toFixed(2)}× base rate</div>
        </div>
        <div class="metric">
            <div class="metric-label">Recall</div>
            <div class="metric-value">${h.recall.toFixed(3)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value">${h.roc_auc.toFixed(3)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">PR-AUC</div>
            <div class="metric-value">${h.pr_auc.toFixed(3)}</div>
            <div class="metric-delta">${(h.pr_auc / baseRate).toFixed(2)}× base rate</div>
        </div>
    `;

    // Accuracy callout
    document.getElementById('accuracy-callout').innerHTML = `
        <div class="callout callout-warning">
            <span class="icon">⚠️</span>
            <span><strong>Accuracy is ${h.accuracy.toFixed(3)} — below the ${(1 - baseRate).toFixed(3)} you get by
            flagging nothing.</strong> That is expected at a recall-heavy operating point, and it is why accuracy
            is not the selection metric here. The base rate is ${(baseRate * 100).toFixed(2)}%; quote it beside
            every number above or none of them.</span>
        </div>
    `;

    // Confusion matrix
    const c = h.confusion;
    document.getElementById('cm-tn').textContent = c.tn.toLocaleString();
    document.getElementById('cm-fp').textContent = c.fp.toLocaleString();
    document.getElementById('cm-fn').textContent = c.fn.toLocaleString();
    document.getElementById('cm-tp').textContent = c.tp.toLocaleString();
    document.getElementById('cm-title').textContent = `Confusion matrix at threshold ${META.threshold}`;

    // Cost model inputs table
    const cm = META.cost_model;
    const tbody = document.getElementById('cost-inputs-tbody');
    tbody.innerHTML = '';
    if (cm) {
        const addRow = (name, val, status) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td style="font-family:Inter; color:var(--ink-2);">${escHtml(name)}</td>
                            <td>${typeof val === 'number' ? val : escHtml(String(val))}</td>
                            <td><span style="color:${status === 'measured' ? 'var(--success)' : 'var(--warning)'};">${status}</span></td>`;
            tbody.appendChild(tr);
        };
        if (cm.measured) Object.entries(cm.measured).forEach(([k, v]) => addRow(k, v, 'measured'));
        if (cm.assumed) Object.entries(cm.assumed).forEach(([k, v]) => addRow(k, v, 'assumed'));
    }

    // Not measured prose
    const breakEven = META.analytic_break_even || 0.198;
    const gbpToInr = META.gbp_to_inr || 75;
    document.getElementById('not-measured-prose').innerHTML = `
        <ul>
            <li><strong>Chargebacks are unevaluated.</strong> No public dataset carries disputes, so
            <code>label_disputed</code> is always 0. The dispute path exists in the schema and the API
            and reports <strong>no metrics</strong>.</li>
            <li><strong>The seven assumed cost inputs are not measured.</strong> This dataset contains no
            cost data at all. Read the <em>shape</em> — the optimum sits near ${breakEven.toFixed(2)},
            well below the 0.5 default, and moves slowly — not the cash figures.</li>
            <li><strong>The source is a UK wholesale gift retailer.</strong> Median order £304 (₹22,833),
            customers mostly businesses, so return behaviour is B2B-flavoured. <strong>This is not Indian data.</strong>
            Amounts are stored and measured in GBP and displayed in INR at an assumed ${gbpToInr}/GBP.</li>
            <li><strong>22.8% of source rows have no customer ID</strong> and are excluded entirely, because
            a return that cannot be attributed to a customer cannot become a label.</li>
            <li>No payment method, addresses or discount data exist in the source, so those planned features do not exist.</li>
        </ul>
    `;

    // Model selection
    document.getElementById('model-rationale').textContent = META.rationale || '';
    document.getElementById('model-selection-caption').textContent =
        `Selection rule: ${META.selection_rule || ''}. ` +
        `The ${((META.marginal_gain_threshold || 0.02) * 100).toFixed(0)}% bar was committed to before the result was known; ` +
        `measured gain was ${META.relative_gain_over_logreg != null ? (META.relative_gain_over_logreg > 0 ? '+' : '') + (META.relative_gain_over_logreg * 100).toFixed(2) + '%' : '—'}.`;
}


// --- toggles
function initToggles() {
    setupToggle('sensitivity-toggle', 'sensitivity-content');
    setupToggle('model-selection-toggle', 'model-selection-content');
}

function setupToggle(btnId, contentId) {
    const btn = document.getElementById(btnId);
    const content = document.getElementById(contentId);
    if (!btn || !content) return;
    btn.addEventListener('click', () => {
        const isOpen = content.classList.toggle('open');
        btn.textContent = (isOpen ? '▾ ' : '▸ ') + btn.textContent.substring(2);
    });
}


// --- helpers
function fmtNum(n) {
    if (n == null) return '—';
    return Math.round(n).toLocaleString('en-IN');
}

function escHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
