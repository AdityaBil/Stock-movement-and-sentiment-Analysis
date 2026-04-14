/**
 * StockAI - Frontend Logic Main Engine
 * Powered by Aditya Bilgaiyan
 */

let globalChartInstance = null;
let currentStockData = [];

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

function initApp() {
    const predictBtn = document.getElementById('predictBtn');
    const inputEl = document.getElementById('stockSymbol');
    
    // Auto captilize
    inputEl.addEventListener('input', function() {
        this.value = this.value.toUpperCase();
    });

    // Enter to submit
    inputEl.addEventListener('keypress', (e) => {
        if(e.key === 'Enter') {
            predictBtn.click();
            e.preventDefault();
        }
    });

    // Quick tag binding
    const tags = document.querySelectorAll('.symbol-tag');
    tags.forEach(t => {
        t.addEventListener('click', () => {
            inputEl.value = t.innerText;
            predictBtn.click();
        });
    });

    // Main action binding
    predictBtn.addEventListener('click', async () => {
        const symbol = inputEl.value.trim().toUpperCase();
        if(!symbol || symbol.length > 5) {
            triggerError("Invalid Stock Symbol provided.");
            return;
        }

        await executePrediction(symbol);
    });

    // Timeframe Pill listeners
    const pills = document.querySelectorAll('.chart-pill');
    pills.forEach(pill => {
        pill.addEventListener('click', (e) => {
            if (currentStockData.length === 0) return;
            
            pills.forEach(p => p.classList.remove('active'));
            e.target.classList.add('active');
            
            const range = e.target.innerText;
            let sliceCount = 10;
            let title = "10 Days";
            
            if (range === '10D') { sliceCount = 10; title = "10 Days"; }
            else if (range === '1M') { sliceCount = 22; title = "1 Month"; }
            else if (range === '3M') { sliceCount = 60; title = "3 Months"; }
            else if (range === 'D') { sliceCount = 10; title = "10 Days"; }
            else if (range === 'W') { sliceCount = 22; title = "1 Month"; }
            else if (range === 'M') { sliceCount = 60; title = "3 Months"; }
            
            document.querySelector('.chart-panel .header-title h3').innerText = `Price Action History (${title})`;
            
            renderStunningChart(currentStockData.slice(-sliceCount));
            injectTableData(currentStockData.slice(-sliceCount));
        });
    });
}

function triggerError(msg) {
    const b = document.getElementById('errorMessage');
    document.getElementById('errorText').innerText = msg;
    b.style.display = 'flex';
    setTimeout(() => { b.style.display = 'none'; }, 6000);
}

window.closeAlert = function(btn) {
    btn.parentElement.style.display = 'none';
}

async function executePrediction(symbol) {
    const loadingView = document.getElementById('loadingState');
    const resultView = document.getElementById('resultsSection');
    const btn = document.getElementById('predictBtn');
    
    // reset UI
    document.getElementById('errorMessage').style.display='none';
    loadingView.style.display = 'flex';
    resultView.style.display = 'none';
    resultView.style.opacity = '0';
    
    // button loading
    btn.classList.add('loading-btn');
    btn.querySelector('.btn-text').style.opacity = '0';
    btn.querySelector('.btn-icon').style.opacity = '0';
    document.getElementById('btnLoader').style.display = 'block';

    try {
        const payload = { symbol: symbol };
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        
        if(!res.ok) {
            throw new Error(data.error || "Critical backend execution failure.");
        }

        buildDashboard(data);

    } catch(err) {
        console.error(err);
        triggerError(err.message);
    } finally {
        // Stop loader
        loadingView.style.display = 'none';
        btn.classList.remove('loading-btn');
        btn.querySelector('.btn-text').style.opacity = '1';
        btn.querySelector('.btn-icon').style.opacity = '1';
        document.getElementById('btnLoader').style.display = 'none';
    }
}

function buildDashboard(data) {
    const resultView = document.getElementById('resultsSection');
    
    // Primary Signal Data
    document.getElementById('predSymbol').innerText = data.symbol;
    document.getElementById('currentPrice').innerText = `$${parseFloat(data.current_price).toFixed(2)}`;
    
    // Handle specific UP vs DOWN themes
    const isUp = (data.prediction === 'UP');
    const ring = document.getElementById('signalRing');
    const resultWord = document.getElementById('predictionResult');
    const probDisplay = document.getElementById('probabilityText');
    
    ring.className = `signal-ring ${isUp ? 'up' : 'down'}`;
    resultWord.className = `ring-core ${isUp ? 'text-up' : 'text-down'}`;
    resultWord.innerHTML = isUp ? '<i class="fas fa-arrow-trend-up"></i> UP' : '<i class="fas fa-arrow-trend-down"></i> DN';
    
    probDisplay.innerText = `${data.probability.toFixed(1)}%`;
    probDisplay.style.color = isUp ? 'var(--success)' : 'var(--danger)';

    // Sentiment Output
    const sentVal = parseFloat(data.avg_sentiment);
    const sentTextDiv = document.getElementById('sentimentValue');
    const sentScoreDiv = document.getElementById('sentimentScore');
    
    if (sentVal > 0.1) {
        sentTextDiv.innerText = 'Bullish Bias';
        sentTextDiv.style.color = 'var(--success)';
    } else if (sentVal < -0.1) {
        sentTextDiv.innerText = 'Bearish Bias';
        sentTextDiv.style.color = 'var(--danger)';
    } else {
        sentTextDiv.innerText = 'Neutral Drift';
        sentTextDiv.style.color = 'var(--accent-primary)';
    }
    sentScoreDiv.innerText = `FinBERT Score: ${sentVal.toFixed(2)}`;

    // News Integration
    const newsSection = document.getElementById('newsSection');
    const tickerBlock = document.getElementById('tickerContent');
    tickerBlock.innerHTML = '';
    if(data.headlines && data.headlines.length > 0) {
        data.headlines.forEach(hl => {
            const span = document.createElement('span');
            span.className = 'ticker-item';
            span.innerText = hl;
            tickerBlock.appendChild(span);
        });
        newsSection.style.display = 'block';
    } else {
        newsSection.style.display = 'none';
    }

    // Store Data Globally
    currentStockData = data.recent_data;

    // Tabular Data Rendering (default 10 days)
    injectTableData(currentStockData.slice(-10));
    
    // Reset Default Pills
    document.querySelectorAll('.chart-pill').forEach(p => p.classList.remove('active'));
    const firstPill = document.querySelector('.chart-pill');
    if(firstPill) firstPill.classList.add('active');
    document.querySelector('.chart-panel .header-title h3').innerText = `Price Action History (10 Days)`;

    // Fade UI in rapidly
    resultView.style.display = 'block';
    setTimeout(() => {
        resultView.style.opacity = '1';
        resultView.style.transition = 'opacity 0.6s ease';
        renderStunningChart(currentStockData.slice(-10));
    }, 50);

    // Smooth Scroll down to dashboard
    resultView.scrollIntoView({ behavior: 'smooth', block: 'start' });
}


function injectTableData(dataArr) {
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    if(!dataArr || dataArr.length === 0) return;

    dataArr.slice().reverse().forEach(row => {
        const tr = document.createElement('tr');
        
        const dateStr = formatDateShort(row.Date);
        const pClose = `$${parseFloat(row.Close).toFixed(2)}`;
        const pVol = formatVol(row.Volume);

        tr.innerHTML = `
            <td>${dateStr}</td>
            <td style="color: var(--text-main); font-weight: 700;">${pClose}</td>
            <td style="color: var(--text-muted);">${pVol}</td>
        `;
        tbody.appendChild(tr);
    });
}


function renderStunningChart(dataArr) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    if(globalChartInstance) {
        globalChartInstance.destroy();
    }

    const labels = dataArr.map(d => formatDateShort(d.Date));
    const closes = dataArr.map(d => parseFloat(d.Close));
    const highs = dataArr.map(d => parseFloat(d.High));
    const lows = dataArr.map(d => parseFloat(d.Low));

    // Dynamic Gradient Background under line
    const gradientFill = ctx.createLinearGradient(0, 0, 0, 350);
    gradientFill.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
    gradientFill.addColorStop(1, 'rgba(99, 102, 241, 0.0)');

    const chartConfig = {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Close Point',
                    data: closes,
                    borderColor: '#a855f7',
                    borderWidth: 3,
                    fill: true,
                    backgroundColor: gradientFill,
                    tension: 0.4, // Smooth curve
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#6366f1',
                    pointBorderWidth: 2,
                    z: 5
                },
                {
                    label: 'High Bounds',
                    data: highs,
                    borderColor: 'rgba(16, 185, 129, 0.4)',
                    borderWidth: 1,
                    borderDash: [5,5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Low Bounds',
                    data: lows,
                    borderColor: 'rgba(244, 63, 94, 0.4)',
                    borderWidth: 1,
                    borderDash: [5,5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(5, 5, 8, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#e2e8f0',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false, drawBorder: false },
                    ticks: { color: '#64748b', font: { family: 'JetBrains Mono' } }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.03)', drawBorder: false },
                    ticks: {
                        color: '#64748b',
                        font: { family: 'JetBrains Mono' },
                        callback: (v) => `$${v}`
                    }
                }
            }
        }
    };

    globalChartInstance = new Chart(ctx, chartConfig);
}

// Utils
function formatDateShort(dateStr) {
    if(!dateStr) return '';
    try {
        const d = new Date(dateStr);
        return d.toLocaleDateString('en-US', {month: 'short', day: 'numeric'});
    } catch {
        return dateStr;
    }
}

function formatVol(volStr) {
    if(!volStr) return '';
    const v = parseInt(volStr);
    if(v > 1000000) return (v/1000000).toFixed(1) + 'M';
    if(v > 1000) return (v/1000).toFixed(1) + 'K';
    return v.toString();
}
