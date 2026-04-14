// ============================================
// STOCK PREDICTION UI - JavaScript
// ============================================

let priceChart = null;

// Initialize app on page load
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
});

function initializeApp() {
    const predictBtn = document.getElementById('predictBtn');
    const stockSymbolInput = document.getElementById('stockSymbol');

    // Allow Enter key to trigger prediction
    stockSymbolInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            predictBtn.click();
        }
    });

    // Auto-uppercase input
    stockSymbolInput.addEventListener('input', function () {
        this.value = this.value.toUpperCase();
    });

    // Predict button click
    predictBtn.addEventListener('click', function () {
        const symbol = stockSymbolInput.value.trim();
        
        if (!symbol || symbol.length === 0) {
            showError('Please enter a stock symbol');
            return;
        }
        
        if (symbol.length > 5) {
            showError('Stock symbol must be 5 characters or less');
            return;
        }

        makePrediction(symbol);
    });
}

// Show error messages
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    
    errorText.textContent = message;
    errorDiv.style.display = 'flex';
    
    // Auto-hide after 6 seconds
    setTimeout(() => {
        closeAlert(errorDiv.querySelector('.alert-close'));
    }, 6000);
}

// Close alert
function closeAlert(btn) {
    const alert = btn.closest('.alert');
    alert.style.display = 'none';
}

// Make prediction API call
async function makePrediction(symbol) {
    const predictBtn = document.getElementById('predictBtn');
    const btnContent = predictBtn.querySelector('.btn-content');
    const btnLoader = predictBtn.querySelector('.btn-loader');
    const loadingState = document.getElementById('loadingState');
    const resultsSection = document.getElementById('resultsSection');
    const errorDiv = document.getElementById('errorMessage');

    // Clear previous errors
    errorDiv.style.display = 'none';

    // Show loading state
    predictBtn.disabled = true;
    btnContent.style.display = 'none';
    btnLoader.style.display = 'flex';
    loadingState.style.display = 'block';
    resultsSection.style.display = 'none';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symbol: symbol })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to get prediction');
        }

        // Hide loading and display results
        loadingState.style.display = 'none';
        displayResults(data);

    } catch (error) {
        loadingState.style.display = 'none';
        showError(error.message);
    } finally {
        // Reset button state
        predictBtn.disabled = false;
        btnContent.style.display = 'flex';
        btnLoader.style.display = 'none';
    }
}

// Display prediction results
function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');

    // Update prediction header
    document.getElementById('predSymbol').textContent = data.symbol;
    
    const predictionResult = document.getElementById('predictionResult');
    const predictionText = document.getElementById('predictionText');
    const probabilityText = document.getElementById('probabilityText');
    
    const isUp = data.prediction === 'UP';
    
    predictionResult.textContent = data.prediction;
    predictionResult.className = 'prediction-badge ' + (isUp ? 'up' : 'down');
    predictionResult.innerHTML = (isUp ? '📈 UP' : '📉 DOWN');
    
    predictionText.textContent = data.prediction;
    predictionText.style.color = isUp ? '#38ef7d' : '#f5576c';
    
    probabilityText.textContent = `Confidence: ${data.probability}%`;

    // Update current price
    document.getElementById('currentPrice').textContent = `$${data.current_price.toFixed(2)}`;

    // Update sentiment
    const sentiment = data.avg_sentiment;
    const sentimentValue = document.getElementById('sentimentValue');
    const sentimentScore = document.getElementById('sentimentScore');
    
    if (sentiment > 0.1) {
        sentimentValue.textContent = 'Bullish';
        sentimentValue.style.color = '#38ef7d';
    } else if (sentiment < -0.1) {
        sentimentValue.textContent = 'Bearish';
        sentimentValue.style.color = '#f5576c';
    } else {
        sentimentValue.textContent = 'Neutral';
        sentimentValue.style.color = '#667eea';
    }
    
    sentimentScore.textContent = `Score: ${sentiment.toFixed(3)}`;

    // Update data table
    updateDataTable(data.recent_data);

    // Update chart
    setTimeout(() => {
        updateChart(data.recent_data);
    }, 300);

    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Update data table
function updateDataTable(data) {
    const tableBody = document.getElementById('tableBody');
    tableBody.innerHTML = '';

    if (!data || data.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #999;">No data available</td></tr>';
        return;
    }

    data.forEach((row, index) => {
        const tr = document.createElement('tr');
        
        tr.innerHTML = `
            <td>${formatDate(row.Date)}</td>
            <td>$${parseFloat(row.Open).toFixed(2)}</td>
            <td>$${parseFloat(row.High).toFixed(2)}</td>
            <td>$${parseFloat(row.Low).toFixed(2)}</td>
            <td><strong>$${parseFloat(row.Close).toFixed(2)}</strong></td>
            <td>${formatVolume(row.Volume)}</td>
        `;
        
        tableBody.appendChild(tr);
        
        // Stagger animation
        setTimeout(() => {
            tr.style.animation = 'fadeIn 0.4s ease-out';
        }, index * 30);
    });
}

// Update chart
function updateChart(data) {
    const canvas = document.getElementById('priceChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Extract data
    const dates = data.map(d => formatDate(d.Date));
    const closes = data.map(d => parseFloat(d.Close));
    const opens = data.map(d => parseFloat(d.Open));
    const highs = data.map(d => parseFloat(d.High));
    const lows = data.map(d => parseFloat(d.Low));

    // Calculate min/max for better scaling
    const minPrice = Math.min(...lows) * 0.99;
    const maxPrice = Math.max(...highs) * 1.01;

    // Destroy existing chart if it exists
    if (priceChart) {
        priceChart.destroy();
    }

    // Create new chart
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Close Price',
                    data: closes,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#667eea',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointHoverRadius: 6,
                },
                {
                    label: 'High',
                    data: highs,
                    borderColor: '#11998e',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 2,
                    pointBackgroundColor: '#11998e',
                },
                {
                    label: 'Low',
                    data: lows,
                    borderColor: '#f5576c',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 2,
                    pointBackgroundColor: '#f5576c',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: { size: 12, weight: '600' },
                        usePointStyle: true,
                        padding: 15,
                        boxWidth: 6,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: { size: 13, weight: 'bold' },
                    bodyFont: { size: 11 },
                    callbacks: {
                        label: function (context) {
                            return context.dataset.label + ': $' + context.parsed.y.toFixed(2);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: minPrice,
                    max: maxPrice,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        drawBorder: false,
                    },
                    ticks: {
                        callback: function (value) {
                            return '$' + value.toFixed(0);
                        },
                        font: { size: 11 }
                    }
                },
                x: {
                    grid: {
                        display: false,
                        drawBorder: false,
                    },
                    ticks: {
                        font: { size: 11 }
                    }
                }
            }
        }
    });
}

// Format date string
function formatDate(dateStr) {
    if (!dateStr) return '-';
    
    try {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric', 
            year: '2-digit' 
        });
    } catch (e) {
        return dateStr;
    }
}

// Format volume number
function formatVolume(volume) {
    if (!volume) return '-';
    
    volume = parseInt(volume);
    
    if (volume >= 1_000_000) {
        return (volume / 1_000_000).toFixed(1) + 'M';
    } else if (volume >= 1_000) {
        return (volume / 1_000).toFixed(1) + 'K';
    }
    
    return volume.toLocaleString();
}

// Format large numbers
function formatNumber(num) {
    if (!num) return '-';
    return parseInt(num).toLocaleString();
}
