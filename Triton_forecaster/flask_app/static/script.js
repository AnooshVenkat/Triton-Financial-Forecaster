document.addEventListener('DOMContentLoaded', () => {
    const industrySelect = document.getElementById('industry-select');
    const stockSelect = document.getElementById('stock-select');
    const predictBtn = document.getElementById('predict-btn');
    const resultsContainer = document.getElementById('results-container');
    const resultsContent = document.getElementById('results-content');
    const spinner = document.getElementById('spinner');
    const resultTicker = document.getElementById('result-ticker');
    const predictionResult = document.getElementById('prediction-result');
    const sentimentResult = document.getElementById('sentiment-result');
    const priceChart = document.getElementById('price-chart');


    function populateIndustries() {
        const industries = Object.keys(STOCKS_DATA);
        industries.forEach(industry => {
            const option = document.createElement('option');
            option.value = industry;
            option.textContent = industry;
            industrySelect.appendChild(option);
        });
    }

    function populateStocks(industry) {
        stockSelect.innerHTML = '';
        
        const stocks = STOCKS_DATA[industry];
        if (stocks) {
            Object.keys(stocks).forEach(stockName => {
                const option = document.createElement('option');
                option.value = stocks[stockName]; 
                option.textContent = stockName; 
                stockSelect.appendChild(option);
            });
        }
    }


    industrySelect.addEventListener('change', () => {
        populateStocks(industrySelect.value);
    });

    predictBtn.addEventListener('click', async () => {
        const ticker = stockSelect.value;
        const selectedStockName = stockSelect.options[stockSelect.selectedIndex].text;

        resultsContainer.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        spinner.classList.remove('hidden');
        priceChart.src = ""; 

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker: ticker }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Prediction request failed');
            }

            const data = await response.json();
            resultTicker.textContent = `${selectedStockName} (${data.ticker})`;

            predictionResult.textContent = data.prediction;
            predictionResult.className = data.prediction === 'UP' ? 'up' : 'down';

            const sentimentScore = parseFloat(data.sentiment_score).toFixed(4);
            let sentimentText = `Score: ${sentimentScore}`;
            if (sentimentScore > 0.05) sentimentText += " (Positive)";
            else if (sentimentScore < -0.05) sentimentText += " (Negative)";
            else sentimentText += " (Neutral)";
            sentimentResult.textContent = sentimentText;
            priceChart.src = data.chart_url;
            resultsContent.classList.remove('hidden'); 

        } catch (error) {
            console.error('Error:', error);
            resultsContent.innerHTML = `<p style="color: var(--accent-red);">Error: ${error.message}</p>`;
            resultsContent.classList.remove('hidden');
        } finally {
            spinner.classList.add('hidden');
        }
    });

    populateIndustries();
    populateStocks(industrySelect.value); 
});