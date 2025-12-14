// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
let elements = {};

// Initialize extension
document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    elements = {
        form: document.getElementById('vmForm'),
        predictBtn: document.getElementById('predictBtn'),
        btnText: document.getElementById('btnText'),
        btnLoader: document.getElementById('btnLoader'),
        results: document.getElementById('results'),
        errorMessage: document.getElementById('errorMessage'),
        apiStatus: document.getElementById('apiStatus'),
        statusText: document.getElementById('statusText'),
        
        // Results elements
        monthlyCost: document.getElementById('monthlyCost'),
        priceCategory: document.getElementById('priceCategory'),
        probLow: document.getElementById('probLow'),
        probMedium: document.getElementById('probMedium'),
        probHigh: document.getElementById('probHigh'),
        probLowText: document.getElementById('probLowText'),
        probMediumText: document.getElementById('probMediumText'),
        probHighText: document.getElementById('probHighText'),
        clusterGroup: document.getElementById('clusterGroup'),
        totalClusters: document.getElementById('totalClusters'),
        
        // Sentiment elements
        sentimentSection: document.getElementById('sentimentSection'),
        sentimentEmoji: document.getElementById('sentimentEmoji'),
        sentimentLabel: document.getElementById('sentimentLabel'),
        sentimentMeaning: document.getElementById('sentimentMeaning'),
        valueScore: document.getElementById('valueScore'),
        
        // Recommendations
        getRecommendations: document.getElementById('getRecommendations'),
        recBtnText: document.getElementById('recBtnText'),
        recBtnLoader: document.getElementById('recBtnLoader'),
        recommendationsList: document.getElementById('recommendationsList'),
        recommendationsContainer: document.getElementById('recommendationsContainer')
    };

    // Check API health
    checkAPIHealth();

    // Form submission
    elements.form.addEventListener('submit', handlePrediction);

    // Recommendations button
    elements.getRecommendations.addEventListener('click', handleRecommendations);

    // GPU count change handler
    document.getElementById('gpuCount').addEventListener('change', function(e) {
        const gpuModel = document.getElementById('gpuModel');
        if (parseInt(e.target.value) === 0) {
            gpuModel.value = 'none';
            gpuModel.disabled = true;
        } else {
            gpuModel.disabled = false;
            if (gpuModel.value === 'none') {
                gpuModel.value = 't4'; // Default to T4
            }
        }
    });

    // Load saved configuration
    loadSavedConfig();
});

// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            updateAPIStatus(true, 'API Connected');
        } else {
            updateAPIStatus(false, 'API Error');
        }
    } catch (error) {
        updateAPIStatus(false, 'API Offline - Start server with: python main.py');
    }
}

// Update API status indicator
function updateAPIStatus(isOnline, message) {
    elements.statusText.textContent = message;
    elements.apiStatus.className = isOnline ? 'status-bar status-online' : 'status-bar status-offline';
}

// Handle prediction
async function handlePrediction(e) {
    e.preventDefault();
    
    hideError();
    showLoading(true);
    elements.results.style.display = 'none';

    // Get form values
    const formData = {
        vcpus: parseFloat(document.getElementById('vcpus').value),
        memory_gb: parseFloat(document.getElementById('memory').value),
        boot_disk_gb: parseFloat(document.getElementById('storage').value),
        gpu_count: parseFloat(document.getElementById('gpuCount').value),
        gpu_model: document.getElementById('gpuModel').value,
        usage_hours_month: parseFloat(document.getElementById('usageHours').value)
    };

    // Save configuration
    saveConfig(formData);

    try {
        const response = await fetch(`${API_BASE_URL}/predict/simplified`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }

        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        showError(`Error: ${error.message}. Make sure the API server is running (python main.py)`);
    } finally {
        showLoading(false);
    }
}

// Display prediction results
function displayResults(data) {
    // Cost and category
    elements.monthlyCost.textContent = data.regression.monthly_cost_formatted;
    elements.priceCategory.textContent = data.classification.category;
    elements.priceCategory.className = `category-${data.classification.category.toLowerCase()}`;

    // Probabilities
    const probs = data.classification.probabilities;
    updateProbability('Low', probs.Low);
    updateProbability('Medium', probs.Medium);
    updateProbability('High', probs.High);

    // Cluster info
    elements.clusterGroup.textContent = `Cluster ${data.clustering.cluster}`;
    elements.totalClusters.textContent = data.clustering.total_clusters;

    // Sentiment analysis
    if (data.sentiment) {
        displaySentiment(data.sentiment);
    } else {
        elements.sentimentSection.style.display = 'none';
    }

    // Show results
    elements.results.style.display = 'block';
    
    // Reset recommendations
    elements.recommendationsList.style.display = 'none';
    elements.recommendationsContainer.innerHTML = '';
    
    // Scroll to results
    elements.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Update probability bar
function updateProbability(category, value) {
    const percentage = (value * 100).toFixed(1);
    elements[`prob${category}`].style.width = `${percentage}%`;
    elements[`prob${category}Text`].textContent = `${percentage}%`;
}

// Display sentiment analysis
function displaySentiment(sentiment) {
    elements.sentimentSection.style.display = 'block';
    
    const emojiMap = {
        'positive': '😊',
        'neutral': '👍',
        'negative': '🤔'
    };
    
    elements.sentimentEmoji.textContent = emojiMap[sentiment.sentiment] || '👍';
    elements.sentimentLabel.textContent = sentiment.sentiment.toUpperCase();
    elements.sentimentMeaning.textContent = sentiment.meaning;
    elements.valueScore.textContent = sentiment.value_score.toFixed(2);
}

// Handle recommendations
async function handleRecommendations() {
    hideError();
    showRecommendationsLoading(true);

    const formData = {
        vcpus: parseFloat(document.getElementById('vcpus').value),
        memory_gb: parseFloat(document.getElementById('memory').value),
        boot_disk_gb: parseFloat(document.getElementById('storage').value),
        gpu_count: parseFloat(document.getElementById('gpuCount').value),
        gpu_model: document.getElementById('gpuModel').value,
        usage_hours_month: parseFloat(document.getElementById('usageHours').value)
    };

    try {
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Failed to get recommendations');
        }

        const data = await response.json();
        displayRecommendations(data.recommendations);
        
    } catch (error) {
        showError(`Error loading recommendations: ${error.message}`);
    } finally {
        showRecommendationsLoading(false);
    }
}

// Display recommendations
function displayRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        elements.recommendationsContainer.innerHTML = '<p class="no-results">No similar VMs found</p>';
        elements.recommendationsList.style.display = 'block';
        return;
    }

    elements.recommendationsContainer.innerHTML = recommendations.map((rec, index) => `
        <div class="recommendation-card">
            <div class="rec-header">
                <span class="rec-number">#${index + 1}</span>
                <span class="rec-category category-${rec.category.toLowerCase()}">${rec.category}</span>
            </div>
            <div class="rec-specs">
                <div class="spec-item">
                    <span class="spec-icon">🖥️</span>
                    <span>${rec.vcpus} vCPUs</span>
                </div>
                <div class="spec-item">
                    <span class="spec-icon">💾</span>
                    <span>${rec.memory_gb} GB RAM</span>
                </div>
                <div class="spec-item">
                    <span class="spec-icon">💿</span>
                    <span>${rec.storage_gb} GB</span>
                </div>
                ${rec.gpu_count > 0 ? `
                    <div class="spec-item">
                        <span class="spec-icon">🎮</span>
                        <span>${rec.gpu_count} GPU</span>
                    </div>
                ` : ''}
            </div>
            <div class="rec-footer">
                <div class="rec-cost">
                    <span class="cost-label">Cost:</span>
                    <span class="cost-value">${rec.monthly_cost_formatted}</span>
                </div>
                <div class="rec-metrics">
                    <div class="metric">
                        <span class="metric-label">Value:</span>
                        <span class="metric-value">${rec.value_score.toFixed(2)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Match:</span>
                        <span class="metric-value">${(rec.similarity * 100).toFixed(0)}%</span>
                    </div>
                </div>
            </div>
            ${rec.region ? `<div class="rec-region">📍 ${rec.region}</div>` : ''}
        </div>
    `).join('');

    elements.recommendationsList.style.display = 'block';
}

// Show/hide loading state
function showLoading(isLoading) {
    if (isLoading) {
        elements.btnText.style.display = 'none';
        elements.btnLoader.style.display = 'inline-block';
        elements.predictBtn.disabled = true;
    } else {
        elements.btnText.style.display = 'inline';
        elements.btnLoader.style.display = 'none';
        elements.predictBtn.disabled = false;
    }
}

// Show/hide recommendations loading
function showRecommendationsLoading(isLoading) {
    if (isLoading) {
        elements.recBtnText.style.display = 'none';
        elements.recBtnLoader.style.display = 'inline-block';
        elements.getRecommendations.disabled = true;
    } else {
        elements.recBtnText.style.display = 'inline';
        elements.recBtnLoader.style.display = 'none';
        elements.getRecommendations.disabled = false;
    }
}

// Show error message
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.style.display = 'block';
    setTimeout(() => {
        elements.errorMessage.style.display = 'none';
    }, 8000);
}

// Hide error message
function hideError() {
    elements.errorMessage.style.display = 'none';
}

// Save configuration to Chrome storage
function saveConfig(config) {
    chrome.storage.local.set({ lastConfig: config });
}

// Load saved configuration
function loadSavedConfig() {
    chrome.storage.local.get(['lastConfig'], function(result) {
        if (result.lastConfig) {
            const config = result.lastConfig;
            document.getElementById('vcpus').value = config.vcpus || 2;
            document.getElementById('memory').value = config.memory_gb || 8;
            document.getElementById('storage').value = config.boot_disk_gb || 100;
            document.getElementById('gpuCount').value = config.gpu_count || 0;
            document.getElementById('gpuModel').value = config.gpu_model || 'none';
            document.getElementById('usageHours').value = config.usage_hours_month || 730;
        }
    });
}
