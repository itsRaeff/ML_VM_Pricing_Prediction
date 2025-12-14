# GCP VM Pricing Chrome Extension

A Chrome extension that predicts Google Cloud Platform (GCP) Virtual Machine pricing and provides intelligent recommendations using machine learning models.

## Features

- 💰 **Price Prediction**: Real-time cost estimation for GCP VMs based on configuration
- 🎯 **Price Category**: Classifies VMs into Low/Medium/High price categories
- 🔮 **Smart Clustering**: Groups similar VM configurations
- 📊 **Recommendations**: Suggests similar VM configurations with better value
- 🎨 **Modern UI**: Beautiful gradient interface with smooth animations

## Architecture

- **Frontend**: Chrome Extension (Manifest V3)
- **Backend**: FastAPI server with REST API
- **ML Models**: 
  - XGBoost Regression (R² = 0.866)
  - Gradient Boosting Classification (Accuracy = 0.797)
  - K-Means Clustering (5 clusters)

## Installation

### 1. Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The server will start at `http://localhost:8000`

### 2. Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top-right corner)
3. Click **Load unpacked**
4. Select the `chrome-extension` folder from this project
5. Pin the extension via the puzzle icon in Chrome toolbar

## Usage

1. Ensure the backend server is running
2. Click the extension icon in Chrome
3. Enter VM configuration:
   - vCPUs (1-96)
   - Memory in GB (1-624)
   - Storage in GB (10-10000)
   - Number of GPUs (0-8)
4. Click **Get Prediction** to see cost estimates
5. Click **Load Recommendations** for similar VM suggestions

## API Endpoints

- `GET /` - Health check
- `POST /predict/simplified` - Get price prediction and classification
- `POST /recommend` - Get VM recommendations

## Model Performance

- **Regression Model**: RMSE = $939.40, R² = 0.866
- **Classification Model**: Accuracy = 79.68%
- **Clustering Model**: 5 clusters with optimal silhouette score

## Dataset

Trained on 5,951 cleaned GCP VM pricing samples from a 12k+ record dataset.

## Tech Stack

- **Backend**: FastAPI, Uvicorn, scikit-learn, XGBoost, Pandas, NumPy
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **ML**: scikit-learn, XGBoost, RobustScaler
- **Data Processing**: Pandas, NumPy, Joblib

## Project Structure

```
GCP-VM-Pricing-Extension/
├── chrome-extension/          # Chrome extension files
│   ├── manifest.json         # Extension configuration
│   ├── popup.html            # Extension UI
│   ├── popup.js              # Frontend logic
│   ├── styles.css            # Styling
│   ├── icons/                # Extension icons
│   └── README.md             # Extension setup guide
├── main.py                   # FastAPI backend server
├── *.pkl                     # Trained ML models
├── model_metadata.json       # Model performance metrics
├── gcp_vm_pricing_raw_dirty_12k.csv  # Training dataset
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## License

MIT License

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
