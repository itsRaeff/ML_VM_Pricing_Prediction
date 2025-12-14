# GCP VM Pricing Calculator - Chrome Extension

A powerful Chrome extension that provides ML-powered predictions for Google Cloud Platform (GCP) Virtual Machine pricing, recommendations, and cost analysis.

## 🌟 Features

- 💰 **Cost Prediction** - Get accurate monthly cost estimates for VM configurations
- 📊 **Price Classification** - Categorize VMs as Low, Medium, or High cost
- 🎯 **Smart Recommendations** - Find similar VMs with better value
- 😊 **Value Analysis** - Sentiment-based value scoring
- 🔄 **Real-time API** - Connects to your local FastAPI ML backend
- 💾 **Auto-save** - Remembers your last configuration
- 🎨 **Beautiful UI** - Modern, gradient-based design

## 📋 Prerequisites

1. **Chrome Browser** (or any Chromium-based browser)
2. **Python 3.7+** with FastAPI backend
3. **ML Models** - Pre-trained models (from the Jupyter notebook)

## 🚀 Installation

### Step 1: Start the API Server

First, make sure your FastAPI backend is running:

```bash
# Navigate to your project directory
cd c:\Users\raefh\Downloads\MLAPP-VMpricing-main\MLAPP-VMpricing-main

# Start the FastAPI server
python main.py
```

The API should start on `http://localhost:8000`

### Step 2: Add Icons (Required)

Before loading the extension, you need to add icon files:

1. Navigate to `chrome-extension/icons/`
2. Add these three PNG files:
   - `icon16.png` (16x16 pixels)
   - `icon48.png` (48x48 pixels)
   - `icon128.png` (128x128 pixels)

**Quick Icon Solution:**
- Use an emoji-to-PNG converter: https://emoji-to-png.com/
- Use the ☁️ or 💰 emoji
- Download in 16px, 48px, and 128px sizes

### Step 3: Load Extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable **"Developer mode"** (toggle in top-right corner)
3. Click **"Load unpacked"**
4. Select the `chrome-extension` folder:
   ```
   c:\Users\raefh\Downloads\MLAPP-VMpricing-main\MLAPP-VMpricing-main\chrome-extension
   ```
5. The extension should now appear in your extensions list!

### Step 4: Pin the Extension

1. Click the puzzle icon 🧩 in Chrome toolbar
2. Find "GCP VM Pricing Calculator"
3. Click the pin 📌 icon to pin it to your toolbar

## 💡 Usage

### Basic Usage

1. **Click the extension icon** in your Chrome toolbar
2. **Enter VM specifications:**
   - vCPUs (1-96)
   - Memory in GB (0.5-624)
   - Storage in GB (10-65536)
   - GPU count (optional, 0-8)
   - GPU model (K80, T4, P4, P100, V100, A100)
   - Usage hours per month (1-744)

3. **Click "Get Prediction"** to see:
   - Monthly cost estimate
   - Price category (Low/Medium/High)
   - Probability distribution
   - Value analysis (if available)
   - Cluster classification

4. **Load Recommendations** to find similar VMs:
   - Click "Load Recommendations"
   - Browse 5 similar VM configurations
   - Compare costs, specs, and value scores

### Features Explained

#### Cost Prediction
Shows the predicted monthly cost in USD with confidence levels for Low, Medium, and High price categories.

#### Value Analysis
If your API has sentiment analysis enabled, you'll see:
- **Sentiment score** (positive/neutral/negative)
- **Value score** - Performance per dollar ratio
- **Meaning** - Quick interpretation of value

#### Recommendations
Get 5 similar VMs based on:
- **Cluster similarity** - VMs from the same cluster
- **Value score** - Performance per dollar
- **Match percentage** - How similar to your input

### API Status Indicator

The green/red indicator at the top shows API connectivity:
- 🟢 **Green** = API Connected
- 🔴 **Red** = API Offline (check if server is running)

## 🛠️ Configuration

### Change API URL

If your API runs on a different address, edit `popup.js`:

```javascript
// Line 2
const API_BASE_URL = 'http://localhost:8000';  // Change this
```

### Modify Port

If your API uses a different port, update `manifest.json`:

```json
"host_permissions": [
  "http://localhost:YOUR_PORT/*"
]
```

## 🐛 Troubleshooting

### Extension shows "API Offline"
- ✅ Make sure FastAPI server is running: `python main.py`
- ✅ Check server is on port 8000: `http://localhost:8000`
- ✅ Verify CORS is enabled in `main.py`

### "Prediction failed" error
- ✅ Ensure all ML models are loaded (check console output when starting server)
- ✅ Verify model files exist:
  - `regression_model.pkl`
  - `classification_model.pkl`
  - `clustering_model.pkl`
  - `scaler_*.pkl` files
  - `model_metadata.json`

### Icons not showing
- ✅ Add icon files to `chrome-extension/icons/` folder
- ✅ Ensure files are named exactly: `icon16.png`, `icon48.png`, `icon128.png`
- ✅ Reload extension after adding icons

### Extension won't load
- ✅ Check for syntax errors in Chrome Extensions page
- ✅ Verify all files are in the correct location
- ✅ Make sure manifest.json is valid JSON

### Recommendations not loading
- ✅ Ensure `gcp_vm_pricing_raw_dirty_12k.csv` exists in project root
- ✅ Check API logs for errors
- ✅ Try with different VM configurations

## 📁 File Structure

```
chrome-extension/
├── manifest.json          # Extension configuration
├── popup.html            # Main UI interface
├── popup.js              # JavaScript logic & API calls
├── styles.css            # Styling & animations
├── icons/                # Extension icons
│   ├── icon16.png
│   ├── icon48.png
│   └── icon128.png
└── README.md             # This file
```

## 🔧 Development

### Debugging

1. Right-click extension icon → "Inspect popup"
2. Open Chrome DevTools Console
3. View network requests, errors, and logs

### Testing API Endpoints

Test your API directly:

```bash
# Health check
curl http://localhost:8000/health

# Simplified prediction
curl -X POST http://localhost:8000/predict/simplified \
  -H "Content-Type: application/json" \
  -d '{"vcpus": 2, "memory_gb": 8, "boot_disk_gb": 100, "gpu_count": 0, "gpu_model": "none", "usage_hours_month": 730}'
```

### Making Changes

After modifying files:
1. Save your changes
2. Go to `chrome://extensions/`
3. Click the refresh icon ↻ on your extension
4. Test the changes

## 🚀 Future Enhancements

Potential features to add:
- 📊 Cost comparison charts
- 📈 Historical cost tracking
- 💾 Save favorite configurations
- 📤 Export results to CSV/PDF
- 🌍 Multi-region comparison
- ⏰ Budget alerts
- 🔔 Price change notifications

## 📝 API Endpoints Used

- `GET /health` - Check API status
- `POST /predict/simplified` - Get VM predictions
- `POST /recommend` - Get similar VM recommendations

## 🤝 Support

If you encounter issues:

1. Check API server logs for errors
2. Inspect extension popup (right-click → Inspect)
3. Verify all model files are present
4. Ensure Chrome allows localhost connections

## 📄 License

Part of the MLAPP-VMpricing project.

## 🎉 Credits

Built with:
- FastAPI for the ML backend
- Chrome Extension APIs
- Gradient magic ✨
