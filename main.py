"""
FastAPI Server for GCP VM Pricing ML Models
Serves 3 models: Regression, Classification, and Clustering
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import json
from pydantic import BaseModel
from typing import List, Dict, Any
import traceback

app = FastAPI(
    title="GCP VM Pricing API",
    description="ML API for GCP VM cost prediction, classification, and clustering",
    version="1.0.0"
)

# CORS middleware - allows requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scalers at startup
print("Loading models...")
try:
    regression_model = joblib.load("regression_model.pkl")
    classification_model = joblib.load("classification_model.pkl")
    clustering_model = joblib.load("clustering_model.pkl")
    
    scaler_reg = joblib.load("scaler_regression.pkl")
    scaler_clf = joblib.load("scaler_classification.pkl")
    scaler_cluster = joblib.load("scaler_clustering.pkl")
    
    label_encoders = joblib.load("label_encoders.pkl")
    
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load sentiment analysis model (optional)
    try:
        sentiment_model = joblib.load("sentiment_model.pkl")
        sentiment_vectorizer = joblib.load("sentiment_vectorizer.pkl")
        print("✅ Sentiment analysis model loaded")
    except:
        sentiment_model = None
        sentiment_vectorizer = None
        print("⚠️ Sentiment analysis model not found (optional)")
    
    # Load original dataset for recommendations
    try:
        df_original = pd.read_csv("gcp_vm_pricing_raw_dirty_12k.csv")
        
        # Clean and prepare the dataset
        # Convert memory from string (e.g., "16 GB") to numeric
        if 'memory' in df_original.columns and 'memory_gb' not in df_original.columns:
            df_original['memory_gb'] = df_original['memory'].astype(str).str.extract(r'(\d+\.?\d*)')[0]
            df_original['memory_gb'] = pd.to_numeric(df_original['memory_gb'], errors='coerce')
        
        # Convert vcpus to numeric if it's not already
        if 'vcpus' in df_original.columns:
            df_original['vcpus'] = pd.to_numeric(df_original['vcpus'], errors='coerce')
        
        # Convert boot_disk_gb to numeric
        if 'boot_disk_gb' in df_original.columns:
            df_original['boot_disk_gb'] = pd.to_numeric(df_original['boot_disk_gb'], errors='coerce')
        
        # Fill NaN boot_disk_gb with default
        df_original['boot_disk_gb'] = df_original['boot_disk_gb'].fillna(100)
        
        # Fill NaN gpu_count with 0
        if 'gpu_count' in df_original.columns:
            df_original['gpu_count'] = df_original['gpu_count'].fillna(0)
        
        print(f"✅ Loaded {len(df_original)} VMs from original dataset for recommendations")
    except Exception as e:
        df_original = None
        print(f"⚠️ Error loading dataset: {e}")
    
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# Request models
class RegressionRequest(BaseModel):
    features: List[float]

class ClassificationRequest(BaseModel):
    features: List[float]

class ClusteringRequest(BaseModel):
    features: List[float]

class AllModelsRequest(BaseModel):
    features_regression: List[float]
    features_classification: List[float]
    features_clustering: List[float]

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "GCP VM Pricing ML API",
        "version": "1.0.0",
        "endpoints": {
            "/predict/regression": "Predict monthly cost (USD)",
            "/predict/classification": "Classify VM price category",
            "/predict/clustering": "Find VM cluster group",
            "/predict/all": "Get all predictions",
            "/models/info": "Get models information"
        }
    }

# Model info endpoint
@app.get("/models/info")
def get_model_info():
    return {
        "models": {
            "regression": {
                "name": metadata["regression_model"],
                "r2_score": metadata["regression_r2"],
                "rmse": metadata["regression_rmse"],
                "mae": metadata["regression_mae"],
                "num_features": len(metadata["regression_features"])
            },
            "classification": {
                "name": metadata["classification_model"],
                "accuracy": metadata["classification_accuracy"],
                "categories": metadata["price_categories"],
                "num_features": len(metadata["classification_features"])
            },
            "clustering": {
                "name": metadata["clustering_model"],
                "silhouette_score": metadata["clustering_silhouette"],
                "num_clusters": metadata["num_clusters"],
                "num_features": len(metadata["clustering_features"])
            }
        },
        "feature_names": {
            "regression": metadata["regression_features"],
            "classification": metadata["classification_features"],
            "clustering": metadata["clustering_features"]
        }
    }

# Regression endpoint - Predict monthly cost
@app.post("/predict/regression")
def predict_regression(data: RegressionRequest):
    try:
        # Convert to numpy array and reshape
        features = np.array(data.features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler_reg.transform(features)
        
        # Make prediction
        prediction = regression_model.predict(features_scaled)
        
        return {
            "prediction": float(prediction[0]),
            "prediction_formatted": f"${prediction[0]:.2f}",
            "model": metadata["regression_model"],
            "unit": "USD/month"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Classification endpoint - Classify price category
@app.post("/predict/classification")
def predict_classification(data: ClassificationRequest):
    try:
        # Convert to numpy array and reshape
        features = np.array(data.features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler_clf.transform(features)
        
        # Make prediction
        prediction = classification_model.predict(features_scaled)
        prediction_proba = classification_model.predict_proba(features_scaled)
        
        # Get category name
        category_map = {0: "Low", 1: "Medium", 2: "High"}
        category = category_map.get(int(prediction[0]), "Unknown")
        
        return {
            "prediction": int(prediction[0]),
            "category": category,
            "probabilities": {
                "Low": float(prediction_proba[0][0]),
                "Medium": float(prediction_proba[0][1]),
                "High": float(prediction_proba[0][2])
            },
            "model": metadata["classification_model"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

# Clustering endpoint - Find cluster group
@app.post("/predict/clustering")
def predict_clustering(data: ClusteringRequest):
    try:
        # Convert to numpy array and reshape
        features = np.array(data.features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler_cluster.transform(features)
        
        # Make prediction
        cluster = clustering_model.predict(features_scaled)
        
        return {
            "cluster": int(cluster[0]),
            "total_clusters": metadata["num_clusters"],
            "model": metadata["clustering_model"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")

# Combined endpoint - Get all predictions
@app.post("/predict/all")
def predict_all(data: AllModelsRequest):
    try:
        # Regression prediction
        features_reg = np.array(data.features_regression).reshape(1, -1)
        features_reg_scaled = scaler_reg.transform(features_reg)
        cost_prediction = regression_model.predict(features_reg_scaled)
        
        # Classification prediction
        features_clf = np.array(data.features_classification).reshape(1, -1)
        features_clf_scaled = scaler_clf.transform(features_clf)
        category_prediction = classification_model.predict(features_clf_scaled)
        category_proba = classification_model.predict_proba(features_clf_scaled)
        
        # Clustering prediction
        features_cluster = np.array(data.features_clustering).reshape(1, -1)
        features_cluster_scaled = scaler_cluster.transform(features_cluster)
        cluster_prediction = clustering_model.predict(features_cluster_scaled)
        
        # Get category name
        category_map = {0: "Low", 1: "Medium", 2: "High"}
        category = category_map.get(int(category_prediction[0]), "Unknown")
        
        return {
            "regression": {
                "monthly_cost": float(cost_prediction[0]),
                "monthly_cost_formatted": f"${cost_prediction[0]:.2f}"
            },
            "classification": {
                "category": category,
                "category_id": int(category_prediction[0]),
                "probabilities": {
                    "Low": float(category_proba[0][0]),
                    "Medium": float(category_proba[0][1]),
                    "High": float(category_proba[0][2])
                }
            },
            "clustering": {
                "cluster": int(cluster_prediction[0]),
                "total_clusters": metadata["num_clusters"]
            },
            "models_used": {
                "regression": metadata["regression_model"],
                "classification": metadata["classification_model"],
                "clustering": metadata["clustering_model"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}\n{traceback.format_exc()}")

# Simplified prediction endpoint for mobile app
class SimplifiedInput(BaseModel):
    vcpus: float
    memory_gb: float
    boot_disk_gb: float
    gpu_count: float = 0.0
    gpu_model: str = "none"
    usage_hours_month: float = 730.0

@app.post("/predict/simplified")
def predict_simplified(data: SimplifiedInput):
    """
    Simplified prediction endpoint that takes minimal inputs
    and fills in reasonable defaults for other features
    """
    try:
        # Determine GPU features
        has_gpu = 1.0 if data.gpu_count > 0 else 0.0
        
        # Intelligently select machine family based on specs
        # n2: general purpose, balanced CPU/memory ratio (1:4)
        # c2/c3: compute optimized, high CPU ratio (1:2 or less)
        # m2/m3: memory optimized, high memory ratio (1:8 or more)
        memory_cpu_ratio = data.memory_gb / max(data.vcpus, 1)
        
        if memory_cpu_ratio >= 8:
            machine_family_str = "m2"  # Memory optimized
        elif memory_cpu_ratio <= 2:
            machine_family_str = "c2"  # Compute optimized  
        else:
            machine_family_str = "n2"  # General purpose (most common)
        
        # Encode categorical features using label encoders
        try:
            machine_family_encoded = float(label_encoders['machine_family'].transform([machine_family_str])[0])
        except:
            machine_family_encoded = 0.0
            
        # Machine type based on specs (simplified)
        machine_type_str = f"{machine_family_str}-standard-{int(data.vcpus)}"
        try:
            machine_type_encoded = float(label_encoders['machine_type'].transform([machine_type_str])[0])
        except:
            # If exact match not found, use a default
            machine_type_encoded = 0.0
        
        # CPU architecture - default to x86_64 (most common)
        try:
            cpu_arch_encoded = float(label_encoders['cpu_arch'].transform(['x86_64'])[0])
        except:
            cpu_arch_encoded = 0.0
        
        # Region - default to us-central1 (most common and middle pricing)
        try:
            region_encoded = float(label_encoders['region'].transform(['us-central1'])[0])
            region_code_encoded = float(label_encoders['region_code'].transform(['US-CENTRAL1'])[0])
        except:
            region_encoded = 0.0
            region_code_encoded = 0.0
        
        # Zone - default to us-central1-a
        try:
            zone_encoded = float(label_encoders['zone'].transform(['us-central1-a'])[0])
        except:
            zone_encoded = 0.0
        
        # OS - default to Linux (cheapest)
        try:
            os_encoded = float(label_encoders['os'].transform(['Linux'])[0])
        except:
            os_encoded = 0.0
        
        # Network tier - premium (default)
        try:
            network_tier_encoded = float(label_encoders['network_tier'].transform(['premium'])[0])
        except:
            network_tier_encoded = 0.0
        
        # Price model - on-demand/standard
        try:
            price_model_encoded = float(label_encoders['price_model'].transform(['on-demand'])[0])
        except:
            price_model_encoded = 0.0
        
        # Currency - USD
        try:
            currency_encoded = float(label_encoders['currency'].transform(['USD'])[0])
        except:
            currency_encoded = 0.0
        
        # Boot disk type - pd-standard
        try:
            boot_disk_type_encoded = float(label_encoders['boot_disk_type'].transform(['pd-standard'])[0])
        except:
            boot_disk_type_encoded = 0.0
        
        # GPU model encoding
        gpu_model_lower = data.gpu_model.lower()
        if gpu_model_lower not in ['none', '', 'no'] and data.gpu_count > 0:
            try:
                gpu_model_encoded = float(label_encoders['gpu_model'].transform([gpu_model_lower])[0])
            except:
                # Try common variations
                if 't4' in gpu_model_lower:
                    try:
                        gpu_model_encoded = float(label_encoders['gpu_model'].transform(['nvidia-tesla-t4'])[0])
                    except:
                        gpu_model_encoded = 1.0
                elif 'v100' in gpu_model_lower:
                    try:
                        gpu_model_encoded = float(label_encoders['gpu_model'].transform(['nvidia-tesla-v100'])[0])
                    except:
                        gpu_model_encoded = 2.0
                else:
                    gpu_model_encoded = 1.0
        else:
            try:
                gpu_model_encoded = float(label_encoders['gpu_model'].transform(['none'])[0])
            except:
                gpu_model_encoded = 0.0
        
        # Egress destination - default to internet
        try:
            egress_dest_encoded = float(label_encoders['egress_destination'].transform(['internet'])[0])
        except:
            egress_dest_encoded = 0.0
        
        # Billing frequency - monthly
        try:
            billing_freq_encoded = float(label_encoders['billing_frequency'].transform(['monthly'])[0])
        except:
            billing_freq_encoded = 0.0
        
        # Feedback - neutral/default
        try:
            feedback_encoded = float(label_encoders['feedback'].transform(['neutral'])[0])
        except:
            feedback_encoded = 0.0
        
        # Sustained use discount eligible - yes
        try:
            sud_eligible_encoded = float(label_encoders['sustained_use_discount_eligible'].transform(['yes'])[0])
        except:
            sud_eligible_encoded = 1.0
        
        # Preemptible available - yes
        try:
            preempt_avail_encoded = float(label_encoders['preemptible_available'].transform(['yes'])[0])
        except:
            preempt_avail_encoded = 1.0
        
        # Estimate GPU hourly cost (rough estimate based on GCP pricing)
        gpu_hourly_cost = 0.0
        if has_gpu:
            gpu_model_lower = data.gpu_model.lower()
            if 'k80' in gpu_model_lower:
                gpu_hourly_cost = 0.45 * data.gpu_count
            elif 't4' in gpu_model_lower:
                gpu_hourly_cost = 0.35 * data.gpu_count
            elif 'p4' in gpu_model_lower:
                gpu_hourly_cost = 0.60 * data.gpu_count
            elif 'p100' in gpu_model_lower:
                gpu_hourly_cost = 1.46 * data.gpu_count
            elif 'v100' in gpu_model_lower:
                gpu_hourly_cost = 2.48 * data.gpu_count
            elif 'a100' in gpu_model_lower:
                gpu_hourly_cost = 3.67 * data.gpu_count
            else:
                gpu_hourly_cost = 0.5 * data.gpu_count  # default estimate
        
        # Note: Feature order MUST match trained scaler (33 features, excluding price_category target)
        # Using properly encoded categorical features for accurate predictions
        features = [
            machine_family_encoded,  # machine_family (intelligently selected)
            machine_type_encoded,  # machine_type (based on family + vcpus)
            float(data.vcpus),  # vcpus (from user input)
            cpu_arch_encoded,  # cpu_arch (x86_64 default)
            region_encoded,  # region (us-central1 default)
            region_code_encoded,  # region_code (US-CENTRAL1 default)
            zone_encoded,  # zone (us-central1-a default)
            os_encoded,  # os (Linux default)
            network_tier_encoded,  # network_tier (premium)
            price_model_encoded,  # price_model (on-demand)
            sud_eligible_encoded,  # sustained_use_discount_eligible
            0.85,  # cud_1yr_discount (15% discount)
            0.75,  # cud_3yr_discount (25% discount)
            preempt_avail_encoded,  # preemptible_available
            currency_encoded,  # currency (USD)
            boot_disk_type_encoded,  # boot_disk_type (pd-standard)
            float(data.boot_disk_gb),  # boot_disk_gb (from user input)
            0.0,  # local_ssd_count (default: 0)
            0.0,  # local_ssd_total_gb (default: 0)
            gpu_model_encoded,  # gpu_model (encoded from user input)
            float(data.gpu_count),  # gpu_count (from user input)
            gpu_hourly_cost,  # gpu_hourly_usd (estimated based on model)
            egress_dest_encoded,  # egress_destination (internet)
            0.0,  # egress_gb (default: 0)
            0.0,  # egress_unit_price_usd (default: 0)
            billing_freq_encoded,  # billing_frequency (monthly)
            float(data.usage_hours_month),  # usage_hours_month (from user input)
            feedback_encoded,  # feedback (neutral)
            float(data.memory_gb),  # memory_gb (from user input)
            0.0,  # sustained_use_discount (calculated, default: 0)
            0.0,  # preemptible (0 = no)
            has_gpu,  # has_gpu (calculated from gpu_count)
            0.0,  # has_local_ssd (0 = no)
        ]
        
        features_array = np.array(features).reshape(1, -1)
        
        print(f"✅ Created feature array with {features_array.shape[1]} features (properly encoded)")
        print(f"   Machine family: {machine_family_str}, Memory/CPU ratio: {memory_cpu_ratio:.1f}")
        
        # Make predictions
        # Regression
        features_scaled_reg = scaler_reg.transform(features_array)
        cost_prediction = regression_model.predict(features_scaled_reg)
        
        # Classification
        features_scaled_clf = scaler_clf.transform(features_array)
        category_prediction = classification_model.predict(features_scaled_clf)
        category_proba = classification_model.predict_proba(features_scaled_clf)
        
        # Clustering
        features_scaled_cluster = scaler_cluster.transform(features_array)
        cluster_prediction = clustering_model.predict(features_scaled_cluster)
        
        # Get category name
        category_map = {0: "Low", 1: "Medium", 2: "High"}
        category = category_map.get(int(category_prediction[0]), "Unknown")
        
        # Sentiment Analysis (if model is available)
        sentiment_result = None
        if sentiment_model is not None and sentiment_vectorizer is not None:
            try:
                # Generate VM description for sentiment analysis
                gpu_text = f"{data.gpu_count} GPU" if data.gpu_count > 0 else "no GPU"
                vm_description = (
                    f"VM with {data.vcpus} vCPUs {data.memory_gb} GB memory "
                    f"{data.boot_disk_gb} GB storage {gpu_text} "
                    f"category {category} estimated cost ${cost_prediction[0]:.2f}"
                )
                
                # Calculate value score (performance per dollar)
                performance_score = data.vcpus * data.memory_gb + (data.gpu_count * 100)
                value_score = performance_score / (cost_prediction[0] + 1)  # Add 1 to avoid division by zero
                
                # Vectorize description
                description_vectorized = sentiment_vectorizer.transform([vm_description])
                
                # Predict sentiment
                sentiment_pred = sentiment_model.predict(description_vectorized)[0]
                sentiment_proba = sentiment_model.predict_proba(description_vectorized)[0]
                
                # Get probabilities for each sentiment
                sentiment_classes = sentiment_model.classes_
                sentiment_probs = {str(cls): float(prob) for cls, prob in zip(sentiment_classes, sentiment_proba)}
                
                # Sentiment interpretation
                sentiment_meaning = {
                    "positive": "Excellent value for money 💰✨",
                    "neutral": "Fair value for the price 👍",
                    "negative": "Consider alternatives for better value 🤔"
                }
                
                sentiment_result = {
                    "sentiment": str(sentiment_pred),
                    "meaning": sentiment_meaning.get(str(sentiment_pred), "Unknown"),
                    "probabilities": sentiment_probs,
                    "value_score": float(value_score),
                    "confidence": float(max(sentiment_proba))
                }
            except Exception as e:
                print(f"Sentiment analysis error: {e}")
                sentiment_result = None
        
        response = {
            "regression": {
                "monthly_cost": float(cost_prediction[0]),
                "monthly_cost_formatted": f"${cost_prediction[0]:.2f}"
            },
            "classification": {
                "category": category,
                "category_id": int(category_prediction[0]),
                "probabilities": {
                    "Low": float(category_proba[0][0]),
                    "Medium": float(category_proba[0][1]),
                    "High": float(category_proba[0][2])
                }
            },
            "clustering": {
                "cluster": int(cluster_prediction[0]),
                "total_clusters": metadata["num_clusters"]
            },
            "input_summary": {
                "vcpus": data.vcpus,
                "memory_gb": data.memory_gb,
                "storage_gb": data.boot_disk_gb,
                "gpu_count": data.gpu_count,
                "gpu_model": data.gpu_model,
                "usage_hours_month": data.usage_hours_month,
                "has_gpu": bool(has_gpu)
            },
            "models_used": {
                "regression": metadata["regression_model"],
                "classification": metadata["classification_model"],
                "clustering": metadata["clustering_model"]
            }
        }
        
        # Add sentiment if available
        if sentiment_result is not None:
            response["sentiment"] = sentiment_result
        
        return response
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"\n❌ ERROR in /predict/simplified:")
        print(f"   Error: {str(e)}")
        print(f"   Input: vcpus={data.vcpus}, memory={data.memory_gb}, storage={data.boot_disk_gb}")
        print(f"   GPU: count={data.gpu_count}, model={data.gpu_model}")
        print(f"📋 Full Traceback:\n{error_details}\n")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}\n{error_details}")

# Recommendation endpoint
@app.post("/recommend")
def recommend_similar_vms(data: SimplifiedInput):
    """
    Recommend similar VMs from the original dataset based on clustering similarity
    Uses the same logic as the notebook's recommendation system
    """
    try:
        if df_original is None:
            return {"recommendations": [], "message": "Recommendations unavailable - dataset not loaded"}
        
        # Get query VM features and prediction
        has_gpu = 1.0 if data.gpu_count > 0 else 0.0
        
        # Use same intelligent encoding as simplified prediction
        memory_cpu_ratio = data.memory_gb / max(data.vcpus, 1)
        if memory_cpu_ratio >= 8:
            machine_family_str = "m2"
        elif memory_cpu_ratio <= 2:
            machine_family_str = "c2"
        else:
            machine_family_str = "n2"
        
        try:
            machine_family_encoded = float(label_encoders['machine_family'].transform([machine_family_str])[0])
        except:
            machine_family_encoded = 0.0
        
        machine_type_str = f"{machine_family_str}-standard-{int(data.vcpus)}"
        try:
            machine_type_encoded = float(label_encoders['machine_type'].transform([machine_type_str])[0])
        except:
            machine_type_encoded = 0.0
        
        try:
            cpu_arch_encoded = float(label_encoders['cpu_arch'].transform(['x86_64'])[0])
        except:
            cpu_arch_encoded = 0.0
        
        try:
            region_encoded = float(label_encoders['region'].transform(['us-central1'])[0])
            region_code_encoded = float(label_encoders['region_code'].transform(['US-CENTRAL1'])[0])
        except:
            region_encoded = 0.0
            region_code_encoded = 0.0
        
        try:
            zone_encoded = float(label_encoders['zone'].transform(['us-central1-a'])[0])
        except:
            zone_encoded = 0.0
        
        try:
            os_encoded = float(label_encoders['os'].transform(['Linux'])[0])
        except:
            os_encoded = 0.0
        
        try:
            network_tier_encoded = float(label_encoders['network_tier'].transform(['premium'])[0])
        except:
            network_tier_encoded = 0.0
        
        try:
            price_model_encoded = float(label_encoders['price_model'].transform(['on-demand'])[0])
        except:
            price_model_encoded = 0.0
        
        try:
            currency_encoded = float(label_encoders['currency'].transform(['USD'])[0])
        except:
            currency_encoded = 0.0
        
        try:
            boot_disk_type_encoded = float(label_encoders['boot_disk_type'].transform(['pd-standard'])[0])
        except:
            boot_disk_type_encoded = 0.0
        
        gpu_model_lower = data.gpu_model.lower()
        if gpu_model_lower not in ['none', '', 'no'] and data.gpu_count > 0:
            try:
                gpu_model_encoded = float(label_encoders['gpu_model'].transform([gpu_model_lower])[0])
            except:
                if 't4' in gpu_model_lower:
                    try:
                        gpu_model_encoded = float(label_encoders['gpu_model'].transform(['nvidia-tesla-t4'])[0])
                    except:
                        gpu_model_encoded = 1.0
                elif 'v100' in gpu_model_lower:
                    try:
                        gpu_model_encoded = float(label_encoders['gpu_model'].transform(['nvidia-tesla-v100'])[0])
                    except:
                        gpu_model_encoded = 2.0
                else:
                    gpu_model_encoded = 1.0
        else:
            try:
                gpu_model_encoded = float(label_encoders['gpu_model'].transform(['none'])[0])
            except:
                gpu_model_encoded = 0.0
        
        try:
            egress_dest_encoded = float(label_encoders['egress_destination'].transform(['internet'])[0])
        except:
            egress_dest_encoded = 0.0
        
        try:
            billing_freq_encoded = float(label_encoders['billing_frequency'].transform(['monthly'])[0])
        except:
            billing_freq_encoded = 0.0
        
        try:
            feedback_encoded = float(label_encoders['feedback'].transform(['neutral'])[0])
        except:
            feedback_encoded = 0.0
        
        try:
            sud_eligible_encoded = float(label_encoders['sustained_use_discount_eligible'].transform(['yes'])[0])
        except:
            sud_eligible_encoded = 1.0
        
        try:
            preempt_avail_encoded = float(label_encoders['preemptible_available'].transform(['yes'])[0])
        except:
            preempt_avail_encoded = 1.0
        
        gpu_hourly_cost = 0.0
        if has_gpu:
            if 'k80' in gpu_model_lower:
                gpu_hourly_cost = 0.45 * data.gpu_count
            elif 't4' in gpu_model_lower:
                gpu_hourly_cost = 0.35 * data.gpu_count
            elif 'p4' in gpu_model_lower:
                gpu_hourly_cost = 0.60 * data.gpu_count
            elif 'p100' in gpu_model_lower:
                gpu_hourly_cost = 1.46 * data.gpu_count
            elif 'v100' in gpu_model_lower:
                gpu_hourly_cost = 2.48 * data.gpu_count
            elif 'a100' in gpu_model_lower:
                gpu_hourly_cost = 3.67 * data.gpu_count
            else:
                gpu_hourly_cost = 0.5 * data.gpu_count
        
        # Query features (33 features, properly encoded)
        query_features = [
            machine_family_encoded,
            machine_type_encoded,
            float(data.vcpus),
            cpu_arch_encoded,
            region_encoded,
            region_code_encoded,
            zone_encoded,
            os_encoded,
            network_tier_encoded,
            price_model_encoded,
            sud_eligible_encoded,
            0.85,  # cud_1yr_discount
            0.75,  # cud_3yr_discount
            preempt_avail_encoded,
            currency_encoded,
            boot_disk_type_encoded,
            float(data.boot_disk_gb),
            0.0,  # local_ssd_count
            0.0,  # local_ssd_total_gb
            gpu_model_encoded,
            float(data.gpu_count),
            gpu_hourly_cost,
            egress_dest_encoded,
            0.0,  # egress_gb
            0.0,  # egress_unit_price_usd
            billing_freq_encoded,
            float(data.usage_hours_month),
            feedback_encoded,
            float(data.memory_gb),
            0.0,  # sustained_use_discount
            0.0,  # preemptible
            has_gpu,
            0.0,  # has_local_ssd
        ]
        query_array = np.array(query_features).reshape(1, -1)
        query_scaled = scaler_cluster.transform(query_array)
        query_cluster = int(clustering_model.predict(query_scaled)[0])
        
        # Sample VMs from original dataset
        # Filter out rows with missing critical data first
        df_valid = df_original.dropna(subset=['vcpus', 'memory_gb']).copy()
        df_valid = df_valid[(df_valid['vcpus'] > 0) & (df_valid['memory_gb'] > 0)]
        
        # Take a reasonable sample for fast processing
        sample_size = min(500, len(df_valid))  # Reduced for speed
        df_sample = df_valid.sample(n=sample_size, random_state=None).copy()
        
        print(f"Searching {sample_size} VMs for recommendations...")
        
        # Extract basic features from dataset
        recommendations = []
        category_map = {0: "Low", 1: "Medium", 2: "High"}
        
        for idx, row in df_sample.iterrows():
            try:
                # Extract features
                vcpus = row.get('vcpus', 2)
                memory_gb = row.get('memory_gb', 8)
                boot_disk_gb = row.get('boot_disk_gb', 100)
                gpu_count = row.get('gpu_count', 0) if pd.notna(row.get('gpu_count', 0)) else 0
                
                # Skip invalid entries
                if pd.isna(vcpus) or pd.isna(memory_gb):
                    continue
                if vcpus <= 0 or memory_gb <= 0:
                    continue
                    
                # Build feature vector for this VM (33 features)
                vm_has_gpu = 1.0 if gpu_count > 0 else 0.0
                vm_features = [
                    0.0,  # machine_family
                    0.0,  # machine_type
                    float(vcpus),  # vcpus
                    0.0,  # cpu_arch
                    0.0,  # region
                    0.0,  # region_code
                    0.0,  # zone
                    0.0,  # os
                    0.0,  # network_tier
                    0.0,  # price_model
                    1.0,  # sustained_use_discount_eligible
                    0.85,  # cud_1yr_discount
                    0.75,  # cud_3yr_discount
                    1.0,  # preemptible_available
                    0.0,  # currency
                    0.0,  # boot_disk_type
                    float(boot_disk_gb),  # boot_disk_gb
                    0.0,  # local_ssd_count
                    0.0,  # local_ssd_total_gb
                    0.0,  # gpu_model
                    float(gpu_count),  # gpu_count
                    0.0,  # gpu_hourly_usd
                    0.0,  # egress_destination
                    0.0,  # egress_gb
                    0.0,  # egress_unit_price_usd
                    0.0,  # billing_frequency
                    730.0,  # usage_hours_month
                    0.0,  # feedback
                    float(memory_gb),  # memory_gb
                    0.0,  # sustained_use_discount
                    0.0,  # preemptible
                    vm_has_gpu,  # has_gpu
                    0.0,  # has_local_ssd
                ]
                vm_array = np.array(vm_features).reshape(1, -1)
                vm_scaled = scaler_cluster.transform(vm_array)
                vm_cluster = int(clustering_model.predict(vm_scaled)[0])
                
                # Accept VMs from all clusters for diversity
                # Add small penalty for different clusters
                cluster_penalty = 0.0 if vm_cluster == query_cluster else 0.2
                
                # Predict cost and category
                vm_reg_scaled = scaler_reg.transform(vm_array)
                predicted_cost = float(regression_model.predict(vm_reg_scaled)[0])
                
                vm_clf_scaled = scaler_clf.transform(vm_array)
                predicted_category = int(classification_model.predict(vm_clf_scaled)[0])
                
                # Calculate similarity (Euclidean distance in simple feature space)
                query_simple = np.array([data.vcpus, data.memory_gb, data.boot_disk_gb / 100, data.gpu_count])
                vm_simple = np.array([vcpus, memory_gb, boot_disk_gb / 100, gpu_count])
                distance = np.sqrt(np.sum((query_simple - vm_simple) ** 2)) + cluster_penalty
                similarity = 1 / (1 + distance)
                
                # Calculate value score
                value_score = (vcpus + memory_gb / 4) / (predicted_cost + 1)
                
                # Include region and machine type for diversity
                region = row.get('region', 'Unknown')
                machine_type = row.get('machine_type', 'Standard')
                
                recommendations.append({
                    "vcpus": int(vcpus),
                    "memory_gb": float(memory_gb),
                    "storage_gb": float(boot_disk_gb),
                    "gpu_count": int(gpu_count),
                    "monthly_cost": predicted_cost,
                    "monthly_cost_formatted": f"${predicted_cost:.2f}",
                    "category": category_map.get(predicted_category, "Medium"),
                    "value_score": value_score,
                    "similarity": similarity,
                    "distance": distance,
                    "region": str(region)[:20],  # Truncate long region names
                    "machine_type": str(machine_type)
                })
            except Exception as e:
                continue
        
        if len(recommendations) == 0:
            print(f"No recommendations found in cluster {query_cluster}")
            return {
                "recommendations": [],
                "message": f"No similar VMs found in cluster {query_cluster}"
            }
        
        print(f"Found {len(recommendations)} similar VMs in cluster {query_cluster}")
        
        # Sort by distance (lowest = most similar), then by value score
        recommendations.sort(key=lambda x: (x['distance'], -x['value_score']))
        
        # Select diverse recommendations - different configurations
        unique_recommendations = []
        seen_configs = set()
        
        # Strategy: Pick the best from each configuration tier
        # 1. Exact match (same vCPUs, memory)
        # 2. Lower tier (fewer resources, cheaper)
        # 3. Higher tier (more resources)
        # 4. Different balance (more CPU vs more RAM)
        # 5. With storage variance
        
        for rec in recommendations:
            # Create a looser config key to allow price variations
            config_key = (rec['vcpus'], rec['memory_gb'], rec['gpu_count'])
            
            if config_key not in seen_configs or len(unique_recommendations) < 5:
                # Only add if truly unique or we need more variety
                is_unique = config_key not in seen_configs
                
                if is_unique:
                    seen_configs.add(config_key)
                    unique_recommendations.append(rec)
                
                if len(unique_recommendations) >= 10:  # Get more candidates first
                    break
        
        # Now select top 5 with maximum diversity based on value and similarity
        # Sort by a combination of similarity and value
        unique_recommendations.sort(key=lambda x: (-x['similarity'], -x['value_score']))
        
        top_5 = []
        seen_exact_specs = set()
        
        for rec in unique_recommendations:
            # Create spec signature including region/machine type for diversity
            exact_spec = (rec['vcpus'], rec['memory_gb'], rec['storage_gb'], 
                         rec['gpu_count'], rec['region'], rec['machine_type'])
            
            # Only add if this exact config hasn't been added
            if exact_spec not in seen_exact_specs:
                top_5.append(rec)
                seen_exact_specs.add(exact_spec)
            
            if len(top_5) >= 5:
                break
        
        # If we still don't have 5, add the best remaining ones
        if len(top_5) < 5:
            for rec in unique_recommendations:
                if len(top_5) >= 5:
                    break
                if rec not in top_5:
                    top_5.append(rec)
        
        # Remove distance field and fix similarity display (internal use only)
        for rec in top_5:
            rec.pop('distance', None)
            # Similarity is already 0-1, just keep it as is for display
        
        print(f"Returning {len(top_5)} unique recommendations")
        
        return {
            "recommendations": top_5,
            "query_summary": {
                "cluster": query_cluster,
                "total_similar_vms": len(recommendations),
                "unique_configs": len(unique_recommendations)
            }
        }
        
    except Exception as e:
        # Don't fail the whole request if recommendations fail
        print(f"Recommendation error: {str(e)}")
        return {"recommendations": [], "message": "Recommendations temporarily unavailable"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "version": "1.0.0"
    }

# Feedback sentiment analysis endpoint
class FeedbackRequest(BaseModel):
    feedback: str
    vm_specs: dict = None  # Optional VM specs context

@app.post("/analyze/feedback")
def analyze_feedback(data: FeedbackRequest):
    """
    Analyze user feedback and return sentiment
    """
    try:
        if sentiment_model is None or sentiment_vectorizer is None:
            return {
                "error": "Sentiment analysis model not available",
                "sentiment": "neutral",
                "confidence": 0.0
            }
        
        # Vectorize the feedback text
        feedback_vectorized = sentiment_vectorizer.transform([data.feedback])
        
        # Predict sentiment
        sentiment_pred = sentiment_model.predict(feedback_vectorized)[0]
        sentiment_proba = sentiment_model.predict_proba(feedback_vectorized)[0]
        
        # Get probabilities for each sentiment
        sentiment_classes = sentiment_model.classes_
        sentiment_probs = {str(cls): float(prob) for cls, prob in zip(sentiment_classes, sentiment_proba)}
        
        # Sentiment interpretation
        sentiment_meaning = {
            "positive": "Great feedback! This indicates positive experience 😊",
            "neutral": "Neutral feedback. Average experience 👍",
            "negative": "Negative feedback. Room for improvement 🤔"
        }
        
        return {
            "sentiment": str(sentiment_pred),
            "meaning": sentiment_meaning.get(str(sentiment_pred), "Unknown"),
            "probabilities": sentiment_probs,
            "confidence": float(max(sentiment_proba)),
            "feedback": data.feedback
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

