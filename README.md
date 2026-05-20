![CI Pipeline](https://github.com/islem14623/my_project_FE/actions/workflows/ci.yml/badge.svg)

# IIoT Intrusion Detection System with Nature-Inspired Feature Selection

MLOps project deploying a CNN-based intrusion detection model with PSO feature selection as a REST API.

## 🎯 Project Overview

This project implements an **Intrusion Detection System (IDS)** for Industrial IoT networks using:
- **Deep Learning**: CNN model for classification
- **Feature Selection**: PSO (Particle Swarm Optimization) for optimal feature extraction
- **Deployment**: Flask REST API with Docker support

**Dataset**: Edge-IIoTset (Industrial IoT intrusion detection dataset)

---

## 🚀 Features

- ✅ REST API for real-time intrusion detection
- ✅ CNN model with PSO-optimized features
- ✅ Docker containerization (deployment-ready)
- ✅ CI/CD pipeline with GitHub Actions (coming soon)

---

## 📋 Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/DL** | TensorFlow, Keras, scikit-learn |
| **API** | Flask |
| **Optimization** | PSO, GA, ACO algorithms |
| **DevOps** | Docker, Git |
| **Language** | Python 3.10 |

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/islem14623/my_project_FE.git
cd my_project_FE

# Install dependencies
pip install -r requirements.txt

# Run the API
cd src
python api.py
```

The API will be available at `http://localhost:5000`

---

## 📡 API Usage

### Check API status
```bash
curl http://localhost:5000/
```

### Make a prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ..., 0.9]}'  # 39 features
```

**Response:**
```json
{
  "prediction": 1,
  "result": "ATTACK",
  "confidence": 0.95
}
```

---

## 🎓 Academic Context

**Master's Thesis (PFE)** - Master 2 in Computer Networks  
**University**: Université Ferhat Abbas Sétif-1, Algeria  
**Student**: Islem Chenafi  
**Topic**: Feature Selection for IIoT Intrusion Detection Using Nature-Inspired Optimization

---

## 📁 Project Structure
