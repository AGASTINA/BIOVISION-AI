# 🩺 BioVision AI — Clinical Wound Assessment & Monitoring Platform

An AI-assisted clinical decision support system for chronic wound care, integrating deep segmentation, multi-modal feature extraction, explainable classification, and longitudinal patient monitoring.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.63-red)
![AWS](https://img.shields.io/badge/AWS-EC2-orange)
![License](https://img.shields.io/badge/License-Academic-green)

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     BioVision AI Platform                       │
├─────────────────────────────────────────────────────────────────┤
│  Frontend:  Streamlit (Python)                                  │
│  Database:  SQLite (biovision.db)                               │
│  ML Stack:  YOLO11n-Seg + EfficientNet-B3 + XGBoost + SHAP     │
│  Storage:   AWS S3 (Image & Report storage via boto3)          │
│  Hosting:   AWS EC2 (t3.medium) or Streamlit Cloud             │
└─────────────────────────────────────────────────────────────────┘
```

### AI Pipeline (4-Stage Multi-Modal)

1. **Wound Segmentation** — YOLO11n-Seg generates pixel-level wound boundary masks
2. **Feature Extraction** — 27 handcrafted features (color, texture, shape) + 1,536 deep CNN features = **1,563-dimensional fused vector**
3. **Classification** — XGBoost boosters predict healing probability and 3-tier infection risk
4. **Explainability** — Per-prediction feature attribution + composite BioVision Tissue Index Score (BTIS)

---

## 🚀 Quick Start (Local)

```bash
# Clone
git clone https://github.com/AGASTINA/BIOVISION-AI.git
cd BIOVISION-AI

# Virtual environment
python3.12 -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Seed demo database (first time only)
python seed_data.py

# Run
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## ☁️ AWS EC2 Deployment

### Prerequisites
- AWS account with EC2 access
- Key pair (.pem file) for SSH

### Step 1: Launch EC2 Instance
1. Open **AWS Console → EC2 → Launch Instance**
2. **Name:** `BioVision-AI`
3. **AMI:** Ubuntu Server 24.04 LTS (64-bit x86)
4. **Instance type:** `t3.medium` (4 GB RAM)
5. **Key pair:** Select or create one (.pem download)
6. **Security group:** Allow SSH (port 22, your IP) + Custom TCP (port 8501, 0.0.0.0/0)
7. **Storage:** 20 GB gp3
8. Click **Launch Instance**

### Step 2: Deploy via SSH
```bash
# Connect to EC2
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Clone & deploy (one command)
git clone https://github.com/AGASTINA/BIOVISION-AI.git
cd BIOVISION-AI
chmod +x deploy_aws.sh
sudo ./deploy_aws.sh
```

### Step 3: Access
Open **http://\<EC2_PUBLIC_IP\>:8501** in your browser.

### Management Commands
```bash
# Check status
sudo systemctl status biovision

# View live logs
sudo journalctl -u biovision -f

# Restart app
sudo systemctl restart biovision

# Stop app
sudo systemctl stop biovision
```

### ⚠️ Cost Management
**Stop the EC2 instance when not in use** to avoid charges:
- AWS Console → EC2 → Instances → Select → Instance State → **Stop Instance**
- Cost when running: ~$0.042/hour (~$1/day)
- Cost when stopped: ~$0.00/hour (EBS storage only: ~$0.08/month)

---

## 📁 Project Structure

```
BIOVISION-AI/
├── app.py                    # Main Streamlit application (1,920 lines)
├── database.py               # SQLite ORM & data access layer
├── seed_data.py              # Demo patient data seeder
├── models/
│   ├── best.pt               # YOLO11n-Seg wound segmentation model
│   ├── healing_xgboost.json  # Healing outcome classifier
│   └── infection_xgboost.json # Infection risk classifier
├── test_images/              # Sample wound images for inference
├── data/storage/             # Patient visit images & masks
├── outputs/SurgWound/        # Pre-computed features & SHAP values
├── static/hero.jpg           # Landing page hero image
├── .streamlit/config.toml    # Streamlit theme & server config
├── requirements.txt          # Python dependencies
├── packages.txt              # Linux system dependencies
├── deploy_aws.sh             # EC2 one-command deployment script
├── biovision.service         # systemd service unit
└── README.md                 # This file
```

---

## 🎯 Features

- **Clinician Dashboard** — Evidence-based priority queue ranking patients by wound deterioration
- **9-Tab Case Workspace** — Case overview, AI analysis, clinical assessment, wound timeline, visit comparison, longitudinal graphs, clinician review, reports, system info
- **Patient Portal** — Patient-facing wound journey tracker with photo upload
- **Explainable AI** — Per-prediction feature attribution for clinical transparency
- **Longitudinal Monitoring** — Wound area trajectory tracking across visits

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Web Framework | Streamlit 1.63 |
| Segmentation | YOLO11n-Seg (Ultralytics) |
| Feature Extraction | EfficientNet-B3 (torchvision) |
| Classification | XGBoost (Gradient Boosting) |
| Explainability | SHAP (TreeExplainer) |
| Image Processing | OpenCV, PIL, NumPy |
| Database | SQLite3 |
| Cloud Hosting | AWS EC2 (t3.medium) |
| Process Manager | systemd |

---

## 👨‍💻 Authors

**AGASTINA** — BioVision AI Clinical Platform

---

*AI-assisted clinical decision support. Clinician confirmation remains required for all assessments.*
