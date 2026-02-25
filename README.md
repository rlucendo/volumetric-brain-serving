# MLOps serving pipeline for 3D Brain Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-darkgreen)
![MLOps](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF)

## 1. Project Summary & Vision

This repository (NeuroSeg-3D API) serves as the **Production, Serving, and MLOps counterpart** to the research-focused [volumetric-brain-segmentation](https://github.com/rlucendo/volumetric-brain-segmentation) project. 

While the previous repository addressed the data science and deep learning challenges—training a 3D U-Net on multi-modal MRI sequences (Task01_BrainTumour) and performing clinical XAI evaluation—this project solves the **Software Engineering and Infrastructure** challenges. 

Industry models often fail to cross the chasm from a Jupyter Notebook to a production environment. Bridging that gap for volumetric medical imaging requires handling massive I/O operations, strict memory constraints (preventing VRAM/RAM Out-Of-Memory crashes), and maintaining spatial metadata integrity. 

**Core Objectives of this System:**
* **Decoupling:** Transform a static PyTorch `.ckpt` artifact into a dynamic, containerized RESTful API.
* **Resilience:** Ensure high availability under concurrent requests using asynchronous non-blocking I/O.
* **Immutability:** Implement a CI/CD pipeline that injects model weights at build-time, producing a secure, reproducible Docker image ready for Cloud deployment (AWS, GCP, or DigitalOcean).

---

## 2. System Architecture

To guarantee maintainability and scalability, this project strictly adheres to **Clean Architecture** principles. The application is divided into isolated layers, preventing the web framework (FastAPI) from tightly coupling with the mathematical domain (PyTorch/MONAI).

### 2.1. Directory Structure

```text
volumetric-brain-serving/
├── .github/workflows/
│   └── ci_cd_pipeline.yml       # GitHub Actions workflow for Linting, Testing, and GHCR Build
├── scripts/
│   └── download_model.py        # Build-time script to securely fetch the artifact from W&B
├── src/
│   ├── api/                     # 1. PRESENTATION LAYER
│   │   ├── __init__.py
│   │   ├── routes.py            # FastAPI endpoint definitions (/health, /predict)
│   │   └── schemas.py           # Pydantic models for strict data validation and Swagger UI
│   ├── core/                    # 2. CONFIGURATION LAYER
│   │   ├── __init__.py
│   │   ├── config.py            # Environment variable management (Pydantic BaseSettings)
│   │   └── logger.py            # Structured JSON logging setup (structlog)
│   ├── services/                # 3. DOMAIN & BUSINESS LOGIC LAYER
│   │   ├── __init__.py
│   │   ├── inference_engine.py  # PyTorch singleton: Model loading and Sliding Window Inference
│   │   └── medical_transforms.py# MONAI deterministic ETL pipeline (Spacing, Orientation, Norm)
│   └── main.py                  # ASGI Application Factory and Lifespan Manager
├── tests/
│   ├── __init__.py
│   ├── test_api.py              # Integration tests using FastAPI TestClient
│   └── test_transforms.py       # Unit tests for the medical tensor pipeline
├── Dockerfile                   # Multi-stage build definition targeting CPU execution
├── Makefile                     # Developer Experience (DX) commands (lint, test, run)
├── pyproject.toml               # Single source of truth for dependencies (PEP 518)
├── .dockerignore                # Excludes heavy environments and secrets from the image
├── .env.example                 # Template for local development secrets
└── .gitignore
```

### 2.2. The Request Lifecycle (Data Flow)

When a client (e.g., a hospital's PACS system or a clinical frontend) interacts with the system, the data flows through the following pipeline:

1.  **Ingestion (Asynchronous):** The client sends a multipart `POST` request to `/api/v1/predict` containing a multi-channel `.nii.gz` file (typically 50-150 MB).
2.  **I/O Offloading:** FastAPI writes the file to a secure, temporary disk location to avoid saturating the server's RAM.
3.  **Pre-processing (Domain Layer):** The `MedicalDataProcessor` ingests the file using MONAI. It ensures the tensor is channeled correctly, reoriented to the neurological RAS standard, resampled to isotropic spacing, and Z-score normalized.
4.  **Forward Pass (Engine Layer):** The pre-processed tensor is passed to the `InferenceEngine`. Using *Sliding Window Inference*, the 3D U-Net predicts the tumor mask without exceeding memory limits.
5.  **Reconstruction:** The resulting mask tensor is converted back to a Numpy array. Crucially, the original spatial `Affine` matrix is restored via NiBabel, ensuring the mask perfectly overlaps the patient's original MRI in clinical viewers.
6.  **Response & Cleanup:** The server returns the segmented `.nii.gz` file via a streaming response and triggers a Background Task to safely delete all temporary files from the server's disk.

---

## 3. MLOps & Production Practices

Deploying deep learning models safely requires bridging the gap between Data Science and DevOps. This repository implements several industry-standard MLOps patterns:

### 3.1. Build-Time Artifact Injection (Model Registry)
A common anti-pattern is downloading the model weights dynamically when the API boots. This introduces runtime latency, points of failure (if the registry is down), and mutable states. 
Instead, we utilize a **Build-Time Injection** strategy:
* The `.ckpt` file is tracked externally in **Weights & Biases (W&B) Artifacts**.
* During the Docker build process, a secure script (`download_model.py`) authenticates via a CI/CD Secret, downloads the specific version of the weights, and bakes them into the image. 
* **Result:** The deployed Docker container is completely self-contained, immutable, and air-gap ready.

### 3.2. Stateful Lifespan Management
Instantiating a PyTorch model and moving it to hardware memory takes time. To prevent the API from blocking or crashing on concurrent requests, we leverage FastAPI's asynchronous `lifespan`. 
The `InferenceEngine` acts as a Singleton; the 3D U-Net is loaded into memory *exactly once* during server startup. All subsequent HTTP requests share this pre-loaded engine, reducing inference latency to pure compute time.

### 3.3. State Dict Deserialization Strategy
Due to the model being originally trained using PyTorch Lightning, the `.ckpt` file encapsulates deep nested structures (e.g., `model.net.model...`) and implicit architectural layers (like Instance Normalization injected by MONAI). The `inference_engine.py` implements a robust deserializer that systematically peels away Lightning's "matryoshka" nesting, mapping the raw tensor weights precisely to the native MONAI architecture, preventing weight initialization failures (amnesia).

### 3.4. Cost-Optimized Containerization (Multi-Stage CPU Build)
While GPUs are mandatory for training volumetric models, they are often cost-prohibitive for serving MVP portfolios (e.g., an AWS `g4dn.xlarge` instance runs at ~$400/month).
* **Target Architecture:** The `Dockerfile` explicitly installs the `CPU-only` wheels for PyTorch. This allows the API to be hosted on standard, affordable VMs (like a $10/month DigitalOcean Droplet).
* **Multi-Stage Build:** A `builder` stage compiles dependencies, while the `runner` stage contains only the bare essentials.
* **Security:** The container runs under a strictly restricted `non-root` user profile, complying with standard cloud security policies.

### 3.5. Observability (Structured JSON Logging)
Standard Python `print()` statements are insufficient for cloud monitoring. This API uses `structlog` to output logs in strict JSON format. If deployed to a Kubernetes cluster or AWS ECS, tools like Datadog or ELK can instantly parse these logs to track request IDs, monitor inference latency, and trace exceptions.

---

## 4. API Reference

The API provides auto-generated OpenAPI (Swagger) documentation. When the server is running locally, navigate to `http://localhost:8000/docs` to interact with the endpoints.

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/v1/health` | `GET` | **System Probe:** Returns a JSON response validating if the API is up, the PyTorch engine is loaded in memory, and identifying the hardware device (CPU/CUDA). Ideal for Kubernetes liveness/readiness probes. |
| `/api/v1/predict` | `POST` | **Inference Engine:** Accepts a multipart 3D `.nii.gz` file (FLAIR, T1w, T1gd, T2w sequences), performs Sliding Window Inference, and returns the segmentation mask (Background, Necrosis, Edema, Active Tumor) as a downloadable `.nii.gz` file, preserving the original Affine matrix. |

---

## 5. Getting Started & Deployment

This project supports two environments: a rapid local development setup (using virtual environments) and a production-grade Docker deployment.

### Prerequisites
* Python 3.10+
* Git
* A [Weights & Biases (W&B)](https://wandb.ai/) account and an active API Key (for downloading the model weights).
* Docker (Optional, for Scenario B).

### Scenario A: Local Development (Fast Iteration)

This setup is ideal for testing code changes, debugging the MONAI pipeline, and running the `pytest` suite without rebuilding the Docker image.

**1. Clone the repository and configure the environment**
```bash
git clone https://github.com/rlucendo/volumetric-brain-serving.git
cd volumetric-brain-serving

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

**2. Install dependencies**
Install the project in editable mode `[-e]` along with development tools (Ruff, Pytest).
```bash
pip install -e .[dev]
```

**3. Fetch the Model Artifact**
The repository does not store the `.ckpt` file. You must authenticate with W&B to download it securely into the `models/` directory.
```bash
# Export your W&B API key to your terminal session
export WANDB_API_KEY="your_personal_api_key_here"

# (Optional) If your artifact is named differently, override the default path
# export WANDB_ARTIFACT_PATH="your_workspace/project/model:version"

# Execute the secure downloader script
python scripts/download_model.py
```

**4. Boot the Server**
Start the FastAPI server using Uvicorn with live-reloading enabled.
```bash
uvicorn src.main:app --reload
```
*The API is now live at `http://127.0.0.1:8000`. Navigate to `/docs` to upload a sample `.nii.gz` file.*

---

### Scenario B: Production Docker Deployment (Immutable Build)

This simulates the CI/CD pipeline, building a secure, CPU-optimized image ready for deployment on any cloud provider (e.g., AWS EC2, DigitalOcean Droplet, GCP Compute Engine).

**1. Ensure the model is available locally**
The multi-stage `Dockerfile` expects the `last.ckpt` file to be present in the `models/` directory before building. If you haven't run Step 3 from Scenario A, do it now.

**2. Build the Docker Image**
This process leverages a multi-stage build. It will install the CPU-only version of PyTorch to drastically reduce image size and cloud hosting costs.
```bash
docker build -t neuroseg-api:latest .
```

**3. Run the Container**
Boot the container in detached mode (`-d`), mapping the internal port 8000 to your host machine. The container runs under a secure, non-root user profile.
```bash
docker run -d -p 8000:8000 --name neuroseg_serving neuroseg-api:latest
```

**4. Check container health**
```bash
# View the structured JSON startup logs
docker logs -f neuroseg_serving
```

*To stop the container, run `docker stop neuroseg_serving`.*

---

## 6. Future Roadmap

Given more compute budget and time, the following enhancements would scale this MVP to enterprise-level readiness:

* [ ] **Automated CI/CD Workflows:** Fully implement `.github/workflows/ci_cd_pipeline.yml` to trigger Ruff linting, Pytest execution, and automatic Docker pushes to the GitHub Container Registry (GHCR) on every merge to `main`.
* [ ] **Test-Time Augmentation (TTA) in Serving:** Implement multi-axis flipping during the inference pass to average out predictions and smooth boundary artifacts, trading latency for higher HD95 accuracy.
* [ ] **Infrastructure as Code (IaC):** Develop Terraform (`.tf`) or Pulumi scripts to automatically provision the cloud infrastructure (VPC, Security Groups, EC2 instance) and pull the latest Docker image securely.

---

## Author

**Rubén Lucendo**  
*AI Engineer & Product Builder*  
[LinkedIn Profile](https://www.linkedin.com/in/rubenlucendo/)

Building systems that bridge the gap between theory and business value.

---

## Legal, Clinical, and Regulatory Disclaimer

**STRICTLY FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. NOT FOR CLINICAL USE.**

The software, code, models, weights, algorithms, and any associated documentation provided in this repository (collectively referred to as the "Software") are experimental in nature and are intended strictly for academic, educational, and non-commercial research purposes. 

By accessing, downloading, or utilizing this Software, you explicitly acknowledge and agree to the following terms:

1. **No Medical Advice or Clinical Decision Support:** This Software does not constitute, nor is it intended to be a substitute for, professional medical advice, diagnosis, treatment, or clinical decision-making. The simulated macroscopic projections, spatiotemporal predictions, and any other outputs generated by the PINN (Physics-Informed Neural Network) or associated models are purely theoretical and have not been validated for clinical efficacy or accuracy. You must not rely on any output from this Software to make clinical or medical decisions regarding any patient.
2. **No Regulatory Clearance:** This Software has NOT been cleared, approved, or evaluated by the U.S. Food and Drug Administration (FDA), the European Medicines Agency (EMA), or any other global regulatory authority as a medical device or Software as a Medical Device (SaMD).
3. **Data Privacy and Compliance:** The user assumes full and sole responsibility for ensuring that any data (including but not limited to medical imaging, NIfTI files, or patient metadata) processed using this Software complies with all applicable local, state, national, and international data protection and privacy laws, including but not limited to the Health Insurance Portability and Accountability Act (HIPAA) in the United States and the General Data Protection Regulation (GDPR) in the European Union. The author(s) explicitly disclaim any responsibility for the unlawful or non-compliant use of Protected Health Information (PHI) or Personally Identifiable Information (PII) in conjunction with this Software.
4. **Limitation of Liability and Indemnification:** IN NO EVENT SHALL THE AUTHOR(S), CONTRIBUTORS, OR AFFILIATED ENTITIES BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS OR MEDICAL MALPRACTICE CLAIMS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION. By using this Software, you agree to indemnify, defend, and hold harmless the author(s) from any and all claims, liabilities, damages, and expenses (including legal fees) arising from your use or misuse of the Software.
5. **Disclaimer of Warranty:** THE SOFTWARE IS PROVIDED ON AN "AS IS" AND "AS AVAILABLE" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. THE AUTHOR(S) MAKE NO REPRESENTATIONS THAT THE MACROSCOPIC DIGITAL TWIN SIMULATIONS OR PREDICTIONS WILL BE ACCURATE, ERROR-FREE, OR BIOLOGICALLY PLAUSIBLE.

Use of this repository constitutes your unconditional acceptance of these terms.