# Sentinelle: MLOps Spam Detection with DVC, DVCLive, and AWS S3

## Overview
- **Purpose**: End-to-end MLOps pipeline for a Spam Detection model with experiment tracking and data versioning.

- **Key Tools**:
  - DVC → Data & artifact versioning, pipeline orchestration.
  - DVCLive → Experiment tracking, metric logging.
  - AWS S3 → Remote storage for datasets, artifacts, experiment outputs.
- **Repository Status**:
  - Public with initial pipeline configured.
  - Languages: Jupyter Notebook (63.1%), Python (36.9%).

---

## Repository Structure
```
.dvc/          → DVC internals and remote configuration metadata
dvclive/       → Logs, metrics, plots from training/evaluation
experiments/   → Experiment definitions (DVC experiments)
src/           → Source code (data processing, training, evaluation, utils)
.dvcignore     → Ignore rules for DVC
.gitignore     → Ignore rules for Git (excludes data/, models/, reports/)
README.md      → Project documentation
dvc.lock       → Locked pipeline stages and dependencies for reproducibility
dvc.yaml       → Pipeline stages, dependencies, outputs, metrics
params.yaml    → Hyperparameters and configuration
```

---

## Features
- **Data Versioning**
  - Track datasets and artifacts with DVC hashes.
  - Reproduce experiments using `dvc.lock` snapshots.
- **Experiment Tracking**
  - Log metrics (accuracy, precision, recall, etc.) via DVCLive.
  - Compare experiments with DVC’s built-in tooling.
- **Remote Storage (AWS S3)**
  - Push/pull datasets, artifacts, and logs to S3.
  - Enables collaboration and reproducible results across environments.

---

## Prerequisites
- Python 3.10+ (recommended: virtual environment).
- AWS credentials with S3 bucket access.
- Git + DVC installed locally.

## Installation
```bash
# Clone repository
git clone https://github.com/sarveshmhadgut/Sentinelle.git
cd Sentinelle

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
# OR minimal install
pip install dvc[s3] dvclive
```

### Configure DVC Remote (AWS S3)
```bash
# Add remote
dvc remote add -d s3remote s3://<your-bucket>/<path>

# Configure AWS credentials
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export AWS_DEFAULT_REGION=<region>

# Verify
dvc remote list
```

---

## Pipeline

### Core Files
- `dvc.yaml`: Defines stages (ingest, preprocess, train, evaluate).
- `params.yaml`: Hyperparameters (model type, thresholds, splits).
- `dvc.lock`: Snapshots dependencies/outputs.

### Useful Commands
```bash
dvc init              # Initialize DVC
dvc repro             # Reproduce pipeline
dvc repro <stage>     # Run specific stage
dvc dag               # Visualize pipeline DAG
```

### Typical Stages
1. **data_ingest** → Load raw spam dataset.  
2. **preprocess** → Data cleaning, tokenization, vectorization.  
3. **train** → Train spam classifier, log metrics (DVCLive).  
4. **evaluate** → Compute metrics, save reports/plots.  

---

## Data & Artifacts
- **Local Cache** → Managed by DVC in `.dvc/cache`.
- **Remote Storage (S3)**:
  ```bash
  dvc push   # Upload to S3
  dvc pull   # Download from S3
  ```
- **Best Practices**:
  - Ensure `.gitignore` excludes `data/`, `models/`, `reports/`.

---

## Experiment Tracking (DVCLive + DVC)
### Setup
```bash
pip install dvclive
```
Add logging inside training → outputs saved to `dvclive/`.

### Experiment Workflow
```bash
dvc exp run            # Run new experiment
dvc exp show           # Compare experiments
dvc exp apply <id>     # Promote best experiment
dvc exp remove <id>    # Clean up
```

### Visualization
- CLI: `dvc exp show`
- VSCode: [DVC Extension](https://marketplace.visualstudio.com/items?itemName=Iterative.dvc)

---

## AWS Configuration Notes
- **IAM** → Use least-privilege policies (read/write only project prefix).
- **Credentials** → Prefer short-lived (SSO/MFA).
- **Bucket Layout Example**:
  ```
  s3://<bucket>/sentinelle/data/
  s3://<bucket>/sentinelle/artifacts/
  s3://<bucket>/sentinelle/experiments/
  ```

---

## Development Workflow
1. Update code in `src/` and parameters in `params.yaml`.
2. Run pipeline:  
   ```bash
   dvc repro
   ```
3. Track changes:  
   ```bash
   git add .
   git commit -m "Updated pipeline"
   dvc push
   ```
4. Fetch: run `dvc pull` to fetch corresponding data version.

---

## Roadmap
- CI with GitHub Actions (`dvc repro` on PRs).
- Dockerized environment parity.
- Add more metrics/plots (ROC, PR curves).
- Model registry (MLflow or DVC registry).
- API service (FastAPI/Flask) for inference.
- Batch scoring + monitoring.

---

## Troubleshooting
- **S3 Access Denied** → Check IAM, region, credentials. 

- **Slow dvc push/pull** → Use parallel transfers:  
  ```bash
  dvc config core.checksum_jobs <n>
  ```
- **Experiments not showing** → Verify `dvclive/` is written.

---

## Summary
Sentinelle delivers a reproducible, collaborative MLOps pipeline for Spam Detection.  
- Configure DVC with AWS S3 remote.  
- Tune hyperparameters via `params.yaml`.  
- Orchestrate runs with `dvc repro` / `dvc exp run`.  
- Sync datasets and artifacts with `dvc push` / `dvc pull`.  
- Track experiments and metrics seamlessly with DVCLive + DVC.
