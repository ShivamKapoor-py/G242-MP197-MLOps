# G242 – MP197: Metro Station Crowd (Kochi)

## Group
- Group: G242
- Problem: MP197
- Batch: 54
- Faculty: Dr. Upasna Joshi

## Members
- S24CSEU2496 – AARAVCHAUDHARY
- S24CSEU0044 – Krishnasolanki
- S24CSEU0414 – SHIVAMKAPOOR
- S24CSEU1188 – HARDIKGARG

## Stage 1 Setup

Keep the original CSV files unchanged. Do not open and resave `train.csv` in Excel.

### Expected Initial Layout
```text
G242_MP197/
├── data/
│   └── raw/
│       ├── train.csv
│       ├── later.csv
│       └── holdout_features.csv
├── src/
│   └── data_audit.py
├── reports/
│   └── findings.yaml
├── README.md
└── .gitignore
```

### Install and Initialize
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install pandas pyyaml dvc

git init
dvc init
```

### Track Dataset Files
```bash
dvc add data/raw/train.csv data/raw/later.csv

git add .
git commit -m "Initialize project and track raw datasets"
```

### Run the Audit
```bash
python src/data_audit.py --train data/raw/train.csv --later data/raw/later.csv --holdout data/raw/holdout_features.csv
```

The audit script is read-only and does not modify the original CSV files. It checks the dataset structure, missing values, duplicates, and basic data quality.

Review the audit output and update the Stage 1 section of `reports/findings.yaml`.

## Additional Files

### README.md
Add a short description of the project, its purpose, installation steps, and how to run the audit script.

### .gitignore
```gitignore
.venv/
__pycache__/
*.pyc
.DS_Store
```

## Suggested Commit History
```bash
git add README.md .gitignore
git commit -m "Add project documentation and ignore files"

git add src/data_audit.py
git commit -m "Add dataset audit script"

git add data/raw/*.dvc
git commit -m "Track datasets with DVC"

git add reports/findings.yaml
git commit -m "Add Stage 1 dataset audit findings"
```

## Important
- Create ONE public GitHub repository for the group.
- Add all four members as collaborators.
- Every member must commit from their own GitHub account.
- Do not share this group's data or code with other groups.
- Every member must understand and explain the code in the viva.
