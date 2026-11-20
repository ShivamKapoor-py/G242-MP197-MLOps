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

## Stage 1 setup
Keep the original CSV files unchanged. Do not open and resave `train.csv` in Excel.

Expected initial layout:
```text
G242_MP197/
├── data/
│   └── raw/
│       ├── train.csv
│       ├── later.csv
│       └── holdout_features.csv
├── src/
│   └── data_audit.py
└── reports/
    └── findings.yaml
```

Install and initialize:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install pandas pyyaml dvc
git init
dvc init
dvc add data/raw/train.csv data/raw/later.csv
git add .
git commit -m "Stage 1: initialize repository and track raw data"
```

Run the audit:
```bash
python src/data_audit.py --train data/raw/train.csv --later data/raw/later.csv --holdout data/raw/holdout_features.csv
```

The audit is read-only: it does not alter the CSV files. Review the output and fill the Stage 1 section of `reports/findings.yaml`.

## Important
- Create ONE public GitHub repository for the group.
- Add all four members as collaborators.
- Every member must commit from their own GitHub account.
- Do not share this group's data or code with other groups.
- Every member must understand and explain the code in the viva.
