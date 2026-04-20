# Complete Data Pipeline and Exploratory Data Analysis Structure

This module sets up an automated Data Engineering pipeline over the Titanic dataset.

## Folder Structure

- `data/` : Holds the raw `.csv` file.
- `src/` : The brain of the project broken into `collection`, `cleaning`, `eda`, `visualization`, and `feature engineering`. All glued together by `main.py`.
- `outputs/` : Generated dynamically. Stores the 8 outputted `.png` plots and the consolidated `report.html`.

## Getting Started

1. Install requirements
```bash
pip install -r requirements.txt
```

2. Run the Main Pipeline
```bash
cd src
python main.py
```

This will automatically trigger data ingest, cleaning rules, produce charts, engineer features, run an example Random Forest classifier, and spit everything out into `outputs/report.html`.
