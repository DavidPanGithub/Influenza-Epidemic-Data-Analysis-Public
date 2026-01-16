# Influenza-Epidemic-Data-Analysis-Public

## 📋 Project Overview

This repository contains a data science project conducted in 2025 investigating the relationship between influenza epidemic data and Google search trends in Hong Kong S.A.R.

## 🎯 Project Intentions

The primary objectives of this project are:

- To analyze correlations between keyword search trends on Google and actual influenza cases during the same periods
- To examine data from 2020-2025 to gain insights into the potential impact of the COVID-19 epidemic on influenza patterns and public search behavior

## 📊 Data Sources

### Google Trends Data
- **Source:** [Google Trends API](https://trends.google.com/trends/)
- **Method:** Data fetched via API
- **Content:** Search volume trends for influenza-related keywords

### Influenza Epidemic Data
- **Source:** [Hong Kong Government Open Data Portal](https://data.gov.hk/en-data/dataset/hk-dh-chpsebcddr-flu-express)
- **Dataset:** Flu Express - Weekly influenza surveillance data from the Centre for Health Protection
- **Key Metric:** Weekly new influenza cases (Column: **A+B**)

## 📁 Repository Structure

```
Influenza-Epidemic-Data-Analysis-Public/
├── data/
│   ├── raw/
│   │   ├── __init__.py
│   │   ├── extract.py              # Data loading functions
│   │   ├── query_trend.csv
│   │   └── flux_data.csv
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── models/                    # Model implementations
│   │   ├── __init__.py
│   │   ├── regression.py          # Regression model training
│   │   └── feature_selection.py   # Feature selection
│   ├── experiments/              # Experiment scripts
│   │   ├── __init__.py
│   │   └── main_experiment.py    # Main experiment runner
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── metrics.py            # Evaluation metrics
│       ├── visualization.py      # Plotting functions
│       └── helpers.py            # Helper functions
├── results/                      # Generated results
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required Python packages (see `requirements.txt`)

### Installation
```bash
git clone https://github.com/DavidPanGithub/Influenza-Epidemic-Data-Analysis-Public.git
cd Influenza-Epidemic-Data-Analysis-Public
pip install -r requirements.txt
