# Bank Customer Churn Prediction & Cost Optimisation

> Most churn models optimise for accuracy. This one optimises for money.

## Overview

A full ML pipeline comparing four classification models evaluated on **actual business cost** — not just accuracy. Every wrong prediction has a price tag, and the model that minimises total retention cost wins.

**Dataset:** Kaggle Bank Customer Churn · 10,000 customers · 14 features

## Key Results

| Model | Best Threshold | Minimum Cost | Saving vs Baseline |
|---|---|---|---|
| Logistic Regression | 0.36 | $132,200 | — |
| Random Forest | 0.07 | $113,700 | −$18,500 |
| Gradient Boosting | 0.09 | $104,600 | −$27,600 |
| **XGBoost** | **0.23** | **$103,500** | **−$28,700** |

- **AUC:** XGBoost 0.867 vs Logistic Regression 0.777
- **Germany** has a 32.4% churn rate — nearly double France and Spain
- **3-product customers** churn at 82.7% — a critical non-linear signal

## Cost Framework

| Error Type | Cost |
|---|---|
| False Negative (missed churner) | $1,000 |
| False Positive (unnecessary offer) | $100 |

## Tech Stack

- **Python** — Pandas, NumPy, Scikit-learn, XGBoost, Plotly
- Threshold optimisation loop across 50 decision thresholds
- SMOTE for class imbalance handling

## Project Structure
- Customer_Churn_Analysis.ipynb   # Main notebook
- Churn_Modelling.csv             # Dataset
- README.md

## Key Findings

1. Deploy XGBoost at threshold 0.23 — minimises total cost at $103,500
2. Re-engage inactive customers — churn at 26.9% vs 14.3% for active
3. Investigate Germany urgently — 32.4% churn vs ~16% elsewhere
4. 3-product customers signal a product design issue, not a retention problem

## Live Portfolio

View the full interactive case study: [mandiv-analytics.netlify.app/project-churn.html](https://mandiv-analytics.netlify.app/project-churn.html)
