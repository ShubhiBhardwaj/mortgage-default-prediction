# Mortgage Default Prediction

Logistic regression model to predict mortgage delinquency using Fannie Mae single-family loan data from 2021 Q1.

## What This Project Does

I built a binary classification model to predict whether a loan will go 90+ days delinquent. The dataset is about 22GB so I used chunked reading in pandas to process it without crashing memory, then trained a logistic regression model on 1.4 million loans.

## Dataset

Fannie Mae Single-Family Loan Performance Data, 2021 Q1  
Source: https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data

## Features Used

| Variable | Description |
|---|---|
| orig_interest_rate | Loan interest rate at origination |
| orig_ltv | Loan-to-value ratio |
| dti | Debt-to-income ratio |
| credit_score | Borrower credit score at origination |

**Target variable:** `defaulted` (1 = 90+ days delinquent at any point, 0 = otherwise)

## Results

| Metric | Value |
|---|---|
| AUC-ROC | 0.7988 |
| Precision-Recall AUC | 0.0670 |
| Recall (defaults) | 74% |
| Precision (defaults) | 4% |

The default rate in the dataset is 1.55% so this is a heavily imbalanced classification problem. I used `class_weight="balanced"` to handle this and evaluated using AUC-ROC and PR-AUC rather than accuracy.

## Key Findings

Credit score is the strongest predictor and reduces default risk. LTV and DTI both increase default risk, which lines up with what you would expect. Higher interest rate is also positively associated with default, partly because riskier borrowers tend to get higher rates.

## How to Run

1. Download the Fannie Mae 2021 Q1 data and update the file path in the notebook
2. Run all cells in order (Cell 3 takes a few minutes as it reads 22GB in chunks)

```
pip install pandas scikit-learn matplotlib
```

## Skills Demonstrated

- Handling BigData (22GB) with chunked pandas processing
- Feature engineering and binary target creation
- Logistic regression with class imbalance handling
- Model evaluation: confusion matrix, AUC-ROC, precision-recall
- Threshold analysis to understand the precision-recall tradeoff
- Data visualisation with matplotlib
