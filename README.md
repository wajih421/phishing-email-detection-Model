# phishing-email-detection-Model

# Phishing Email Detection — Statistical ML Model

A machine learning project that uses Gaussian Naive Bayes 
to detect phishing emails based on statistical feature analysis.

## Results
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 88.75% |
| Precision | 86.16% |
| Recall    | 85.62% |
| F1-Score  | 85.89% |

## Dataset
2,000 labeled emails — 1,200 legitimate, 800 phishing

## Features Used (11 total)
- Sender reputation score
- Contains urgency terms
- Contains money terms
- Number of words
- Has suspicious link
- Number of links
- Has attachment
- Number of attachments
- Number of recipients
- Number of exclamation marks
- Number of characters

## Key Findings
- Urgency language present in 80% of phishing emails vs 28% legitimate
- Sender reputation score strongest continuous predictor (r = -0.51)
- Exclamation marks had near-zero predictive value

## Tech Stack
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn

## How to Run
pip install pandas numpy matplotlib seaborn scikit-learn
python Prob_project.py
