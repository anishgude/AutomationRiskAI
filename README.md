# Human Override Gate

## Problem Statement
Risk-gated automation requires routing potentially unsafe or high-risk customer requests to a human reviewer instead of fully automated handling.

## Task Definition
Input: a customer message. Output: a JSON decision with `override_required`, `reason_codes`, `confidence`, and `recommended_next_step`.

## Baselines
- Rules baseline: keyword-based escalation logic.
- Prompted baseline: base OpenAI model with a strict JSON-only prompt.

## Metrics
- Precision
- Recall
- False negative rate (most important)
- Confusion matrix for `override_required`

## How to Run
```powershell
pip install -r requirements.txt
$env:OPENAI_API_KEY="YOUR_KEY"; $env:OPENAI_FT_MODEL="YOUR_FT_MODEL"; $env:OPENAI_BASE_MODEL="gpt-4o-mini"
python eval/evaluate.py
```
