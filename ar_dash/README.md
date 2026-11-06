# Receivables Health Dashboard

An AI-ready analytics dashboard on accounts receivable quality for 16 US-listed companies, built on public SEC EDGAR XBRL data from 10-K filings (FY2016 onward).

## What it does

- **Semantic layer** (`semantic_model.yaml`): one governed definition for each base measure (with XBRL tag mapping) and each metric, such as DSO, allowance coverage, and receivables growth minus revenue growth.
- **Data pipeline** (`pipeline.py`): turns raw XBRL facts into a tidy company-by-year fact table, merges restated period-ends, and assigns fiscal years.
- **Data quality checks**: five rules (missing measures, duplicates, implausible ratios, provision reversals, revenue below receivables) logged by severity.
- **Variance analysis**: a waterfall bridge that splits the year-on-year change in receivables into a revenue (volume) effect and a collection (DSO) effect. The two effects reconcile exactly to the total change.
- **Anomaly detection**: flags years where a company's change in DSO sits at least 2 robust standard deviations (median absolute deviation) from its own history.
- **Insight summaries**: rule-based narrative lines generated from the semantic metrics, so each statement traces back to a number.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

`data/sec_receivables_raw.json` holds annual 10-K values pulled from `https://data.sec.gov/api/xbrl/companyfacts/`. Some companies do not tag their allowance separately; the data quality tab lists these gaps rather than filling them.
