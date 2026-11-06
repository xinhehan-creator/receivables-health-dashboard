"""Turns raw SEC XBRL extracts into a tidy fact table, applies the semantic layer, and runs data-quality checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "sec_receivables_raw.json"

SECTOR = {
    "AAPL": "Technology hardware", "HPQ": "Technology hardware", "CSCO": "Technology hardware",
    "INTC": "Semiconductors", "NVDA": "Semiconductors",
    "MSFT": "Software", "ORCL": "Software", "CRM": "Software", "ADBE": "Software", "IBM": "Software",
    "CAT": "Industrials", "MMM": "Industrials", "DE": "Industrials", "TSLA": "Industrials",
    "PG": "Consumer staples", "KO": "Consumer staples",
}
MEASURES = {
    "ar": "accounts_receivable_net",
    "allow": "allowance_for_credit_losses",
    "rev": "revenue",
    "prov": "credit_loss_provision",
    "wo": "write_offs",
}


def load_long() -> pd.DataFrame:
    raw = json.loads(RAW.read_text())
    rows = []
    for ticker, rec in raw.items():
        for key, measure in MEASURES.items():
            for end, val in rec.get(key, {}).items():
                rows.append({"ticker": ticker, "company": rec["name"], "measure": measure,
                             "period_end": pd.Timestamp(end), "value": float(val)})
    df = pd.DataFrame(rows)
    df["sector"] = df["ticker"].map(SECTOR)
    return df


def merge_restated(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """DQ2: period-ends within 10 days of each other are the same fiscal year reported twice."""
    log, keep = [], []
    for (ticker, measure), g in df.sort_values("period_end").groupby(["ticker", "measure"]):
        g = g.reset_index(drop=True)
        i = 0
        while i < len(g):
            j = i
            while j + 1 < len(g) and (g.loc[j + 1, "period_end"] - g.loc[i, "period_end"]).days <= 10:
                j += 1
            chosen = g.loc[j]
            if j > i:
                log.append({"rule": "DQ2", "severity": "info", "ticker": ticker, "measure": measure,
                            "detail": f"{j - i + 1} values for period ending near {chosen.period_end.date()}; kept the latest"})
            keep.append(chosen)
            i = j + 1
    out = pd.DataFrame(keep)
    # A fiscal year ending in the first week of January belongs to the prior year.
    fy = out["period_end"].dt.year
    out["fiscal_year"] = np.where(out["period_end"].dt.dayofyear <= 7, fy - 1, fy)
    return out, log


def build_facts() -> tuple[pd.DataFrame, pd.DataFrame]:
    long, log = merge_restated(load_long())
    wide = (long.pivot_table(index=["ticker", "company", "sector", "fiscal_year"],
                             columns="measure", values="value", aggfunc="last")
            .reset_index())
    for m in MEASURES.values():
        if m not in wide:
            wide[m] = np.nan
    wide = wide.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)

    # Semantic layer metrics (definitions in semantic_model.yaml).
    wide["gross_receivables"] = wide["accounts_receivable_net"] + wide["allowance_for_credit_losses"]
    wide["dso_days"] = wide["accounts_receivable_net"] / wide["revenue"] * 365
    wide["allowance_coverage_pct"] = wide["allowance_for_credit_losses"] / wide["gross_receivables"] * 100
    g = wide.groupby("ticker")
    wide["ar_growth_pct"] = g["accounts_receivable_net"].pct_change(fill_method=None) * 100
    wide["revenue_growth_pct"] = g["revenue"].pct_change(fill_method=None) * 100
    wide["ar_minus_revenue_growth_pp"] = wide["ar_growth_pct"] - wide["revenue_growth_pct"]
    wide["write_off_rate_pct"] = wide["write_offs"] / g["gross_receivables"].shift(1) * 100
    wide["dso_change_days"] = g["dso_days"].diff()

    log += quality_checks(wide)
    return wide, pd.DataFrame(log)


def quality_checks(w: pd.DataFrame) -> list[dict]:
    log = []
    for t, g in w.groupby("ticker"):
        for m, label in [("accounts_receivable_net", "net receivables"), ("revenue", "revenue"),
                         ("allowance_for_credit_losses", "allowance")]:
            n_missing = g[m].isna().sum()
            if n_missing:
                log.append({"rule": "DQ1", "severity": "warning", "ticker": t, "measure": m,
                            "detail": f"{label} missing in {n_missing} of {len(g)} fiscal years"})
    for _, r in w[w["allowance_coverage_pct"] > 10].iterrows():
        log.append({"rule": "DQ3", "severity": "review", "ticker": r.ticker, "measure": "allowance_coverage_pct",
                    "detail": f"FY{r.fiscal_year}: coverage {r.allowance_coverage_pct:.1f}% is unusually high; check the XBRL tag"})
    for _, r in w[w["credit_loss_provision"] < 0].iterrows():
        log.append({"rule": "DQ4", "severity": "info", "ticker": r.ticker, "measure": "credit_loss_provision",
                    "detail": f"FY{r.fiscal_year}: negative provision of ${r.credit_loss_provision/1e6:,.0f}M (reversal)"})
    for _, r in w[w["revenue"] < w["accounts_receivable_net"]].iterrows():
        log.append({"rule": "DQ5", "severity": "error", "ticker": r.ticker, "measure": "revenue",
                    "detail": f"FY{r.fiscal_year}: revenue below net receivables"})
    return log


def anomalies(w: pd.DataFrame, z: float = 2.0) -> pd.DataFrame:
    """Flags years where a company's DSO moved unusually far from its own history (robust z-score)."""
    out = []
    for t, g in w.dropna(subset=["dso_change_days"]).groupby("ticker"):
        x = g["dso_change_days"]
        med = x.median()
        mad = (x - med).abs().median() * 1.4826
        if mad == 0 or len(x) < 4:
            continue
        score = (x - med) / mad
        for idx in score[score.abs() >= z].index:
            r = w.loc[idx]
            out.append({"ticker": t, "fiscal_year": int(r.fiscal_year), "dso_days": r.dso_days,
                        "dso_change_days": r.dso_change_days, "robust_z": score[idx],
                        "ar_minus_revenue_growth_pp": r.ar_minus_revenue_growth_pp})
    return pd.DataFrame(out).sort_values("robust_z", key=abs, ascending=False) if out else pd.DataFrame()


def ar_variance_bridge(w: pd.DataFrame, ticker: str, year: int) -> dict | None:
    """Splits the YoY change in receivables into a revenue (volume) effect and a collection (DSO) effect."""
    g = w[w.ticker == ticker].set_index("fiscal_year")
    if year not in g.index or year - 1 not in g.index:
        return None
    cur, pre = g.loc[year], g.loc[year - 1]
    if pd.isna([cur.revenue, pre.revenue, cur.dso_days, pre.dso_days]).any():
        return None
    volume = (cur.revenue - pre.revenue) * pre.dso_days / 365
    collection = cur.revenue * (cur.dso_days - pre.dso_days) / 365
    return {"start": pre.accounts_receivable_net, "volume_effect": volume,
            "collection_effect": collection, "end": cur.accounts_receivable_net}


def insight_summary(w: pd.DataFrame, ticker: str) -> list[str]:
    """Rule-based narrative over the semantic metrics. Each sentence cites the numbers it uses."""
    g = w[w.ticker == ticker].dropna(subset=["dso_days"]).sort_values("fiscal_year")
    if len(g) < 2:
        return ["Not enough years with both receivables and revenue to describe a trend."]
    first, last = g.iloc[0], g.iloc[-1]
    lines = [f"DSO moved from {first.dso_days:.0f} days in FY{first.fiscal_year} "
             f"to {last.dso_days:.0f} days in FY{last.fiscal_year}."]
    recent = g.tail(3)
    gap = recent["ar_minus_revenue_growth_pp"].dropna()
    if len(gap) and (gap > 5).all():
        lines.append(f"Receivables have outgrown revenue for {len(gap)} straight years "
                     f"(by {gap.mean():.1f} pp on average), which is worth a closer look at collections.")
    elif len(gap) and (gap < -5).all():
        lines.append(f"Revenue has outgrown receivables for {len(gap)} straight years, "
                     "consistent with faster collection.")
    cov = g.dropna(subset=["allowance_coverage_pct"])
    if len(cov) >= 2:
        d = cov.iloc[-1].allowance_coverage_pct - cov.iloc[0].allowance_coverage_pct
        direction = "rose" if d > 0 else "fell"
        lines.append(f"Allowance coverage {direction} from {cov.iloc[0].allowance_coverage_pct:.1f}% to "
                     f"{cov.iloc[-1].allowance_coverage_pct:.1f}% of gross receivables.")
    else:
        lines.append("The company does not tag its allowance separately in most years, so coverage cannot be tracked.")
    return lines


if __name__ == "__main__":
    facts, dq = build_facts()
    print(facts.shape, "\n", dq.groupby(["rule", "severity"]).size())
    print(anomalies(facts).head(8).to_string())
    print(ar_variance_bridge(facts, "MSFT", 2025))
    print(insight_summary(facts, "CSCO"))
