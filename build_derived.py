"""Prebuild the small tables the app needs from the big Transfermarkt appearances table.

Run locally after refreshing the dataset:  python build_derived.py
The server then loads these few MB instead of ~200MB of match rows (fits a 512MB free plan).
"""
from scout import data_tm as tm, metrics, valuation

for f in ("league_seasons.parquet", "feature_frame.parquet", "reference_date.txt"):
    (tm.TM_DIR / f).unlink(missing_ok=True)

ref = tm.reference_date()
metrics.league_seasons().to_parquet(tm.TM_DIR / "league_seasons.parquet", index=False)
valuation.feature_frame().to_parquet(tm.TM_DIR / "feature_frame.parquet")
(tm.TM_DIR / "reference_date.txt").write_text(str(ref))
print("done", ref)
