"""Capture source evidence, compare quote conventions, optionally backfill official data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backtesting.dataset import load_feature_frame
from app.config import ROOT, settings
from app.database.schema import BankRate, FxPrice
from app.database.session import make_session
from app.database.upsert import upsert_rows
from app.providers.cbc import parse_daily_archive
from app.providers.hncb import HuaNanBankProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'data/raw/quant_review')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'data/analysis/quant_review_2026-10-16'
    out.mkdir(parents=True, exist_ok=True)
    cbc_path = args.cache_dir / 'cbc.txt'
    hncb_path = args.cache_dir / 'hncb.txt'
    cbc = parse_daily_archive(json.loads(cbc_path.read_text()))
    manifest = []
    for name, url in [('cbc', 'https://cpx.cbc.gov.tw/api/OpenData/FTDOpenData_Day'),
                      ('hncb', 'https://www.hncb.com.tw/hncb/rest/exRate/all'),
                      ('fred', 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXTAUS')]:
        path = args.cache_dir / f'{name}.txt'
        manifest.append(dict(source=name, url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             cached_at_utc=pd.Timestamp(path.stat().st_mtime, unit='s', tz='UTC').isoformat()))
    (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    url = settings()['database']['url']
    if args.apply:
        if not url.startswith('sqlite:///'):
            raise RuntimeError('This one-off backfill requires a local SQLite backup')
        path = Path(url.removeprefix('sqlite:///'))
        if not path.is_absolute():
            path = ROOT / path
        backup = ROOT / 'data/quant_review_before_v3_20261016.db'
        if not backup.exists():
            with sqlite3.connect(path) as source, sqlite3.connect(backup) as target:
                source.backup(target)
    with make_session(url) as session:
        previous = load_feature_frame(session, feature_set='daily_v2')
        comparison = previous[['date', 'USDTWD_CLOSE', 'source']].merge(
            cbc[['date', 'close']], on='date', how='outer', indicator=True)
        comparison['relative_difference'] = comparison.USDTWD_CLOSE / comparison.close - 1
        comparison.to_csv(out / 'v2_vs_cbc.csv', index=False)
        fred = pd.read_csv(args.cache_dir / 'fred.txt')
        fred['date'] = pd.to_datetime(fred.observation_date, utc=True)
        cbc.merge(fred[['date', 'DEXTAUS']], on='date', how='inner').to_csv(out / 'cbc_vs_fed_ny_noon.csv', index=False)
        # Exercise the production parser on the exact captured public response.
        class Response:
            def json(self):
                return json.loads(hncb_path.read_text())
        bank = HuaNanBankProvider()
        bank.get = lambda *args, **kwargs: Response()
        quote = bank.fetch_current_rates()
        quote.to_csv(out / 'verified_hncb_quote.csv', index=False)
        if args.apply:
            records = [dict(observed_at_utc=row.date, pair='USD/TWD', source='cbc_taiwan',
                            open=None, high=None, low=None, close=float(row.close), volume=None)
                       for row in cbc.itertuples()]
            upsert_rows(session, FxPrice, records, ('pair', 'observed_at_utc', 'source'))
            rows = quote.to_dict('records')
            for row in rows:
                row['source'] = 'hua_nan_bank'
            upsert_rows(session, BankRate, rows, ('bank_name', 'currency', 'observed_at_utc', 'source'))
        common = comparison[comparison['_merge'] == 'both']
        print(json.dumps(dict(applied=args.apply, official_rows=len(cbc), official_start=str(cbc.date.min().date()),
                              official_end=str(cbc.date.max().date()), compared_dates=len(common),
                              abs_quote_difference_median=float(common.relative_difference.abs().median()),
                              abs_quote_difference_max=float(common.relative_difference.abs().max()),
                              hncb_quote=quote.to_dict('records')), default=str, indent=2))


if __name__ == '__main__':
    main()
