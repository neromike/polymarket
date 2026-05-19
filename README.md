# Polymarket Local Data App

This project is organized around three SQL-backed workflows:

- `update`: API refreshes that write into `data\polymarket.sqlite3`
- `analyze`: offline analysis from the local database
- `dashboard`: a dynamic browser interface for data review and job control

The main entrypoint is:

```powershell
python app.py status
```

Check database table counts:

```powershell
python app.py db status
```

## Common Commands

Refresh market metadata and price history:

```powershell
python app.py update markets
```

Refresh a specific market by conditionId:

```powershell
python app.py update markets --market 0x... --force-market-refresh
```

Refresh user data:

```powershell
python app.py update users
```

Refresh user data without price history:

```powershell
python app.py update users --skip-price-history
```

Refresh user data and re-run skill analysis after the update:

```powershell
python app.py update users --user 0x... --skip-price-history --analyze-after
```

Refresh stored market-trade data used by the scanner:

```powershell
python app.py update market-trades
```

Refresh market trades and re-run the scanner:

```powershell
python app.py update market-trades --analyze-after
```

Run luck/skill analysis from the local database:

```powershell
python app.py analyze luck
```

Reanalyze only one user:

```powershell
python app.py analyze luck --user 0x...
```

Run market scanner analysis from the local database:

```powershell
python app.py analyze scanner
```

Start the local dashboard and job control panel:

```powershell
python app.py dashboard serve
```

Then open `http://127.0.0.1:8765`.

## Dashboard

The served dashboard reads SQLite directly. It organizes markets and users into separate work areas, can launch updates and analyses from the page, and shows the selected user's market-grouped trades on the same screen.

The Markets tab has one database-backed market list. Rows can come from discovery, manual conditionId entry, targeted updates, or market analysis. Use **Discover Markets** to add markets from discovery rules, **Update Markets** to refresh metadata and price history for the listed rows, **Update Trades** to refresh market-trade rows, and **Analyze Markets** to run the signal scanner.

The Users tab prioritizes information-edge signals, candidate users from market analysis, realized profit, and per-trade monetized edge. User updates can optionally re-run analysis after new data is fetched. Job output is written to `reports/jobs` and can be viewed or cancelled from the Jobs tab.

## Behavior

Analysis is offline by default. Use update commands for API calls, or pass explicit opt-in flags:

```powershell
python app.py analyze luck --hydrate-missing-users
python app.py analyze scanner --allow-api
```

Luck/skill analysis skips the large price-history lookup by default. Use `python app.py analyze luck --use-price-history` only when you want the slower settlement-price fallback.

Runs and job metadata are written under `reports/run_state` and `reports/jobs`. Data watermarks are written under `data/.state/watermarks.json`.

`run_pipeline.py` still exists, but it now delegates to `app.py`. For an offline run:

```powershell
python run_pipeline.py --offline
```
