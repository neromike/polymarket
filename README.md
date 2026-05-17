# Polymarket Local Data App

This project is organized around three separate workflows:

- `update`: API-backed cache refreshes
- `analyze`: offline analysis from cached data
- `dashboard`: dashboard generation or local job control

The main entrypoint is:

```powershell
python app.py status
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

Refresh market-trade caches used by the scanner:

```powershell
python app.py update market-trades
```

Refresh market-trade caches and re-run the scanner:

```powershell
python app.py update market-trades --analyze-after
```

Run luck/skill analysis from cached data:

```powershell
python app.py analyze luck
```

Reanalyze only one cached user after updating that user:

```powershell
python app.py analyze luck --user 0x...
```

Run market scanner analysis from cached data:

```powershell
python app.py analyze scanner
```

Build both static dashboards:

```powershell
python app.py dashboard build
```

Start the local dashboard and job control panel:

```powershell
python app.py dashboard serve
```

Then open `http://127.0.0.1:8765`.

The served dashboard is dynamic. It reads the current report CSVs, shows cache freshness,
organizes markets and users into separate work areas, and can launch update or analysis
jobs from the buttons on the page. The Markets tab has one market list; rows can come
from pasted/tracked conditionIds, recent cached markets, or signal discovery. Use
**Discover Markets** to cache more markets from discovery rules, and **Refresh Signals**
to update market-trade caches and re-run the signal scanner. The Users
tab sorts users by skill by default, shows the selected user's market-grouped trades on
the same screen, and refreshes analysis automatically after user updates. Job output is written
to `reports/jobs` and can be viewed or cancelled from the Jobs tab.

## Behavior Changes

Analysis is offline by default. Use update commands for API calls, or pass explicit opt-in flags:

```powershell
python app.py analyze luck --hydrate-missing-users
python app.py analyze scanner --allow-api
```

Luck/skill analysis also skips the large cached price-history scan by default. Use
`python app.py analyze luck --use-price-history` only when you want the slower
settlement-price fallback.

Runs and job metadata are written under `reports/run_state` and `reports/jobs`. Data watermarks are written under `data/.state/watermarks.json`.

`run_pipeline.py` still exists, but it now delegates to `app.py`. For a cache-only run:

```powershell
python run_pipeline.py --offline
```
