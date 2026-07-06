import duckdb

con = duckdb.connect('data/football.duckdb', read_only=True)

# Check sporttery_matches data
print('=== Sporttery Matches Sample ===')
sample = con.execute("SELECT match_id, business_date, match_number, league_name, home_team, away_team, sell_status FROM sporttery_matches LIMIT 10").fetchall()
for s in sample:
    print(f'  {s}')

# Check if today's matches exist
print('\n=== Today matches ===')
today_count = con.execute("SELECT COUNT(*) FROM sporttery_matches WHERE business_date = '2026-07-05'").fetchone()[0]
print(f'Today (2026-07-05) matches: {today_count}')

# Check all dates
dates = con.execute("SELECT DISTINCT business_date, COUNT(*) as cnt FROM sporttery_matches GROUP BY business_date ORDER BY business_date").fetchall()
print('\nAll business dates:')
for d in dates:
    print(f'  {d[0]}: {d[1]} matches')

# Check latest sporttery update
print('\n=== Latest Sporttery Update ===')
latest = con.execute('SELECT MAX(last_update) FROM sporttery_matches').fetchone()[0]
print(f'Latest update: {latest}')

# Check model_registry content
print('\n=== Model Registry ===')
models = con.execute('SELECT model_id, model_type, version, status, metrics_json FROM model_registry').fetchall()
for m in models:
    print(f'  {m[0]} | {m[1]} | {m[2]} | {m[3]} | {str(m[4])[:100] if m[4] else "None"}')

# Check predictions
print('\n=== Predictions ===')
preds = con.execute('SELECT prediction_id, match_id, model_version, home_probability, draw_probability, away_probability, confidence FROM predictions LIMIT 10').fetchall()
for p in preds:
    print(f'  {p}')

con.close()
