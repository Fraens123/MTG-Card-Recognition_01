import sqlite3

c = sqlite3.connect('tcg_database/database/karten.db')

# Prüfe scenario-Spalte
cols = [r[1] for r in c.execute("PRAGMA table_info(card_embeddings)").fetchall()]
print(f"Spalten in card_embeddings: {cols}")
print(f"scenario-Spalte vorhanden: {'scenario' in cols}")

# Count per scenario
if 'scenario' in cols:
    rows = c.execute("""
        SELECT COALESCE(scenario, 'NULL'), mode, COUNT(*) 
        FROM card_embeddings 
        GROUP BY COALESCE(scenario, 'NULL'), mode
    """).fetchall()
    print("\n=== Embeddings nach Scenario/Mode ===")
    for s, m, cnt in rows:
        print(f"  {s}/{m}: {cnt:,}")
else:
    print("\n[FEHLER] scenario-Spalte fehlt in card_embeddings!")
    total = c.execute("SELECT COUNT(*) FROM card_embeddings").fetchone()[0]
    print(f"Gesamt Embeddings (ohne scenario): {total:,}")

c.close()
