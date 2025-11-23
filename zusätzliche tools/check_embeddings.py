import sqlite3

c = sqlite3.connect('tcg_database/database/karten.db')
print("\n=== Embeddings in DB ===")
rows = c.execute('SELECT scenario, mode, COUNT(*) FROM card_embeddings GROUP BY scenario, mode').fetchall()
for scenario, mode, cnt in rows:
    print(f"  {scenario}/{mode}: {cnt:,}")

r = c.execute('SELECT COUNT(*) FROM card_embeddings WHERE scenario=? AND mode=?', ('train500','analysis')).fetchone()
print(f"\n=== train500/analysis: {r[0]:,} ===")
c.close()
