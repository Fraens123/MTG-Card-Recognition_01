import sqlite3

c = sqlite3.connect('tcg_database/database/karten.db')
rows = c.execute('SELECT scenario, COUNT(*) FROM card_embeddings WHERE mode=? GROUP BY scenario', ('analysis',)).fetchall()
print('\n=== Analysis Embeddings per Scenario ===')
for scenario, cnt in rows:
    print(f'  {scenario or "(default/NULL)"}: {cnt:,}')

total = c.execute('SELECT COUNT(*) FROM card_embeddings WHERE mode=?', ('analysis',)).fetchone()[0]
print(f'\nGesamt analysis: {total:,}')
c.close()
