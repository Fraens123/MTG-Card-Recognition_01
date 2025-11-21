import sqlite3

conn = sqlite3.connect('tcg_database/database/karten.db')
cur = conn.cursor()

# Gesamtanzahl
cur.execute('SELECT COUNT(*) FROM karten')
total = cur.fetchone()[0]
print(f"✅ Gesamt Karten in DB: {total:,}\n")

# Nach Sprache
print("Karten nach Sprache:")
cur.execute('SELECT lang, COUNT(*) as cnt FROM karten GROUP BY lang ORDER BY cnt DESC')
for row in cur.fetchall():
    print(f"  {row[0]:5s}: {row[1]:>7,}")

# Beispiel-Karten
print("\n" + "="*60)
print("\nBeispiel-Karten (erste 5):")
cur.execute('SELECT name, "set", lang, collector_number FROM karten LIMIT 5')
for row in cur.fetchall():
    print(f"  {row[0]:30s} | {row[1]:5s} | {row[2]:2s} | #{row[3]}")

conn.close()

print("\n" + "="*60)
print("Erwartung: ~517.820 Karten (all_cards)")
if total >= 517000:
    print("✅ VOLLSTÄNDIG!")
else:
    print(f"⚠️ Nur {total:,} von ~517.820 - eventuell unvollständig")
