import sqlite3

c = sqlite3.connect('tcg_database/database/karten.db')
c.execute('DELETE FROM card_embeddings WHERE scenario=?', ('train500',))
deleted = c.total_changes
c.commit()
print(f'Gelöscht: {deleted:,} alte train500 Embeddings')
c.close()
print('\nJetzt neu exportieren:')
print('  python -m src.training.export_embeddings --config config.train500.yaml --mode analysis')
print('  python -m src.training.export_embeddings --config config.train500.yaml --mode runtime')
