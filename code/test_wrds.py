import wrds
db = wrds.Connection(wrds_username='sidt28')
print(db.list_libraries()[:10])
db.close()
