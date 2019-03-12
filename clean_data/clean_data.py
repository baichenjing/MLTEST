import json

import pymongo
import requests
from bson import ObjectId
from pymongo import MongoClient
def clean_data():
    host="166.111.7.173"
    port=30019
    db_name="bigsci"
    user="kegger_bigsci"
    password="datiantian123!@#"
    db=MongoClient(host,port,connect=False)[db_name]
    db.authenticate(user,password)
    tasks=db['task'].find()
    for t in tasks:
        task_id=t['_id']
        persons=db['crawled_person_final'].find({'task_id':task_id})
        db['crawled_person_final2'].insert_many(persons)




