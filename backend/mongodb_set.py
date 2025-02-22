from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()


uri = os.getenv("uri")
client = MongoClient(uri, server_api=ServerApi('1'))
dbName = "hacklyticDB"
collectionName = "users"
user_collection = client[dbName][collectionName]


def get_set_user(uid=None):
        try:
            user = user_collection.find_one({uid: uid})
            if user:
                return user
            else:
                data = {
                "uid": uid,
                "balance": 0
                }
                user = user_collection.insert_one(data)
                return user
        except Exception as e:
            return {"error": f"Invali: {str(e)}"}