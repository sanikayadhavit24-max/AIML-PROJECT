
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")

db = client["sms_spam_db"]
collection = db["sms_history"]


# CREATE
def create_sms(sms_text, prediction):
    result = collection.insert_one({
        "sms_text": sms_text,
        "prediction": prediction,
        "created_at": datetime.now()
    })
    return str(result.inserted_id)


# READ
def read_sms():
    records = collection.find().sort("created_at", -1)
    return [
        {
            "id": str(r["_id"]),
            "sms_text": r["sms_text"],
            "prediction": r["prediction"]
        }
        for r in records
    ]


# UPDATE
def update_sms(record_id, new_sms_text, new_prediction):
    if not ObjectId.is_valid(record_id):
        return False

    result = collection.update_one(
        {"_id": ObjectId(record_id)},
        {"$set": {
            "sms_text": new_sms_text,
            "prediction": new_prediction,
            "updated_at": datetime.now()
        }}
    )
    return result.matched_count > 0


# DELETE
def delete_sms(record_id):
    if not ObjectId.is_valid(record_id):
        return False

    result = collection.delete_one(
        {"_id": ObjectId(record_id)}
    )
    return result.deleted_count > 0
