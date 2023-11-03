from mongo_db_function import *
from datetime import datetime


def delete_old_datasets():
    db = get_database("FIT4701")

    # Query for documents with "create_date" less than the specified date
    results = db["Dataset"].find({
        "create_date": {
            "$lt": datetime(2023, 8, 18, 0, 0, 0),
            "$gt": datetime(2023, 8, 15, 0, 0, 0)
        }
    })

    dataset_ids = []
    for r in results:
        dataset_ids.append(str(r["_id"]))
    print("Dataset IDs: ", dataset_ids)

    data_results = db["Data"].count_documents({"DATASET_ID": {"$in": dataset_ids}})
    print("Data found: ", data_results)

    # DELETE OPERATIONS; BE VERY CAREFUL
    print(db["Data"].delete_many({"DATASET_ID": {"$in": dataset_ids}}).deleted_count)
    print(db["Dataset"].delete_many({
        "create_date": {
            "$lt": datetime(2023, 8, 18, 0, 0, 0),
            "$gt": datetime(2023, 8, 15, 0, 0, 0)
        }
    }).deleted_count)


if __name__ == '__main__':
    delete_old_datasets()
