import datetime

import chromadb

import pandas as pd
from chromadb.api.types import IncludeEnum
from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"


def get_db_collection():
    # 連接地端的database
    chroma_client = chromadb.PersistentClient(path=dbpath)
    # 建立embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    # 建立collection
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    return collection

def query_result_to_dictlist(query_result):
    ids = query_result['ids'][0]
    metadatas = query_result['metadatas'][0]
    distances = query_result['distances'][0]
    resultDictList = []
    for index in range(len(ids)):
        resultDictList.append({"id": ids[index], "metadatas": metadatas[index], "similar": 1 - distances[index]})
    resultDictList = sorted(resultDictList, key=lambda x: x["similar"], reverse=True)
    return resultDictList


def generate_hw01():
    collection = get_db_collection()
    df = pd.read_csv("COA_OpenData.csv")
    for idx, row in df.iterrows():
        crateDateTimeString = str.strip(row["CreateDate"])
        crateDateTime = datetime.datetime.strptime(crateDateTimeString, "%Y-%m-%d")
        crateDateTimeStamp = int(crateDateTime.timestamp())
        metadata = {"file_name": "COA_OpenData.csv",
                    "name": row["Name"],
                    "type": row["Type"],
                    "address": row["Address"],
                    "tel": row["Tel"],
                    "city": row["City"],
                    "town": row["Town"],
                    "date": crateDateTimeStamp}
        get_result = collection.get([row["ID"]])
        # print(get_result["ids"])
        if len(get_result["ids"]) == 0:
            # print("Document:", row["HostWords"], "Metadata:", metadata)
            collection.add(
                ids=[row["ID"]],
                documents=[row["HostWords"]],
                metadatas=[metadata])
    return collection


def generate_hw02(question, city, store_type, start_date, end_date):
    collection = get_db_collection()
    query_result = collection.query(
        query_texts=[question],
        n_results=10,
        include=[IncludeEnum.metadatas, IncludeEnum.distances],
        where={
            "$and": [
                {
                    "date" : {
                        "$gte": start_date.timestamp()
                    }
                },
                {
                    "date" : {
                        "$lte": end_date.timestamp()
                    }
                },
                {
                    "city" : {
                        "$in" : city
                    }
                },
                {
                    "type" : {
                        "$in" : store_type
                    }
                }
            ]
        }
    )

    resultDictList = query_result_to_dictlist(query_result)
    result = []
    for curl in resultDictList:
        similar = curl["similar"]
        name = curl["metadatas"]["name"]
        if similar >= 0.8:
            result.append(name)
        # print(similar, name)
    return result
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = get_db_collection()
    get_result = collection.get(where={"name": {"$eq": store_name}})
    if len(get_result["ids"]) > 0:
        metadata = get_result["metadatas"][0]
        metadata["new_store_name"] = new_store_name
        collection.update(get_result["ids"], metadatas=metadata)
    query_result = collection.query(query_texts=question,
                                    n_results=10,
                                    include=[IncludeEnum.metadatas, IncludeEnum.distances],
                                    where={
                                        "$and":[
                                            {
                                                "city":{
                                                    "$in" : city
                                                }
                                            },
                                            {
                                                "type": {
                                                    "$in" : store_type
                                                }
                                            }
                                        ]
                                    })
    resultDictList = query_result_to_dictlist(query_result)
    result = []
    for curl in resultDictList:
        similar = curl["similar"]
        metadata = curl["metadatas"]
        name = metadata["name"]
        if similar >= 0.8:
            if "new_store_name" in metadata :
                result.append(metadata["new_store_name"])
                print(similar, name, metadata["new_store_name"])
            else:
                result.append(name)
                print(similar, name)
    return result

if __name__ == "__main__" :
    # print(generate_hw01())
    # print(generate_hw02("我想要找有關茶餐點的店家",
    #                     ["宜蘭縣", "新北市"],
    #                     ["美食"],
    #                     datetime.datetime(2024, 4, 1),
    #                     datetime.datetime(2024, 5, 1)))
    # result = generate_hw03("我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵",
    #               "耄饕客棧",
    #               "田媽媽（耄饕客棧）",
    #               ["南投縣"],
    #               ["美食"])
    # print(result)
    pass
