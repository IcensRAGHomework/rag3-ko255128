import datetime
import time

import chromadb
import traceback
import csv

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

from langchain_chroma import Chroma

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"}
    )
    with open('COA_OpenData.csv', encoding="utf-8-sig") as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            crateDateTimeString = str.strip(row["CreateDate"])
            crateDateTime = datetime.datetime.strptime(crateDateTimeString,"%Y-%m-%d")
            crateDateTimeStamp = int(time.mktime(crateDateTime.timetuple()))
            collection.add(
                ids=[row["ID"]],
                documents=[row["HostWords"]],
                metadatas=[{"file_name": "COA_OpenData.csv",
                            "name": row["Name"],
                            "type":row["Type"],
                            "address":row["Address"],
                            "tel":row["Tel"],
                            "city":row["City"],
                            "town":row["Town"],
                            "date":crateDateTimeStamp}])
    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    return collection

# if __name__ == "__main__" :
#     print(generate_hw01())