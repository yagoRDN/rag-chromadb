import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.foundation_models.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
import subprocess
import gzip
import json
import chromadb
import random
import string



def get_credentials(api_key):

	return {
		"url" : "https://us-south.ml.cloud.ibm.com",
		"apikey" : api_key
	}

def hydrate_chromadb():
    data = client.data_assets.get_content(vector_index_id)
    content = gzip.decompress(data)
    stringified_vectors = str(content, "utf-8")
    vectors = json.loads(stringified_vectors)

    chroma_client = chromadb.Client()

    # make sure collection is empty if it already existed
    collection_name = "my_collection"
    try:
        collection = chroma_client.delete_collection(name=collection_name)
    except:
        print("Collection didn't exist - nothing to do.")
    collection = chroma_client.create_collection(name=collection_name)

    vector_embeddings = []
    vector_documents = []
    vector_metadatas = []
    vector_ids = []

    for vector in vectors:
        vector_embeddings.append(vector["embedding"])
        vector_documents.append(vector["content"])
        metadata = vector["metadata"]
        lines = metadata["loc"]["lines"]
        clean_metadata = {}
        clean_metadata["asset_id"] = metadata["asset_id"]
        clean_metadata["asset_name"] = metadata["asset_name"]
        clean_metadata["url"] = metadata["url"]
        clean_metadata["from"] = lines["from"]
        clean_metadata["to"] = lines["to"]
        vector_metadatas.append(clean_metadata)
        asset_id = vector["metadata"]["asset_id"]
        random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        id = "{}:{}-{}-{}".format(asset_id, lines["from"], lines["to"], random_string)
        vector_ids.append(id)

    collection.add(
        embeddings=vector_embeddings,
        documents=vector_documents,
        metadatas=vector_metadatas,
        ids=vector_ids
    )
    return collection
	
def proximity_search( question ):
    query_vectors = emb.embed_query(question)
    query_result = chroma_collection.query(
        query_embeddings=query_vectors,
        n_results=vector_index_properties["settings"]["top_k"],
        include=["documents", "metadatas", "distances"]
    )


def main():
	
    load_dotenv()
    load_dotenv('config.env')

    model_id = "meta-llama/llama-3-405b-instruct"

    parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1
}

    project_id = os.getenv("PROJECT_ID")
    space_id = os.getenv("SPACE_ID")


    model = Model(
	model_id = model_id,
	params = parameters,
	credentials = get_credentials(),
	project_id = project_id,
	#space_id = space_id
	)

    wml_credentials = get_credentials(os.getenv("API_KEY"))
    client = APIClient(credentials=wml_credentials, project_id=project_id, space_id=space_id)

    vector_index_id = "d7293128-d677-487b-a8d4-1bd40487dade"
    vector_index_details = client.data_assets.get_details(vector_index_id)
    vector_index_properties = vector_index_details["entity"]["vector_index"]




    return 0