import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.foundation_models.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
import subprocess
import gzip
import json
import chromadb
from chromadb import Client
import random
import string



def get_credentials(api_key):

	return {
		"url" : "https://us-south.ml.cloud.ibm.com",
		"apikey" : api_key
	}

def hydrate_chromadb(client, vector_index_id):
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
	
def proximity_search(question, emb, vector_properties):
    
    query_vectors = emb.embed_query(question)
    query_result = chroma_collection.query(
        query_embeddings=query_vectors,
        n_results=vector_index_properties["settings"]["top_k"],
        include=["documents", "metadatas", "distances"]
    )

    documents = list(reversed(query_result["documents"][0]))

    return "\n".join(documents)


def main():
	
    prompt_input = """<|start_header_id|>system<|end_header_id|>

You always answer the questions with markdown formatting. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.

Any HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.

When returning code blocks, specify language.

Given the document and the current conversation between a user and an assistant, your task is as follows: answer any user query by using information from the document. Always answer as helpfully as possible, while being safe. When the question cannot be answered using the context or document, output the following response: "I cannot answer that question based on the provided document.".

Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. ${teste}
{test}<|eot_id|><|start_header_id|>user<|end_header_id|>

__grounding__"""

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

    emb = SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2')

    wml_credentials = get_credentials(os.getenv("API_KEY"))
    client = APIClient(credentials=wml_credentials, project_id=project_id, space_id=space_id)

    vector_index_id = os.getenv("VECTOR_INDEX_ID")
    vector_index_details = client.data_assets.get_details(vector_index_id)
    vector_index_properties = vector_index_details["entity"]["vector_index"]

    chroma_collection = hydrate_chromadb(client=client, vector_index_id=vector_index_id)

    question = "Question: tenho um carro audi manual, como funciona o seguro para ele?"
    grounding = proximity_search(question=question, emb=emb, vector_properties=vector_index_properties)
    formattedQuestion = f"""<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>

    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
    prompt = f"""{prompt_input}{formattedQuestion}"""
    generated_response = model.generate_text(prompt=prompt.replace("__grounding__", grounding), guardrails=False)
    print(f"AI: {generated_response}")

    return generated_response



if __name__ == "__main__":
	main()