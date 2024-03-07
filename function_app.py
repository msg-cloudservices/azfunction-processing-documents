import azure.functions as func
import logging
from azure.storage.blob import BlobClient, BlobServiceClient
import os
from io import BytesIO
import PyPDF2
import re
from openai import AzureOpenAI
import json

#imports needed for upload to azure ai search:
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient



app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", path="raw-data/{name}",
                               connection="datastorage_STORAGE") 
def blobtriggertryagain(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name}, "
                f"Blob Size: {myblob.length} bytes")
    
    filename=myblob.name.split('/', 1)[-1]
    connectionstring = os.getenv("datastorage_STORAGE")
    
    #download blob
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=connectionstring)
    container_client = blob_service_client.get_container_client(container="raw-data")
    
    download_pdf = container_client.download_blob(filename)     
    
    #parse pdf
    datastream = BytesIO() 
    download_pdf.download_to_stream(datastream)
    
    logging.info(f"downloaded stream: {datastream}")    
    
    filereader = PyPDF2.PdfReader(datastream)
    logging.info(f"PDF file has {len(filereader.pages)} pages")

    #extract text from pdf
    full_text=""
    for page in filereader.pages:
        full_text += page.extract_text()

    #normalize text
    text = re.sub(
        r"\s+", " ", full_text
    ).strip()  
    # Replace multiple spaces with a single space
    text = re.sub(r"[. ]+,", "", text)  
    # Remove spaces before commas
    text = text.replace("..", ".").replace(". .", ".").replace("\n", "").strip()
    
    logging.info(f"succesfully extracted and normalised text")


    #dividing text in chunks
    chunk_length=500
    chunks = []
    while len(text) > chunk_length:
        # Find the last period in the chunk to avoid breaking a sentence
        last_period_index = text[:chunk_length].rfind(".")
        if last_period_index == -1:  # If no period is found, use the chunk length
            last_period_index = chunk_length

        # Append the chunk and remove it from the text
        chunks.append(text[: last_period_index + 1])
        text = text[last_period_index + 1 :]
    # Add any remaining text as a chunk
    if text:
        chunks.append(text)
    
    logging.info(f"succesfully chunked text")


    
    #importing Azure OpenAI creds
    api_key = os.getenv("AZURE_OPENAI_KEY")
    os.environ["OPENAI_API_KEY"]= api_key
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name=os.getenv("EMBEDDING_DEPLOYMENT_NAME")

    #initialising AZure OpenAI client
    client = AzureOpenAI(
    azure_endpoint=azure_endpoint, api_key=api_key, api_version="2023-05-15"
    )
    
    #generate embeddings
    document_embeddings = []
    for counter, chunk in enumerate(chunks):
        try:
            response = client.embeddings.create(input=chunk, model=deployment_name)
            embedding_data = {
                "id": str(counter),
                "line": chunk,
                "embedding": response.data[0].embedding,
                "filename": filename,
            }
            document_embeddings.append(embedding_data)
            logging.info(f"Generated embedding for chunk {counter} of {filename}")
        except Exception as e:
            logging.info(f"Error generating embedding for chunk {counter} of {filename}: {e}")
    

    # upload finalised document with embeddings to blob storage  
    upload_data= json.dumps(document_embeddings)        
    
    #rename document
    jsonname=(filename.split('.')[0]+".json")

    upload_client= BlobClient.from_connection_string(conn_str=connectionstring, container_name="processed-data", blob_name=jsonname)
    
    upload_client.upload_blob(upload_data)

    logging.info(f"succesfully added embedded document to processed data storage")

    ################################################
    #create 2nd function!!!

    #this code relies on the creation of a second function, blob-triggered on container "processed-sampledata"
    
    #on upload of the processed data (including the embeddings) the processed documents are uploaded to azure ai search as follows:
    
    jsondata= json.load(myblob)
    filename=myblob.name.split('/', 1)[-1]

    logging.info(f'detected new blob: {filename} ')


    #define Azure AI config
    service_endpoint= os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    credential = AzureKeyCredential(key)
    
    #upload file to Ai Search Index
    search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
    search_client.upload_documents(jsondata)

    logging.info(f'Uploaded {filename} to Azure AI Search Index')
    #logging.info(f'Data: {jsondata}')

