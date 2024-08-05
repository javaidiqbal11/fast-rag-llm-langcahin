import os
import logging
from huggingface_hub import login
from fastapi import FastAPI, UploadFile, HTTPException, File, Form ,UploadFile, Depends,Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List,Optional
import openai
import tempfile
import numpy as np
from datetime import datetime
from langchain_openai import OpenAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from enum import Enum
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import pickle
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.document_transformers import (
    LongContextReorder,
)
import uuid
from langchain.retrievers.multi_vector import SearchType
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_community.document_loaders import TextLoader
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_huggingface.llms import HuggingFacePipeline

load_dotenv()
mult_prompt="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: """
QDRANT_DIR = "./Qdrant"
USER_NAME = "user"
ASSISTANT_NAME = "assistant"
model = "gpt-4o"
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = OpenAI(api_key=api_key)

app = FastAPI()

EMBEDDING_MAPPING_FILE = "embedding_mapping.json"

def create_chunks(text, chunk_size, chunk_overlap):
    chunks = []
    text_length = len(text)
    start = 0
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start = end - chunk_overlap if end < text_length else end
    return chunks


client = QdrantClient(url="http://localhost:6333")
# Middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MVR=None

async def load_mappings(json_file: UploadFile):
    try:
        content = await json_file.read()
        data = json.loads(content)
        return {entry['clear_text']: entry['uuid'] for entry in data}
    except json.JSONDecodeError:
        print(f"Error: The provided JSON file is not a valid JSON file.")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

# Function to replace text in documents
def replace_text_in_documents(docs, mappings):
    for doc in docs:
        content = doc.page_content
        for clear_text, uuid in mappings.items():
            content = content.replace(clear_text, uuid)
        doc.page_content = content
    return docs


@app.post("/abc")
async def health():
    return "API working success"


def get_existing_collections() -> List[str]:
    collections = []
    for collection in client.get_collections().collections:
        collections.append(collection.name)
    return collections
existing_collections = get_existing_collections()
import json
def load_embedding_mappings():
    if os.path.exists(EMBEDDING_MAPPING_FILE):
        with open(EMBEDDING_MAPPING_FILE, 'r') as file:
            return json.load(file)
    return {}
EMBEDDING_DIMENSIONS = {
    "bge-small-en": 384,
    "gte-base": 768,
    "GIST-small-Embedding-v0": 384,
    "OpenAi": 1536,
    "openai":1536,
    "Mistral-7B": 1024,
    "Cohere-7B": 1024
}
class RAGType(str, Enum):
    CONTEXTUAL_COMPRESSION = "Contextual Compression"
    NORMAL_RAG = "Normal RAG"
    MULTIVECTOR_RETRIEVER = "MultiVector Retriever"
    MULTIQUERY_RETRIEVER = "MultiQueryRetriever"
class Model(str, Enum):
    Mixtral_7B = "Mixtral 7B"
    Saul_7B = "Saul 7B"
    Tiny_LLM = "Tiny LLM"
    OPENAI = "openAI"
class ChunkingMethod(str, Enum):
    semantic = "semantic chunking $"
    recursive = "recursive chunking"
    token = "token chunking"
    MULTIVECTOR_RETRIEVER = "multivectorstore_retriver"

class EmbedMethod(str, Enum):
    bge_small_en = "bge-small-en"
    gte_small = "gte-base"
    gist_small_embedding_v0 = "GIST-small-Embedding-v0"
    openai = "OpenAI $"
    Mistral_7B = "Mistral-7B"
    Cohere_7B = "Cohere-7B"


class PreProcessing(str, Enum):
    Nothing = " "
    MULTI_QUERY = "Multi Query"
    
class PostProcessing(str, Enum):
    LONG_CONTEXT_REORDER = "Long-Context Reorder"
    RE_RANKER = "Re Ranker"
    Time_Sort = "Time Sort"

def parse_timestamp(doc):
    return datetime.strptime(doc.metadata['TimeStamp'], "%Y-%m-%d %H:%M:%S")

def save_embedding_mappings(mappings):
    with open(EMBEDDING_MAPPING_FILE, 'w') as file:
        json.dump(mappings, file)
def load_embedding_mappings():
    if os.path.exists(EMBEDDING_MAPPING_FILE):
        with open(EMBEDDING_MAPPING_FILE, 'r') as file:
            return json.load(file)
    return {}
def delete_mapping(collection_name):
    mappings = load_embedding_mappings()
    if collection_name in mappings:
        del mappings[collection_name]
        save_embedding_mappings(mappings)
        print(f"Deleted mapping for collection: {collection_name}")
    else:
        print(f"collection name {collection_name} not found.")
def clean_mappings_file():
    if os.path.exists(EMBEDDING_MAPPING_FILE):
        with open(EMBEDDING_MAPPING_FILE, 'w') as file:
            file.write('{}')
        print(f"Cleaned the contents of the file: {EMBEDDING_MAPPING_FILE}")
    else:
        print(f"File {EMBEDDING_MAPPING_FILE} does not exist.")
def get_embeddings(embed_method):
    # Check if embeddings exist in the pickle file
    if embed_method==EmbedMethod.openai:
        return OpenAIEmbeddings()
    elif os.path.exists(embed_method+".pkl"):
        with open(embed_method+".pkl", 'rb') as file:
            embeddings = pickle.load(file)
        return embeddings
    print(embed_method)
    #print(embed_method.name)
    if embed_method == EmbedMethod.bge_small_en:
        emb = "BAAI/bge-small-en"
    elif embed_method == EmbedMethod.gte_small:
        emb = "Alibaba-NLP/gte-base-en-v1.5"
    elif embed_method == EmbedMethod.gist_small_embedding_v0:
        emb = "avsolatorio/GIST-small-Embedding-v0"
    elif embed_method == EmbedMethod.Mistral_7B:
        emb = "Salesforce/SFR-Embedding-Mistral"
    elif embed_method == EmbedMethod.Cohere_7B:
        emb = "Cohere/Cohere-embed-multilingual-v3.0"
    embeddings=HuggingFaceEmbeddings(
        model_name=emb,
        )
    with open(embed_method+".pkl", 'wb') as file:
        pickle.dump(embeddings, file)
    print("HERE")
    return embeddings

embedding_storage = load_embedding_mappings()
store = InMemoryByteStore()
id_key = "doc_id"

@app.post("/ingest")
async def ingest(method: ChunkingMethod = Form(...),embed_method: EmbedMethod = Form(...),chunk_size: int = Form(...),chunk_overlap: int = Form(...),files: List[UploadFile] = File(...),collection: str = Form(..., description="Select an existing collection or provide a new one"),Mask: Optional[bool] = Form(None)):    
    try:
        global MVR
        embeddings = get_embeddings(embed_method)
        print(embed_method.name)
        embedding_key = str(embed_method.name).replace('_', '-')
        print(EMBEDDING_DIMENSIONS.get(embedding_key))

        mappings = {}
        if Mask:
            mappings = await load_mappings("mask.json")
            print("Loaded mappings:", mappings)
        if collection not in get_existing_collections():
            embedding_storage[collection] = embed_method
            save_embedding_mappings(embedding_storage)
        else:
            if embedding_storage.get(collection) != embed_method:
                print('collection alredy exists')
                return JSONResponse(status_code=400, content={"message": f"Embedding method mismatch for collection '{collection}'. Expected: {embedding_storage[collection]}, Provided: {embed_method}"})

        if method.name =="MULTIVECTOR_RETRIEVER":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000)
            child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        elif method.name == "semantic":
            text_splitter = SemanticChunker(OpenAIEmbeddings())
        elif method.name == "recursive":
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        else:
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        for file in files:
            temp_file_path=""
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name
                # Process the file
                print(f"Saved file {file.filename} to {temp_file_path}")
            if file.filename.endswith(".pdf"):
                loader=PyMuPDFLoader(temp_file_path)
            else:
                loader = TextLoader(temp_file_path)
            docs=loader.load_and_split(text_splitter=text_splitter)
            for doc in docs:
                doc.metadata['TimeStamp']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            e = replace_text_in_documents(docs, mappings)
            # for doc in docs:
            #     print(doc.page_content) 
            vectorstore = Qdrant.from_documents(
                    docs,
                    embeddings,
                    location='http://localhost:6333/dashboard',
                    prefer_grpc=True,
                    collection_name=collection)
            os.remove(temp_file_path)
        print("Sucessfully Ingested PDF/PDF's")

        if method.name == "MULTIVECTOR_RETRIEVER":
            MVR = MultiVectorRetriever(
                vectorstore=vectorstore,
                byte_store=store,
                id_key=id_key,
                search_kwargs={"k": 8}
            )
            sub_docs = []
            doc_ids = [str(uuid.uuid4()) for _ in docs]
            for i, doc in enumerate(docs):
                _id = doc_ids[i]
                _sub_docs = child_text_splitter.split_documents([doc])
                for _doc in _sub_docs:
                    _doc.metadata[id_key] = _id
                sub_docs.extend(_sub_docs)
            MVR.search_type = SearchType.mmr
            MVR.vectorstore.add_documents(sub_docs)
            MVR.docstore.mset(list(zip(doc_ids, docs)))
        return JSONResponse(status_code=200, content={"message": "Sucessfully Ingested and Created RAG"})
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        return JSONResponse(status_code=400, content={"message": "Invalid files or error processing files."})


@app.post("/retrieval")
async def retrieval(RAG_Type: RAGType = Form(..., description="Select the RAG type"),PostProcessing: PostProcessing = Form(..., description="Select the PostProcessing type"),PreProcessing: PreProcessing = Form(..., description="Select the PreProcessing type"),query: str = Form(...), k: int = Form(...),collection: str = Form(..., description="Select an existing collection")):
    global MVR
    embedding_name=embedding_storage[collection]
    embedding=get_embeddings(embedding_name)
    vectorstore=Qdrant.from_existing_collection(embedding,collection_name=collection)
    prompts=[]
    if PreProcessing.name == "MULTI_QUERY":
        print("hello")
        llm= ChatOpenAI(model="gpt-4o")
        response=llm([HumanMessage(content=mult_prompt+query+"Seprate each question by a new line")])
        prompts=response.content.split('\n')
        
    prompts.append(query)
    
    if RAG_Type.name == "NORMAL_RAG":
        retriever=vectorstore.as_retriever(search_kwargs={"k": k})
        print("here")
    elif RAG_Type.name == "CONTEXTUAL_COMPRESSION":
        llm = OpenAI(temperature=0)
        if PostProcessing.name == "RE_RANKER":
            compressor = FlashrankRerank()
        else:
            compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vectorstore.as_retriever(search_kwargs={"k": k})
        )
    elif RAG_Type.name == "MULTIQUERY_RETRIEVER":
        llm = ChatOpenAI(model='gpt-4o',temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": k}), llm=llm
        )
    elif RAG_Type.name == "MULTIVECTOR_RETRIEVER":
        retriever = MVR
    query=prompts[0]
    Final_Docs=retriever.invoke(query)
    for query in prompts[1:]:
        docs=retriever.invoke(query)
        for doc in docs:
            Final_Docs.append(doc)
    if PostProcessing.name == "LONG_CONTEXT_REORDER":
        reordering = LongContextReorder()
        Final_Docs = reordering.transform_documents(Final_Docs)
    elif PostProcessing.name == "RE_RANKER" and RAG_Type.name != "CONTEXTUAL_COMPRESSION":
        compressor = FlashrankRerank()
        Final_Docs=compressor.compress_documents(Final_Docs,query)
    elif PostProcessing.name == "Time_Sort":
        Final_Docs=sorted(Final_Docs, key=parse_timestamp, reverse=True)
    return Final_Docs


@app.post("/query_chat_gpt")
async def get_response(RAG_Type: RAGType ,PostProcessing: PostProcessing ,PreProcessing: PreProcessing ,query: str , k: int,temperature: float =0.01,collection: str =None,date: str = datetime.now().isoformat(),time:str="00:00:00",json_data: Optional[dict] = None):   
    try:
        docs=await retrieval(RAG_Type,PostProcessing,PreProcessing,query,k,collection)
        for doc in docs:
            doc.page_content =doc.page_content+" Time of chunk: "+ doc.metadata['TimeStamp']
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        document_variable_name = "context"
        llm = ChatOpenAI(model="gpt-4o",temperature=temperature)
        stuff_prompt_override = """Given this text extracts:
        -----
        {context}
        -----
        Please answer the following question while keeping in mind that current date is {date} and time is {time}.
        Also prefer Recent/Newer answers more then old ones in case of contradictory information:
        {query}"""
        prompt = PromptTemplate(
            template=stuff_prompt_override, input_variables=["context", "query","date","time"]
        )

        # Instantiate the chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        result= chain.run(input_documents=docs, query=query,date=date,time=time)
        if json_data:
            json_template = """
            First, read the complete context:{context} and understand the JSON structure and its fields :{json_structure}.provide the results according to the provided json structure no add extra field.For example, if the provided JSON structure includes only the 'city' field, your response should only include the 'city'.
            Note : Do not starting with"```json".
            """

            prompt_template = PromptTemplate(input_variables = ["json_structure","context"], template = json_template)
            chain = LLMChain(llm = llm, prompt = prompt_template)
            print(chain.run({"json_structure":json_data,"context":result}))
            result_= chain.run({"json_structure":json_data,"context":result})
            if result_[:7]=="```json":
                result_=result_[7:-3]
            try:
                result_json=json.loads(result_)
            except:
                return result_ 
            return result_json
        else:
            return result
    except Exception as e:
        print(e)
        
def get_model(model: Model,temp:float):
    # Check if embeddings exist in the pickle file
    print(model.name+".pkl")
    if model.name=="OPENAI":
        return ChatOpenAI(model="gpt-4o",temperature=temp)
    
    elif os.path.exists(model.name+".pkl"):
        with open(model.name+".pkl", 'rb') as file:
            embeddings = pickle.load(file)
        return embeddings
    if model.name == "Mixtral_7B":
        name = "mistralai/Mistral-7B-v0.3"
    elif model.name == "Saul_7B":
        name = "Equall/Saul-7B-Base"
    elif model.name =="Tiny_LLM":
        name = "stabilityai/stablelm-3b-4e1t"
    login(token=os.environ['HUGGINGFACEHUB_API_TOKEN'])
    llm = HuggingFacePipeline.from_model_id(
    model_id=name,
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 4096},
)
    print("Done")
    with open(model.name+".pkl", 'wb') as file:
        pickle.dump(llm, file)
    return llm

@app.post("/query_local_llm")
async def get_response(Model_Type: Model, RAG_Type: RAGType ,PostProcessing: PostProcessing ,PreProcessing: PreProcessing ,query: str , k: int,temperature: float =0.01,collection: str =None,date: str = datetime.now().isoformat(),time:str="00:00:00",json_data: Optional[dict] = None):   
    try:
        docs=await retrieval(RAG_Type,PostProcessing,PreProcessing,query,k,collection)
        for doc in docs:
            doc.page_content =doc.page_content+" Time of chunk: "+ doc.metadata['TimeStamp']
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        document_variable_name = "context"
        llm = get_model(Model_Type,temperature)
        stuff_prompt_override = """Given this text extracts:
        -----
        {context}
        -----
        Please answer the following question while keeping in mind that current date is {date} and time is {time}.
        Also prefer Recent/Newer answers more then old ones in case of contradictory information:
        {query}"""
        prompt = PromptTemplate(
            template=stuff_prompt_override, input_variables=["context", "query","date","time"]
        )

        # Instantiate the chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        result= chain.run(input_documents=docs, query=query,date=date,time=time)
        if json_data:
            json_template = """
            First, read the complete context:{context} and understand the JSON structure and its fields :{json_structure}.provide the results according to the provided json structure no add extra field.For example, if the provided JSON structure includes only the 'city' field, your response should only include the 'city'.
            Note : Do not starting with"```json".
            """

            prompt_template = PromptTemplate(input_variables = ["json_structure","context"], template = json_template)
            chain = LLMChain(llm = llm, prompt = prompt_template)
            print(chain.run({"json_structure":json_data,"context":result}))
            result_= chain.run({"json_structure":json_data,"context":result})
            if result_[:7]=="```json":
                result_=result_[7:-3]
            try:
                result_json=json.loads(result_)
            except:
                return result_ 
            return result_json
        else:
            return result
    except Exception as e:
        print(e)



@app.get("/list_collections", response_model=List[str])
async def list_collections():
    try:
        collections_info = get_existing_collections()
        return collections_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/drop_collection/{collection_name}")
async def drop_collection(collection_name: str):
    try:
        if client.collection_exists(collection_name):
            delete_mapping(collection_name)
            client.delete_collection(collection_name)
            return {"message": f"collection '{collection_name}' dropped successfully"}
        else:
            raise HTTPException(status_code=404, detail="collection not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/drop_all_collections")
async def drop_all_collections():
    try:
        collections_info = get_existing_collections()
        for collection_name in collections_info:
            client.delete_collection(collection_name)
        clean_mappings_file()
        return {"message": "All collections dropped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
