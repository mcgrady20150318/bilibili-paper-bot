import requests
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import Any
from langchain.vectorstores import Milvus
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
import datetime

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

os.environ["PINECONE_API_KEY"] = 'aca32600-be4c-42a1-959f-689de12a2d13'
os.environ["PINECONE_ENV"] = 'gcp-starter'

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

embeddings = OpenAIEmbeddings(openai_api_base='https://api.aiproxy.io/v1',openai_api_key='mk-WOb5xz6NB9NfXJM6REnzs4RGqcdYe0Q64hnbLulzPOEAXiP0')

def download_pdf(id):
    url = 'https://arxiv.org/pdf/'+id+'.pdf'
    response = requests.get(url)
    if not os.path.exists('./'+id):
        os.mkdir('./'+id)
    output_path = './' + id + '/' + id + '.pdf'
    with open(output_path, 'wb') as file:
        file.write(response.content)

def generate_index(id):
    loader = PyMuPDFLoader('./'+id+'/'+id+'.pdf')
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    index_name = 'qaoverpaper'
    vector_store = Pinecone.from_documents(texts, embeddings, index_name=index_name)

def get_today_list():
    today = (datetime.date.today()).strftime('%Y-%m-%d')
    url = 'https://huggingface.co/papers?date='+today
    x = requests.get(url)
    data = x.text
    regex = re.compile(r'<a href="/papers/(.*?)"')
    papers = re.findall(regex,data)
    paperlist = list(set(papers))
    return paperlist
    
if __name__ == '__main__':
    ids = get_today_list()
    for id in ids:
        try:
            download_pdf(id)
            generate_index(id)
        except:
            pass
