import requests
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import Any
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.redis import Redis
import datetime
import re
import pickle


os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_BASE'] = 'https://api.aiproxy.io/v1'
redis_url = os.getenv('REDIS_URL')

embeddings = OpenAIEmbeddings()

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
    rds = Redis.from_documents(texts,embeddings,redis_url=redis_url,index_name=id)

def get_today_list():
    ids = []
    if os.path.exists('./arxiv.bin'):
        f = open('./arxiv.bin','rb') 
        ids = pickle.load(f)
    today = (datetime.date.today()).strftime('%Y-%m-%d')
    url = 'https://huggingface.co/papers?date='+today
    x = requests.get(url)
    data = x.text
    regex = re.compile(r'<a href="/papers/(.*?)"')
    papers = re.findall(regex,data)
    paperlist = list(set(papers))
    paperlist = [paper for paper in paperlist if len(paper) == 10]
    arxivids = [paper for paper in paperlist if paper not in ids]
    f = open('./arxiv.bin','wb')
    pickle.dump(paperlist,f)
    f.close()
    return arxivids
    
if __name__ == '__main__':
    ids = get_today_list()
    print(ids)
    for id in ids:
        try:
            download_pdf(id)
            generate_index(id)
        except:
            pass
