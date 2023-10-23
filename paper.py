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
import redis

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_BASE'] = 'https://api.aiproxy.io/v1'
redis_url = os.getenv('REDIS_URL')

embeddings = OpenAIEmbeddings()
r = redis.from_url(redis_url)

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
    r.rpush('cached_ids',id)

def get_today_list(day=0):
    ids = r.lrange('cached_ids',0,-1)
    ids = [id.decode("utf-8") for id in ids]
    today = (datetime.date.today() - datetime.timedelta(day)).strftime('%Y-%m-%d')
    url = 'https://huggingface.co/papers?date='+today
    x = requests.get(url)
    data = x.text
    regex = re.compile(r'<a href="/papers/(.*?)"')
    papers = re.findall(regex,data)
    paperlist = list(set(papers))
    paperlist = [paper for paper in paperlist if len(paper) == 10]
    arxivids = [paper for paper in paperlist if paper not in ids]
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



