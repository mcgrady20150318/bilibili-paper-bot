import requests
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import Any
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import datetime
import re
import pickle
import redis
import arxiv
import asyncio
from mutagen.mp3 import MP3
from moviepy.editor import *
import edge_tts
import pdf2image
import codecs

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_BASE'] = 'https://api.aiproxy.io/v1'
redis_url = os.getenv('REDIS_URL')

llm = OpenAI(max_tokens=3000,model_name='gpt-3.5-turbo')
embeddings = OpenAIEmbeddings()
r = redis.from_url(redis_url)
VOICE = "zh-CN-XiaoyiNeural"  

def get_paper_info(id,max_results=1):

    big_slow_client = arxiv.Client(
        page_size = 1,
        delay_seconds = 10,
        num_retries = 1
    )

    search = arxiv.Search(
        id_list=[id],
        max_results = max_results,
    )

    result = big_slow_client.results(search)
    result = list(result)[0]
    title = result.title
    abstract = result.summary
    return title,abstract.replace('\n',' ').replace('{','').replace('}','')

def generate_readme(id):
    title,abstract  = get_paper_info(id)
    f = codecs.open('./'+id+'/readme.txt','w',"utf-8")
    prompt_template =  """现在你是一个人工智能学者，请根据论文摘要"%s",严格按照如下xml格式生成内容，<ctitle>这里生成一个吸引读者的中文专业标题</ctitle> ,回车，<describe>这里生成一段350字左右的中文论文解读</describe>，回车，<tags>这里生成5个中文标签，并且以空格隔开</tags>，回车，如下：""" %(abstract)
    PROMPT = PromptTemplate(template=prompt_template, input_variables=[])
    chain = LLMChain(llm=llm, prompt=PROMPT)
    output = chain.run(text='')
    f.write(output+'\n')
    f.write("<title>"+ title+"</title>\n")
    f.write("<url>https://arxiv.org/pdf/" + id+"</url>\n")
    f.write("<comment>可以试试/ask + 你的提问和本篇论文进行交流</comment>")
    f.close()
    with open('./'+id+'/readme.txt', 'rb') as f:
        file_content = f.read()
    r.set('bilibili:'+id+':readme.txt',file_content)

def get_time_count(audio_file):
    audio = MP3(audio_file)
    time_count = int(audio.info.length)
    return time_count

def download_pdf(id):
    url = 'http://arxiv.org/pdf/'+id+'.pdf'
    response = requests.get(url)
    if not os.path.exists('./'+id):
        os.mkdir('./'+id)
        os.mkdir('./'+id+'/assets')
        os.mkdir('./'+id+'/audio')
        os.mkdir('./'+id+'/video')
    output_path = './' + id + '/' + id + '.pdf'
    with open(output_path, 'wb') as file:
        file.write(response.content)

def gen_assets(id):
    output_dir = './'+id+'/assets/'
    os.makedirs(output_dir, exist_ok=True)
    pages = pdf2image.convert_from_path('./'+id+'/'+id+'.pdf')
    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f'{i}.jpg')
        page.save(image_path, 'JPEG')

async def gen_voice(text,idx,id):
    text = "大家好！这是paperweekly机器人推荐的今日AI热文。" + text 
    text += '欢迎一键三连。'
    communicate = edge_tts.Communicate(text, VOICE, rate = '-8%')  
    await communicate.save('./'+id+'/audio/' + str(idx)+'.mp3')

def generate_video(id,summary):
    asyncio.run(gen_voice(summary,0,id))
    print('...audio...')
    images = os.listdir('./'+id+'/assets')
    images.sort(key=lambda x:int(x[:-4]))
    image_files = ['./'+id+'/assets/' + a for a in images][:8]
    audios = os.listdir('./'+id+'/audio')
    audios.sort(key=lambda x:int(x[:-4]))
    audio_files = ['./'+id+'/audio/' + a for a in audios]

    total_time = 0

    for audio_file in audio_files:
        total_time += get_time_count(audio_file)

    image_clip = ImageSequenceClip(image_files, fps=len(image_files)/total_time)
    audio_clip = concatenate_audioclips([AudioFileClip(c) for c in audio_files])
    video_clip = image_clip.set_audio(audio_clip)
    video_clip.write_videofile('./'+id+'/video/'+id+'.mp4',codec='libx264')
    with open('./'+id+'/video/'+id+'.mp4', 'rb') as f:
        file_content = f.read()
    r.set('bilibili:'+id+':'+id+".mp4",file_content)
    print('...generate video done...')
    
def generate_assets(id):
    download_pdf(id)
    print('...download...')
    gen_assets(id)
    with open('./'+id+'/assets/0.jpg', 'rb') as f:
        file_content = f.read()
    r.set('bilibili:'+id+':cover.jpg',file_content)
    print('...assets...')

def get_upload_info(id):
    f = codecs.open('./'+id+'/readme.txt','r',"utf-8")
    data = f.read()
    rex = r'<title>(.*?)</title>'
    title = re.findall(rex,data)[0]
    print(title)
    rex = r'<ctitle>(.*?)</ctitle>'
    ctitle = re.findall(rex,data)[0]
    print(ctitle)
    rex = r'<tags>(.*?)</tags>'
    # taglist = ['人工智能','机器学习']
    tags = re.findall(rex,data)[0]
    tags = tags.replace(' ',',')
    print(tags)
    rex = r'<describe>(.*?)</describe>'
    speech = re.findall(rex,data)[0]
    print(speech)
    rex = r'<url>(.*?)</url>'
    url = re.findall(rex,data)[0]
    print(url)
    rex = r'<comment>(.*?)</comment>'
    comment = re.findall(rex,data)[0]
    print(comment)
    describe = "论文标题：" + title + "\n" + "论文简述：" + speech + "\n" + "论文链接： " + url
    return ctitle,title,describe,tags,speech

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

def set_status(id):
    r.set('bilibili:'+id+":upload",0)

if __name__ == '__main__':
    ids = get_today_list()        
    print(ids)
    for id in ids:
        r.rpush('paper',id)
        try:
            generate_assets(id)
            generate_readme(id)
            _,_,_,_,summary = get_upload_info(id)
            generate_video(id,summary)
            generate_index(id)
            set_status(id)
        except:
            print('exception')





