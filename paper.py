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
from nider.core import Font
from nider.core import Outline
from nider.models import Content, Header, Image
from snownlp import SnowNLP

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_BASE'] = 'https://api.aiproxy.io/v1'
redis_url = os.getenv('REDIS_URL')

llm = OpenAI(max_tokens=10000,model_name='gpt-3.5-turbo-16k')
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
    prompt_template =  """现在你是一个人工智能学者，请根据论文摘要"%s",严格按照如下xml格式生成内容，<ctitle>这里生成一个吸引读者的中文专业标题，要求有信息量</ctitle> ,回车，<describe>这里生成一段350字左右的中文论文解读</describe>，回车，<problem>这里生成1个引导读者阅读的问题</problem>，回车，<tags>这里生成5个中文标签，并且以空格隔开</tags>，回车，如下：""" %(abstract)
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
    time_count = audio.info.length
    return time_count

def download_pdf(id):
    url = 'http://arxiv.org/pdf/'+id+'.pdf'
    response = requests.get(url)
    if not os.path.exists('./'+id):
        os.mkdir('./'+id)
        os.mkdir('./'+id+'/assets')
        os.mkdir('./'+id+'/poster')
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

def get_poster(text,id,idx):
    header = Header(text=text,
                    text_width=80,
                    font=Font(path='./FangZhengKaiTi-GBK-1.ttf',size=30),
                    align='center',
                    color='#000100',
                    )
    content = Content(header=header)
    img = Image(content,fullpath='./'+id+'/poster/'+str(idx) + '.png')
    img.draw_on_image('./'+id+'/assets/'+ str(idx) +'.jpg')
    os.rename('./'+id+'/poster/'+str(idx) + '.png','./'+id+'/poster/'+str(idx) + '.jpg')

def get_time_count(audio_file):
    audio = MP3(audio_file)
    time_count = audio.info.length
    return time_count

async def gen_voice(text,idx,id):
    communicate = edge_tts.Communicate(text, VOICE)  
    await communicate.save('./'+id+'/audio/' + str(idx)+'.mp3')

def get_text_seq(s,N):
    start = "大家好！这是paperweekly机器人推荐的今日AI热文。" 
    end = '欢迎一键三连。'
    _texts = SnowNLP(s).sentences
    if N > 14:
        N = 14
    pages = N - 2
    n = len(_texts)
    texts = []
    if n >= 2 * pages:
        for i in range(0,2*(pages-1),2):
            texts.append(_texts[i] + _texts[i+1])
        texts.append("".join(_texts[2*(pages-1)-n:]))
    if n >= pages and n < 2 * pages:
        delta = n - pages
        for i in range(0,2*(delta-1),2):
            texts.append(_texts[i] + _texts[i+1])
        for i in range(2*(delta-1),n,1):
            texts.append(_texts[i])
    if n < pages:
        N = n
        for i in range(0,n,1):
            texts.append(_texts[i])
    return [start] + texts + [end]

def generate_video(id):
    generate_assets(id)
    generate_readme(id)
    s = get_texts(id)
    N = len(os.listdir('./'+id+'/assets/'))
    texts = get_text_seq(s,N)
    
    for idx,text in enumerate(texts):
        get_poster(text,id,idx)
        asyncio.run(gen_voice(text,idx,id))

    with open('./'+id+'/poster/0.jpg', 'rb') as f:
        file_content = f.read()
    r.set('bilibili:'+id+':cover.jpg',file_content)

    image_folder = './'+id+'/poster'
    audio_folder = './'+id+'/audio'
    image_files = os.listdir(image_folder)
    audio_files = os.listdir(audio_folder)
    image_files.sort(key=lambda x:int(x[:-4]))
    audio_files.sort(key=lambda x:int(x[:-4]))
    audio_clips = concatenate_audioclips([AudioFileClip(os.path.join(audio_folder,c)) for c in audio_files])
    image_clips = []
    for idx, image in enumerate(image_files[:N]):
        duration = get_time_count(os.path.join(audio_folder, audio_files[idx]))
        _image = ImageClip(os.path.join(image_folder, image)).set_duration(duration)
        image_clips.append(_image)
    video = concatenate_videoclips(image_clips)
    final_video = video.set_audio(audio_clips)
    audio_clips.write_audiofile('./'+id+'/'+id+'.mp3')
    with open('./'+id+'/'+id+'.mp3', 'rb') as f:
        file_content = f.read()
    r.set('bilibili:'+id+':'+id+".mp3",file_content)
    print('...generate radio done...')
    final_video.write_videofile('./'+id+'/'+id+'.mp4',codec='libx264',fps=24)
    with open('./'+id+'/'+id+'.mp4', 'rb') as f:
        file_content = f.read()
    r.set('bilibili:'+id+':'+id+".mp4",file_content)
    r.rpush('cached_ids',id)
    print('...generate video done...')
    set_status(id)

def generate_assets(id):
    download_pdf(id)
    print('...download...')
    gen_assets(id)
    print('...assets...')

def get_texts(id):
    f = codecs.open('./'+id+'/readme.txt','r',"utf-8")
    data = f.read()
    rex = r'<describe>(.*?)</describe>'
    texts = re.findall(rex,data)[0]
    return texts

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
    # ids = get_today_list()        
    # print(ids)
    ids = [b'2311.10642', b'2311.10678', b'2311.10123', b'2311.10709', b'2311.10702', b'2311.10538', b'2311.10775', b'2311.11501', b'2311.12015', b'2311.12022', b'2311.11045', b'2311.10770', b'2311.10751', b'2311.10768', b'2311.10751', b'2311.11243', b'2311.10768', b'2311.11077']
    for id in ids:
        # r.rpush('paper',id)
        id = id.decode("utf-8")
        try:
            generate_video(id)
        except:
            print('exception')
