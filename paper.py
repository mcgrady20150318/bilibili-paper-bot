import requests
import os
import datetime
import re
import pickle
import redis
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
import time
from pathlib import Path
from openai import OpenAI

redis_url = os.getenv('REDIS_URL')
r = redis.from_url(redis_url)

path = os.getcwd() + '/'

api_key = os.getenv('apikey')
client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
)

VOICE = "zh-CN-YunxiNeural"

slide_prompt = '''
    请基于论文内容，严格按照下面的xml格式生成内容，要求均用英语描述:
    <title>英文标题</title>
    <author>作者姓名，以逗号隔开</author>
    <motivation>100字描述研究动机</motivation>
    <contribution>100字描述研究贡献</contribution>
    <method>100字描述研究方法</method>
    <experiment>100字描述重要实验结果</experiment>
    <conclusion>100字描述结论</conclusion>，生成结果如下：
'''

readme_prompt = '''
    请基于论文内容，严格按照下面的xml格式生成内容:
    <ctitle>这里生成一个吸引读者的中文专业标题，要求有信息量</ctitle>
    <describe>这里生成一段200字左右的中文论文解读</describe>
    <tags>这里生成5个中文标签，并且以空格隔开</tags>，生成结果如下：
'''

read_prompt = '''
    你现在是PaperWeekly的学术主播Ian，请基于本文内容写一份讲解词，依次从研究动机、研究贡献、方法介绍、实验结果、研究结论等5个方面展开，要求前后具有连贯性，要求严格有7段话。
'''

def download_pdf(id):
    url = 'http://arxiv.org/pdf/'+id+'.pdf'
    response = requests.get(url)
    if not os.path.exists('./'+id):
        os.mkdir('./'+id)
        os.mkdir('./'+id+'/assets')
        os.mkdir('./'+id+'/slide')
        os.mkdir('./'+id+'/audio')
        os.mkdir('./'+id+'/video')
        os.mkdir('./'+id+'/poster')
    output_path = './' + id + '/' + id + '.pdf'
    with open(output_path, 'wb') as file:
        file.write(response.content)

def get_content(id):
    filepath = './' + id + '/' + id + '.pdf'
    file_object = client.files.create(file=Path(filepath), purpose="file-extract")
    file_content = client.files.content(file_id=file_object.id).text
    f = open('./' + id + '/paper.txt','w')
    f.write(file_content)
    f.close()
    r.set("bilibili:"+id+":paper.txt",file_content)

def gen_slide_content(id):
    content = open('./' + id + '/paper.txt','r').read()
    if len(content) > 10000:
        content = content[:10000]
    f = codecs.open('./'+id+'/slide.txt','w',"utf-8")
    messages=[
        {"role": "system","content": content},
        {"role": "user", "content": slide_prompt},
    ]
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
        temperature=0.0,
    )
    output = completion.choices[0].message.content
    r.set("bilibili:"+id+":slide.txt",output)
    f.write(output)
    f.close()

def gen_readme_content(id):
    content = open('./' + id + '/paper.txt','r').read()
    if len(content) > 10000:
        content = content[:10000]
    f = codecs.open('./'+id+'/readme.txt','w',"utf-8")
    messages=[
        {"role": "system","content": content},
        {"role": "user", "content": readme_prompt},
    ]
    completion = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=messages,
        temperature=0.0,
    )
    output = completion.choices[0].message.content
    r.set("bilibili:"+id+":readme.txt",output)
    f.write(output)
    f.close()

def gen_read_content(id):
    content = open('./' + id + '/paper.txt','r').read()
    if len(content) > 10000:
        content = content[:10000]
    f = codecs.open('./'+id+'/read.txt','w',"utf-8")
    messages=[
        {"role": "system","content": content},
        {"role": "user", "content": read_prompt},
    ]
    completion = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=messages,
        temperature=0.0,
    )
    output = completion.choices[0].message.content
    r.set("bilibili:"+id+":read.txt",output)
    f.write(output)
    f.close()

def gen_paper_assets(id):
    output_dir = './'+id+'/assets/'
    os.makedirs(output_dir, exist_ok=True)
    pages = pdf2image.convert_from_path('./'+id+'/'+id+'.pdf',dpi=300)
    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f'{i}.jpg')
        page.save(image_path, 'JPEG')

def gen_item(content):
    texts = content.split('.')[:-1]
    result = ''
    for text in texts:
        result += '\item ' + text.lstrip() + '.' + '\n'
    return result

def gen_slide(id):
    f = open('./'+id+'/slide.txt','r')
    data = f.read()
    if '</conclusion>' not in data:
        data += '</conclusion>'
    rex = r'<title>(.*?)</title>'
    title = re.findall(rex,data)[0]
    rex = r'<author>(.*?)</author>'
    author = re.findall(rex,data)[0]
    rex = r'<motivation>(.*?)</motivation>'
    motivation = re.findall(rex,data)[0]
    rex = r'<contribution>(.*?)</contribution>'
    contribution = re.findall(rex,data)[0]
    rex = r'<method>(.*?)</method>'
    method = re.findall(rex,data)[0]
    rex = r'<experiment>(.*?)</experiment>'
    experiment = re.findall(rex,data)[0]
    rex = r'<conclusion>(.*?)</conclusion>'
    conclusion = re.findall(rex,data)[0]
    f.close()
    f = open('template.tex','r')
    data = f.read()
    data = data.replace('TITLE',title)
    data = data.replace('AUTHOR',author)
    data = data.replace('MOTIVATION',gen_item(motivation))
    data = data.replace('CONTRIBUTION',gen_item(contribution))
    data = data.replace('METHOD',gen_item(method))
    data = data.replace('EXPERIMENT',gen_item(experiment))
    data = data.replace('CONCLUSION',gen_item(conclusion))
    f.close()
    f = open('./'+id+'/slide/main.tex','w')
    r.set("bilibili:"+id+":main.txt",data)
    f.write(data)
    f.close()

def gen_slide_pdf(id):
    os.chdir(path + id+'/slide/')
    os.system('pdflatex main.tex')
    os.system('mv main.pdf ' + path + id)
    # print(os.system('ls ' + path + id))
    os.system('rm *')
    # print(os.system('ls ' + path + id))
    os.chdir(path)

def gen_slide_assets(id):
    output_dir = './'+id+'/slide/'
    os.makedirs(output_dir, exist_ok=True)
    pages = pdf2image.convert_from_path('./'+id+'/main.pdf')
    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f'{i}.jpg')
        page.save(image_path, 'JPEG')
    # print(os.system('ls ' + './'+id+'/slide/'))

# def get_poster(text,id,idx):
#     header = Header(text=text,
#                     text_width=50,
#                     font=Font(path='./FangZhengKaiTi-GBK-1.ttf',size=15),
#                     align='center',
#                     color='#000100',
#                     )
#     content = Content(header=header)
#     img = Image(content,fullpath='./'+id+'/poster/'+str(idx) + '.png')
#     img.draw_on_image('./'+id+'/slide/'+ str(idx) +'.jpg')
#     os.rename('./'+id+'/poster/'+str(idx) + '.png','./'+id+'/poster/'+str(idx) + '.jpg')

def get_time_count(audio_file):
    audio = MP3(audio_file)
    time_count = audio.info.length
    return time_count

async def gen_voice(text,idx,id):
    communicate = edge_tts.Communicate(text, VOICE)  
    await communicate.save('./'+id+'/audio/' + str(idx)+'.mp3')

def get_texts(id):
    texts = open('./'+id+'/read.txt').read().split('\n\n')
    return texts

def generate_video(id):
    download_pdf(id)
    get_content(id)
    gen_readme_content(id)
    gen_read_content(id)
    gen_paper_assets(id)
    gen_slide_content(id)
    gen_slide(id)
    gen_slide_pdf(id)
    gen_slide_assets(id)
    N = len(os.listdir('./'+id+'/slide/'))
    texts = get_texts(id)
    print('len:',N,len(texts))
    if N > len(texts):
        texts = texts + ['感谢观看，欢迎一键三连！']
    else:
        texts = texts[:N] 

    for idx,text in enumerate(texts):
        # get_poster(text,id,idx)
        asyncio.run(gen_voice(text,idx,id))

    with open('./'+id+'/slide/0.jpg', 'rb') as f:
        file_content = f.read()
    r.set('bilibili:'+id+':cover.jpg',file_content)

    image_folder = './'+id+'/slide'
    audio_folder = './'+id+'/audio'
    image_files = os.listdir(image_folder)
    audio_files = os.listdir(audio_folder)
    image_files.sort(key=lambda x:int(x[:-4]))
    audio_files.sort(key=lambda x:int(x[:-4]))
    audio_clips = concatenate_audioclips([AudioFileClip(os.path.join(audio_folder,c)) for c in audio_files])
    image_clips = []
    for idx, image in enumerate(image_files):
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
    ids = ['2311.17922']
    for id in ids:
        # r.rpush('paper',id)
        # try:
        generate_video(id)
            # time.sleep(10)
        # except:
            # print('exception')



