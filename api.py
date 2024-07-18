import asyncio
from fastapi import APIRouter, UploadFile, File, Form, FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Union, Any
import json
from langchain.text_splitter import CharacterTextSplitter
from threading import Thread
import requests
import re
import ast
import logging
import sqlite3
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLite setup
DB_PATH = os.path.join('/data', 'summarizer.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def initialize_db():
    # Ensure the directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT UNIQUE,
                result TEXT,
                topic TEXT,
                content TEXT,
                status INTEGER,
                msg TEXT
            )
        ''')
        conn.commit()

initialize_db()

# Text splitter setup
text_splitter = CharacterTextSplitter(
    separator="。",
    chunk_size=5000,
    chunk_overlap=0,
    length_function=len,
)

prompt_1 = """针对<正文>内容，撰写总结摘要。
要求：
 - 提取正文的主题
 - 理解核心内容，并重新组织语言形成摘要
 - 在摘要内容中，使用序号罗列要点
 - 使用第三人称
格式：
    会议主题：<>
    会议要点：
        1. <>
        2. <>
        ...
正文：
 {text}
摘要内容："""

prompt_2 = """请根据<现有摘要>和<补充内容>撰写摘要。
下面是现有摘要：{existing_answer}
请根据<补充内容>完善现有摘要，形成一份新的摘要。
请注意，新的摘要也要提供会议主题，并使用序号罗列要点。
补充内容如下：
------------
{text}
------------
如果上面的补充内容对撰写摘要没有帮助，则直接返回现有摘要。"""

class RetrieveRequest(BaseModel):
    get_type: str = None
    meeting_ids: Any = None

def call_api(messages):
    url = "http://localhost:8001/v1/chat/completions"
    data = {
        "model": "qwen2_7b_instruct",
        "messages": messages,
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def save_summary_to_db(meeting_id, result, topic, content):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO summaries (meeting_id, result, topic, content, status, msg) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (meeting_id, json.dumps(result), topic, content, 200, "succeed"))
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving summary to database: {e}")

def async_summarizer(text_list, m_id, _revoke_url):
    l = len(text_list) - 1
    logger.info(f"m_id: {m_id}")
    answer = ''
    for i, line in enumerate(text_list):
        if answer == '':
            text_with_prompt = line
        else:
            text_with_prompt = line + "\n" + answer
        response = call_api([
            {"role": "user", "content": text_with_prompt},
            {"role": "system", "content": "不要有重复内容，同时尽可能准确，不要出现无关内容，序号标对"}
        ])
        answer = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        if i == l:
            res = answer
    result = {}
    try:
        topic = res.split('\n')[0]
        content = '\n'.join(res.split('\n')[1:])
    except:
        topic = ''
        content = res
    tmp_topic = re.split(r'[:：]', topic)
    if len(tmp_topic) > 1:
        topic = '：'.join(re.split(r'[:：]', topic)[1:])
    result['result'] = f"{res}"
    result['topic'] = topic
    result['content'] = content
    result['msg'] = "succeed"
    result['status'] = 200
    result['meeting_id'] = m_id
    logger.info(f"Result: {result}")
    if _revoke_url:
        requests.post(url=_revoke_url, json=jsonable_encoder(result))
    save_summary_to_db(m_id, result, topic, content)
    return result

async def async_task(text_list, m_id, _revoke_url):
    result = await asyncio.to_thread(async_summarizer, text_list, m_id, _revoke_url)
    return result

router = APIRouter()

@router.post("/v2/summarizer")
async def summarizer_server_v2(texts=Form(default=None), 
                               args=Form(default=None), 
                               file: Union[UploadFile, None] = None, 
                               revoke_url=Form(default=None),
                               meeting_id=Form(default=None),
                               async_request=Form(default=None)):
    logger.info(f"Received request with texts: {texts}, meeting_id: {meeting_id}, async_request: {async_request}")
    m_id = meeting_id
    _revoke_url = revoke_url
    if file:
        request = file.file.read()
        request = request.decode()
        logger.info(f"Plain text request: {request}")
        request = request.replace('\n', '\\n')
        request = request.replace('\/', r"/")
        request = ast.literal_eval(request)
        logger.info(f"Parsed request: {request}")
        texts = request.get("texts")
        m_id = request.get("meeting_id", m_id)
        _revoke_url = request.get("revoke_url", _revoke_url)
    elif texts:
        texts = texts.replace('<br>', '\n')
        logger.info(f"Texts: {texts}")
        m_id = meeting_id
    if not _revoke_url:
        _revoke_url = revoke_url

    logger.info(f"Final texts: {texts}, meeting_id: {m_id}, revoke_url: {_revoke_url}")

    text_list = text_splitter.split_text(texts)
    if async_request:
        t = Thread(target=async_summarizer, args=(text_list, m_id, _revoke_url))
        t.start()
        return {"meeting_id": m_id}
    else:
        result = await async_task(text_list, m_id, _revoke_url)
        return result

@router.post("/v2/retrieveSummarizer")
async def retrieve_summarizer_v2(request: RetrieveRequest):
    logger.info(f"Received retrieve request: {request}")
    meeting_ids = json.loads(request.meeting_ids)
    response = [] 
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for mid in meeting_ids:
                cursor.execute('SELECT result FROM summaries WHERE meeting_id = ?', (mid,))
                row = cursor.fetchone()
                if row:
                    resp = json.loads(row[0])
                    response.append(resp)
                    cursor.execute('DELETE FROM summaries WHERE meeting_id = ?', (mid,))
                else:
                    response.append({"meeting_id": mid, "result": None, "stop": True})
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
    logger.info(f"Retrieve response: {response}")
    return jsonable_encoder(response)
port = 8000

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

def start_server():
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        workers=1
    )

if __name__ == "__main__":
    start_server()
