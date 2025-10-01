import os
import shutil
import pandas as pd
import duckdb
import requests
from pathlib import Path
from fastapi import UploadFile
from typing import List, Dict
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
import re
import json

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_LLM_AVAILABLE = True
except ImportError:
    GOOGLE_LLM_AVAILABLE = False

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

BASE_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_DIR = os.path.join(BASE_DIR, "database")
DUCKDB_PATH = os.path.join(DB_DIR, "vectordb.duckdb")
TABLE_NAME = "vectordb_data"

class DummyLLM:
    def invoke(self, prompt: str):
        return type("Result", (object,), {"content": "Dummy Answer"})()

if GOOGLE_LLM_AVAILABLE and GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    except Exception:
        llm = DummyLLM()
else:
    llm = DummyLLM()

class FileUploadService:
    url_cache: Dict[str, str] = {}

    @staticmethod
    def reset_environment():
        for directory in [UPLOAD_DIR, DB_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
        con = duckdb.connect(DUCKDB_PATH)
        con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} (id INTEGER, content TEXT, embedding BLOB);")
        con.close()

    @staticmethod
    async def save_files(files: List[UploadFile]) -> Dict[str, str]:
        saved_files = {}
        FileUploadService.reset_environment()
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename) # type: ignore
            with open(file_path, "wb") as f:
                f.write(await file.read())
            saved_files[file.filename.lower()] = file_path # type: ignore
            FileUploadService.store_file_embedding(file_path)
        return saved_files

    @staticmethod
    def store_file_embedding(file_path: str):
        ext = Path(file_path).suffix.lower()
        if ext in [".csv", ".xlsx", ".xls"]:
            df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
            text_content = df.to_json(orient="records")
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        elif ext in [".png", ".jpg", ".jpeg"]:
            text_content = FileUploadService.encode_image(file_path)
        else:
            text_content = f"[File {file_path}]"
        embedding = embedding_model.embed_query(text_content)
        con = duckdb.connect(DUCKDB_PATH)
        con.execute(f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?)", [np.random.randint(1e9), text_content, json.dumps(embedding)]) # type: ignore
        con.close()

    @staticmethod
    def retrieve_context(query: str, top_k: int = 3) -> str:
        query_embedding = embedding_model.embed_query(query)
        con = duckdb.connect(DUCKDB_PATH)
        rows = con.execute(f"SELECT content, embedding FROM {TABLE_NAME}").fetchall()
        con.close()
        scored = []
        for content, embedding_json in rows:
            try:
                embedding = np.array(json.loads(embedding_json))
                score = float(np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding)))
                scored.append((score, content))
            except:
                continue
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1] if scored else "No context found"

    @staticmethod
    def encode_image(file_path: str, max_bytes: int = 100_000) -> str:
        img = Image.open(file_path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        if len(img_bytes) > max_bytes:
            factor = np.sqrt(max_bytes / len(img_bytes))
            img = img.resize((int(img.width * factor), int(img.height * factor)), Image.LANCZOS) # type: ignore
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def fetch_url_content(url: str) -> str:
        if url in FileUploadService.url_cache:
            return FileUploadService.url_cache[url]
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n")  # Only current page- improvement required
            FileUploadService.url_cache[url] = text
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to fetch URL '{url}': {e}")

    @staticmethod
    def process_questions(saved_files: Dict[str, str]) -> List[str]:
        results = []
        q_file = next((f for f in saved_files if f.lower().endswith("questions.txt")), None)
        if not q_file:
            raise RuntimeError("No 'questions.txt' file found in uploaded files")

        with open(saved_files[q_file], "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        questions = []
        current_q = ""
        for line in lines:
            if re.match(r"^\d+\.", line):
                if current_q:
                    questions.append(current_q.strip())
                current_q = line
            else:
                current_q += " " + line
        if current_q:
            questions.append(current_q.strip())

        url_pattern = re.compile(r"https?://\S+")

        for q in questions:
            answer = "null"
            try:
                urls = url_pattern.findall(q)
                if urls:
                    content = "\n".join(FileUploadService.fetch_url_content(u) for u in urls)
                    answer = llm.invoke(f"Context:\n{content}\n\nQuestion: {q}\nAnswer:").content # type: ignore
                elif any(ext in q.lower() for ext in [".csv", ".xls", ".xlsx", ".png", ".jpg", ".jpeg"]):
                    for f_name, path in saved_files.items():
                        suffix = Path(f_name).suffix.lower()
                        if suffix in [".csv", ".xls", ".xlsx"]:
                            df = pd.read_csv(path) if suffix == ".csv" else pd.read_excel(path)
                            content = df.to_json(orient="records")
                            answer = llm.invoke(f"Context:\n{content}\n\nQuestion: {q}\nAnswer:").content # type: ignore
                            break
                        elif suffix in [".png", ".jpg", ".jpeg"]:
                            content = FileUploadService.encode_image(path)
                            answer = llm.invoke(f"Context:\n{content}\n\nQuestion: {q}\nAnswer:").content # type: ignore
                            break
                else:
                    context = FileUploadService.retrieve_context(q)
                    answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {q}\nAnswer:").content # type: ignore
            except Exception as e:
                answer = f"Error processing question: {e}"

            results.append(answer.strip()) # type: ignore

        return results
