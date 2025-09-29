import os
import shutil
import json
import pandas as pd
import duckdb
import requests
from bs4 import BeautifulSoup
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

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_LLM_AVAILABLE = True
except ImportError:
    GOOGLE_LLM_AVAILABLE = False

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Directories
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
    except Exception as e:
        print(f"Failed to initialize Google LLM: {e}")
        llm = DummyLLM()
else:
    print("Google credentials not found. Using DummyLLM.")
    llm = DummyLLM()

class FileUploadService:

    @staticmethod
    def reset_environment():
        try:
            for directory in [UPLOAD_DIR, DB_DIR]:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)
            con = duckdb.connect(DUCKDB_PATH)
            con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME}(id INTEGER, content TEXT, embedding BLOB);")
            con.close()
        except Exception as e:
            print(f"Environment reset failed: {e}")

    @staticmethod
    async def save_files(files: List[UploadFile]) -> Dict[str, str]:
        saved_files = {}
        try:
            for file in files:
                file_path = os.path.join(UPLOAD_DIR, file.filename) # type: ignore
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                saved_files[file.filename] = file_path
                try:
                    FileUploadService.store_file_embedding(file_path)
                except Exception as e:
                    print(f"Failed to store embedding for {file.filename}: {e}")
        except Exception as e:
            print(f"Failed to save files: {e}")
        return saved_files

    @staticmethod
    def store_file_embedding(file_path: str):
        try:
            ext = Path(file_path).suffix.lower()
            text_content = ""
            if ext == ".csv":
                df = pd.read_csv(file_path)
                text_content = df.to_json(orient="records")
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                text_content = df.to_json(orient="records")
            elif ext in [".txt"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            elif ext in [".png", ".jpg", ".jpeg"]:
                text_content = f"[Image file: {os.path.basename(file_path)}]"
            else:
                text_content = f"[File {file_path}]"
            embedding = embedding_model.embed_query(text_content)
            con = duckdb.connect(DUCKDB_PATH)
            con.execute(
                f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?)",
                [np.random.randint(1e9), text_content, json.dumps(embedding)] # type: ignore
            )
            con.close()
        except Exception as e:
            print(f"Failed to store embedding for {file_path}: {e}")

    @staticmethod
    def retrieve_context(query: str, top_k: int = 3) -> List[str]:
        try:
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
                except Exception as e:
                    print(f"Failed to compute similarity score: {e}")
            scored.sort(reverse=True, key=lambda x: x[0])
            return [c for _, c in scored[:top_k]]
        except Exception as e:
            print(f"Context retrieval failed: {e}")
            return []

    @staticmethod
    def scrape_url(url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "meta", "noscript"]):
                tag.decompose()
            for a_tag in soup.find_all("a"):
                a_tag.replace_with(a_tag.get_text()) # type: ignore
            return soup.get_text(" ", strip=True)
        except Exception as e:
            print(f"URL scraping failed for {url}: {e}")
            return "null"

    @staticmethod
    def analyze_csv_or_excel(file_path: str, columns: List[str] = None, operation: str = "head") -> str: # type: ignore
        try:
            ext = Path(file_path).suffix.lower()
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                return "null"
            if columns:
                columns = [c for c in columns if c in df.columns]
            if operation == "scatter" and columns and len(columns) >= 2:
                x_col, y_col = columns[0], columns[1]
                buf = io.BytesIO()
                plt.figure(figsize=(6, 4))
                plt.scatter(df[x_col], df[y_col])
                z = np.polyfit(df[x_col], df[y_col], 1)
                p = np.poly1d(z)
                plt.plot(df[x_col], p(df[x_col]), "r--")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"{x_col} vs {y_col} Scatterplot")
                plt.tight_layout()
                plt.savefig(buf, format="png")
                plt.close()
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                return f"data:image/png;base64,{img_base64}"
            if columns:
                return df[columns].head().to_string(index=False)
            return df.head().to_string(index=False)
        except Exception as e:
            print(f"CSV/XLSX analysis failed for {file_path}: {e}")
            return "null"

    @staticmethod
    def encode_image(file_path: str, max_bytes: int = 100_000) -> str:
        try:
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
        except Exception as e:
            print(f"Image encoding failed for {file_path}: {e}")
            return "null"

    @staticmethod
    def process_questions(saved_files: Dict[str, str]) -> List[Dict[str, str]]:
        results = []
        try:
            if "questions.txt" not in saved_files:
                print("questions.txt is missing")
                return results

            with open(saved_files["questions.txt"], "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            questions = lines[lines.index("Answer:") + 1:] if "Answer:" in lines else lines

            for q in questions:
                try:
                    answer = "null"
                    if q.startswith("http://") or q.startswith("https://"):
                        answer = FileUploadService.scrape_url(q)
                    elif any(q.endswith(ext) for ext in [".csv", ".xlsx", ".xls"]):
                        file_name = q.split()[0]
                        if file_name in saved_files:
                            answer = FileUploadService.analyze_csv_or_excel(saved_files[file_name])
                    elif any(q.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
                        file_name = q.split()[0]
                        if file_name in saved_files:
                            answer = FileUploadService.encode_image(saved_files[file_name])
                    else:
                        context = FileUploadService.retrieve_context(q)
                        prompt = f"Context:\n{json.dumps(context)}\n\nQuestion: {q}\nAnswer:"
                        try:
                            answer = llm.invoke(prompt).content  # type: ignore
                        except Exception as e:
                            print(f"LLM failed for question '{q}': {e}")
                            answer = "null"

                    if isinstance(answer, list):
                        answer = "\n".join(answer)  # type: ignore
                    results.append({"question": q, "answer": answer})
                except Exception as e:
                    print(f"Failed to process question '{q}': {e}")
                    results.append({"question": q, "answer": "null"})
        except Exception as e:
            print(f"Processing questions failed: {e}")

        return results
