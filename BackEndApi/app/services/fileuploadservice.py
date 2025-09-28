import os
import shutil
import pandas as pd
import duckdb
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from fastapi import HTTPException, UploadFile
from typing import List, Dict
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import base64
#config changes
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_LLM_AVAILABLE = True
except ImportError:
    GOOGLE_LLM_AVAILABLE = False

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Directories
BASE_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_DIR = os.path.join(BASE_DIR, "database")

DUCKDB_PATH = os.path.join(DB_DIR, "vectordb.duckdb")
TABLE_NAME = "vectordb_data"
EMBED_TABLE = "vectordb_embeddings"

class DummyLLM:
    def invoke(self, prompt: str):
        return type("Result", (object,), {"content": f"Dummy answer for prompt: {prompt[:50]}..."})()

if GOOGLE_LLM_AVAILABLE and GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2
    )
else:
    print("Google credentials not found. Using DummyLLM.")
    llm = DummyLLM()

class FileUploadService:
    @staticmethod
    def reset_environment():
        """Clear uploads/ and database/ dirs before processing a new request."""
        for directory in [UPLOAD_DIR, DB_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
        con = duckdb.connect(DUCKDB_PATH)
        con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME}(id INTEGER, data TEXT);")
        con.execute(f"CREATE TABLE IF NOT EXISTS {EMBED_TABLE}(id INTEGER, embedding BLOB);")
        con.close()

    @staticmethod
    async def save_files(files: List[UploadFile]) -> Dict[str, str]:
        """Save uploaded files to disk and return dict {filename: path}"""
        saved_files = {}
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)  # type: ignore
            with open(file_path, "wb") as f:
                f.write(await file.read())
            saved_files[file.filename] = file_path
        return saved_files

    @staticmethod
    def scrape_url(url: str) -> str:
        """Scrape only visible text from a webpage, ignoring href links."""
        headers = { #test headers
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/140.0.0.0 Safari/537.36"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=400, detail=f"Failed to fetch {url} (status {resp.status_code})"
                )
            soup = BeautifulSoup(resp.text, "html.parser")
     
            for tag in soup(["script", "style", "meta", "noscript"]):
                tag.decompose()         
            for a_tag in soup.find_all("a"):
                a_tag.replace_with(a_tag.get_text()) # type:ignore

            text = soup.get_text(" ", strip=True)
            return text
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error fetching {url}: {str(e)}")

    @staticmethod
    def analyze_csv_or_excel(file_path: str, query: str) -> str:
        """Query CSV/XLSX using DuckDB and return results or generate base64 scatterplot."""
        ext = Path(file_path).suffix
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type {ext}")

        if "Rank" in df.columns and "Peak" in df.columns:
            buf = io.BytesIO()
            plt.figure(figsize=(6, 4))
            plt.scatter(df["Rank"], df["Peak"])
   
            z = pd.np.polyfit(df["Rank"], df["Peak"], 1)  # type: ignore
            p = pd.np.poly1d(z)  # type: ignore
            plt.plot(df["Rank"], p(df["Rank"]), "r--")
            plt.xlabel("Rank")  #improvment required with diff fields
            plt.ylabel("Peak")
            plt.title("Rank vs Peak Scatterplot")
            plt.tight_layout()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{img_base64}"
        try:
            result = duckdb.query(query).to_df()
            return result.to_json(orient="records")
        except Exception as e:
            return f"Error running query: {e}"

    @staticmethod
    def process_questions(saved_files: Dict[str, str]) -> List[Dict[str, str]]:
        """Process questions.txt and return list of answers"""
        results = []

        if "questions.txt" not in saved_files:
            raise HTTPException(status_code=400, detail="questions.txt is required")

        with open(saved_files["questions.txt"], "r", encoding="utf-8") as f:
            questions = [q.strip() for q in f.readlines() if q.strip()]

        for q in questions:

            if q.startswith("http://") or q.startswith("https://"):
                context = FileUploadService.scrape_url(q)
                prompt = f"Answer based on scraped content:\n{context}\n\nQuestion: {q}"
                answer = llm.invoke(prompt).content  # type: ignore
                results.append({"question": q, "answer": answer})

            elif q.endswith(".csv") or q.endswith(".xlsx"):
                if q not in saved_files:
                    results.append({"question": q, "answer": f"File {q} not uploaded"})
                    continue
                file_path = saved_files[q]
                query = f"SELECT * FROM read_csv_auto('{file_path}') LIMIT 5;"  
                answer = FileUploadService.analyze_csv_or_excel(file_path, query)
                results.append({"question": q, "answer": answer})

            # Normal text question
            else:
                prompt = f"Answer the question clearly:\n\n{q}"
                answer = llm.invoke(prompt).content  # type: ignore
                results.append({"question": q, "answer": answer})

        return results
