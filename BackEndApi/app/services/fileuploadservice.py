import os
import shutil
import pandas as pd
import duckdb
from pathlib import Path
from fastapi import UploadFile
from typing import List, Dict, Any
from dotenv import load_dotenv
import io
import base64
from PIL import Image
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup
import json
import re
import time
import random
import matplotlib.pyplot as plt

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_LLM_AVAILABLE = True
except ImportError:
    GOOGLE_LLM_AVAILABLE = False

# Load environment
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

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

    @staticmethod
    def reset_environment():
        for directory in [UPLOAD_DIR, DB_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
        con = duckdb.connect(DUCKDB_PATH)
        try:
            con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} (id BIGINT, metadata TEXT, content TEXT, embedding TEXT);")
            con.execute(f"DELETE FROM {TABLE_NAME};")
        finally:
            con.close()

    @staticmethod
    def _new_id() -> int:
        return int(time.time()*1000) ^ random.getrandbits(32)

    @staticmethod
    def _store_embedding_row(id_: int, metadata: Dict[str, Any], content: str, embedding: List[float]):
        con = duckdb.connect(DUCKDB_PATH)
        try:
            con.execute(f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?, ?)", 
                        [id_, json.dumps(metadata, default=str), content[:10000], json.dumps(embedding)])
        finally:
            con.close()

    @staticmethod
    def encode_image_to_datauri(file_path: str, max_bytes: int = 100_000) -> str:
        try:
            img = Image.open(file_path).convert("RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            if len(img_bytes) > max_bytes:
                factor = (max_bytes / len(img_bytes)) ** 0.5
                img = img.resize((max(1, int(img.width * factor)), max(1, int(img.height * factor))), Image.LANCZOS) # type: ignore
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            encoded = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            return f"Error encoding image: {e}"

    @staticmethod
    def _embed_table_file(file_path: str, filename: str):
        try:
            ext = Path(file_path).suffix.lower()
            if ext == ".csv":
                df = pd.read_csv(file_path, dtype=str).fillna("")
                dfs = {"Sheet1": df}
            else:
                dfs = pd.read_excel(file_path, sheet_name=None, dtype=str)
                dfs = {k: v.fillna("") for k, v in dfs.items()}
            
            for sheet_name, df in dfs.items():
                for idx, row in df.iterrows():
                    row_dict = row.to_dict()
                    content_text = " | ".join([f"{k}: {v}" for k, v in row_dict.items()])
                    metadata = {"source": filename, "sheet": sheet_name, "row_index": int(idx), "columns": list(df.columns), "values": row_dict} # type: ignore
                    try:
                        embedding = embedding_model.embed_query(content_text)
                    except:
                        embedding = [0.0]
                    FileUploadService._store_embedding_row(FileUploadService._new_id(), metadata, content_text, embedding)
        except Exception as e:
            print(f"Error embedding table file {filename}: {e}")

    @staticmethod
    def _embed_image_file(file_path: str, filename: str):
        data_uri = FileUploadService.encode_image_to_datauri(file_path)
        metadata = {"source": filename, "type": "image"}
        try:
            embedding = embedding_model.embed_query(Path(filename).stem)
        except:
            embedding = [0.0]
        FileUploadService._store_embedding_row(FileUploadService._new_id(), metadata, data_uri, embedding)

    @staticmethod
    def _embed_text_file(file_path: str, filename: str):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except:
            content = ""
        metadata = {"source": filename, "type": "text"}
        try:
            embedding = embedding_model.embed_query(content)
        except:
            embedding = [0.0]
        FileUploadService._store_embedding_row(FileUploadService._new_id(), metadata, content[:10000], embedding)

    @staticmethod
    async def save_files(files: List[UploadFile]) -> Dict[str, str]:
        saved_files = {}
        FileUploadService.reset_environment()
        for file in files:
            try:
                file_path = os.path.join(UPLOAD_DIR, file.filename) # type: ignore
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                saved_files[file.filename.lower()] = file_path # type: ignore
                ext = Path(file_path).suffix.lower()
                if not file.filename.lower().endswith("questions.txt"): # type: ignore
                    if ext in [".csv", ".xlsx", ".xls"]:
                        FileUploadService._embed_table_file(file_path, file.filename) # type: ignore
                    elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
                        FileUploadService._embed_image_file(file_path, file.filename) # type: ignore
                    elif ext == ".txt":
                        FileUploadService._embed_text_file(file_path, file.filename) # type: ignore
            except Exception as e:
                print(f"Error saving file {file.filename}: {e}")
        return saved_files

    @staticmethod
    def store_url_content(url: str):
        try:
            res = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            text_content = soup.get_text(separator=" ", strip=True)
            if text_content:
                metadata = {"source": url, "type": "url"}
                embedding = embedding_model.embed_query(text_content)
                FileUploadService._store_embedding_row(FileUploadService._new_id(), metadata, text_content[:10000], embedding)
        except Exception as e:
            print(f"Error scraping URL {url}: {e}")

    @staticmethod
    def retrieve_similar_rows(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            q_emb = embedding_model.embed_query(query)
        except:
            q_emb = [0.0]
        con = duckdb.connect(DUCKDB_PATH)
        try:
            rows = con.execute(f"SELECT id, metadata, content, embedding FROM {TABLE_NAME}").fetchall()
        finally:
            con.close()
        scored = []
        for id_, metadata_json, content, embedding_json in rows:
            try:
                emb = np.array(json.loads(embedding_json), dtype=float)
                qv = np.array(q_emb, dtype=float)
                score = float(np.dot(qv, emb) / (np.linalg.norm(qv)*np.linalg.norm(emb))) if np.linalg.norm(qv)*np.linalg.norm(emb) != 0 else 0.0
            except:
                score = 0.0
            try:
                metadata = json.loads(metadata_json)
            except:
                metadata = {"raw": metadata_json}
            scored.append({"id": int(id_), "metadata": metadata, "content": content, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _clean_llm_csv_text(text: str) -> str:
        """
        Extract CSV-like content from LLM response, ignoring extra text/noise.
        """
        lines = text.splitlines()
        csv_lines = [l for l in lines if ',' in l or re.match(r'^[\w\s]+$', l)]
        return "\n".join(csv_lines) if csv_lines else text

    @staticmethod
    def process_questions(saved_files: Dict[str, str]) -> List[List[Any]]:
        results = []
        try:
            q_file = next((f for f in saved_files if f.lower().endswith("questions.txt")), None)
            if not q_file:
                raise RuntimeError("No 'questions.txt' file found")
            with open(saved_files[q_file], "r", encoding="utf-8") as f:
                lines = [line.rstrip() for line in f]

            answer_idx = next((i for i, l in enumerate(lines) if "answer" in l.lower()), None)
            if answer_idx is None:
                raise RuntimeError("No line containing 'answer' found")

            question_lines = [l for l in lines[answer_idx + 1:] if l.strip()]
            questions = []
            current_q = ""
            for line in question_lines:
                if re.match(r"^\d+\.", line):
                    if current_q:
                        questions.append(current_q.strip())
                    current_q = line
                else:
                    current_q += " " + line
            if current_q:
                questions.append(current_q.strip())

            for idx, q in enumerate(questions, start=1):
                try:
                    retrieved = FileUploadService.retrieve_similar_rows(q, top_k=5)
                    context_text = "\n".join([f"- {r['content']}" for r in retrieved]) if retrieved else "No relevant context found."
                    prompt = f"""
You are a smart AI assistant. Use the context below to answer the question.
Context (from vector DB):
{context_text}

Question:
{q}

Instructions:
- Provide a precise answer.
- If a chart is requested, provide numeric CSV/text data suitable for plotting.
"""
                    ans_text = ""
                    try:
                        ans_text = llm.invoke(prompt).content.strip()  # type: ignore
                        if not ans_text:
                            ans_text = "No answer generated."
                    except Exception as e:
                        ans_text = f"Error generating answer: {e}"

                    # Check for chart requests
                    if re.search(r"(scatter|histogram|line|bar|chart|image|diagram)", q, re.I):
                        try:
                            cleaned_text = FileUploadService._clean_llm_csv_text(ans_text)
                            df = pd.read_csv(io.StringIO(cleaned_text))
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            fig = None

                            if "scatter" in q.lower() and len(numeric_cols) >= 2:
                                corr_matrix = df[numeric_cols].corr().abs()
                                np.fill_diagonal(corr_matrix.values, 0)
                                col_pair = corr_matrix.stack().idxmax()
                                x_col, y_col = col_pair # type: ignore
                                plt.close('all')
                                fig, ax = plt.subplots()
                                ax.scatter(df[x_col], df[y_col])
                                ax.set_xlabel(x_col) # type: ignore
                                ax.set_ylabel(y_col) # type: ignore
                                ax.set_title("Scatter plot with highest correlation")
                            elif "histogram" in q.lower() and numeric_cols:
                                plt.close('all')
                                fig, ax = plt.subplots()
                                df[numeric_cols].hist(ax=ax)
                            elif "line" in q.lower() and numeric_cols:
                                plt.close('all')
                                fig, ax = plt.subplots()
                                df[numeric_cols].plot.line(ax=ax)
                            elif "bar" in q.lower() and numeric_cols:
                                plt.close('all')
                                fig, ax = plt.subplots()
                                df[numeric_cols].plot.bar(ax=ax)

                            # Save figure to base64
                            if fig is not None:
                                buf = io.BytesIO()
                                plt.tight_layout()
                                plt.savefig(buf, format='png')
                                plt.close(fig)
                                buf.seek(0)
                                img_bytes = buf.getvalue()
                                encoded = base64.b64encode(img_bytes).decode('utf-8')
                                ans_text = f"data:image/png;base64,{encoded}"
                        except Exception as e:
                            ans_text += f"\nError generating plot: {e}"

                    score = retrieved[0]["score"] if retrieved else 0.0
                    results.append([idx, q, score, ans_text])
                except Exception as e:
                    results.append([idx, q, 0.0, f"Error processing: {e}"])
        except Exception as e:
            print(f"Error processing questions: {e}")
        return results
