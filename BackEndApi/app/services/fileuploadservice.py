import os
import shutil
import pandas as pd
import duckdb
from pathlib import Path
from fastapi import UploadFile
from typing import List, Dict, Any, Optional
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
import logging

# optional Google LLM wrapper
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_LLM_AVAILABLE = True
except ImportError:
    GOOGLE_LLM_AVAILABLE = False

#env setup- from .enb
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    CHUNK_SIZE = 3000 #improvement required
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
    def _chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks

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
            logger.exception("Error encoding image")
            return f"Error encoding image: {e}"

    @staticmethod
    def _read_small_preview(file_path: str, ext: str, max_chars: int = 400) -> str:
        """Return a short preview string for a file to present to LLM for decision-making."""
        try:
            if ext in [".txt"]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read(max_chars)
            elif ext == ".csv":
                df = pd.read_csv(file_path, nrows=5, dtype=str).fillna("")
                return df.to_csv(index=False)[:max_chars]
            elif ext in [".xls", ".xlsx"]:
                dfs = pd.read_excel(file_path, sheet_name=0, nrows=5, dtype=str).fillna("")
                return pd.DataFrame(dfs).to_csv(index=False)[:max_chars]
            elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:              
                stat = os.path.getsize(file_path)
                return f"<image file: {os.path.basename(file_path)}, size: {stat} bytes>"
            else:
                return f"<unknown file type: {ext}>"
        except Exception as e:
            logger.exception("Preview read failed for %s", file_path)
            return f"<error reading preview: {e}>"

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
                    except Exception:
                        logger.exception("Embedding table row failed")
                        embedding = [0.0]
                    FileUploadService._store_embedding_row(FileUploadService._new_id(), metadata, content_text, embedding)
        except Exception as e:
            logger.exception("Error embedding table file %s", filename)

    @staticmethod
    def _embed_image_file(file_path: str, filename: str):
        data_uri = FileUploadService.encode_image_to_datauri(file_path)
        metadata = {"source": filename, "type": "image"}
        try:
            embedding = embedding_model.embed_query(Path(filename).stem)
        except Exception:
            logger.exception("Image embedding failed")
            embedding = [0.0]
        FileUploadService._store_embedding_row(FileUploadService._new_id(), metadata, data_uri, embedding)

    @staticmethod
    def _embed_text_file(file_path: str, filename: str):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            logger.exception("Read text file failed")
            content = ""
        metadata = {"source": filename, "type": "text"}
        try:
            chunks = FileUploadService._chunk_text(content, FileUploadService.CHUNK_SIZE)
            if not chunks:
                embedding = embedding_model.embed_query("")
                FileUploadService._store_embedding_row(FileUploadService._new_id(), metadata, "", embedding)
            else:
                for i, chunk in enumerate(chunks):
                    m = metadata.copy()
                    m["chunk_index"] = i # type: ignore
                    m["chunk_size"] = len(chunk) # type: ignore
                    try:
                        embedding = embedding_model.embed_query(chunk) if chunk.strip() else [0.0]
                    except Exception:
                        logger.exception("Chunk embedding failed")
                        embedding = [0.0]
                    FileUploadService._store_embedding_row(FileUploadService._new_id(), m, chunk[:10000], embedding)
        except Exception:
            logger.exception("Embedding text file failed")
            embedding = [0.0]
            FileUploadService._store_embedding_row(FileUploadService._new_id(), metadata, content[:10000], embedding)

    @staticmethod
    async def save_files(files: List[UploadFile]) -> Dict[str, str]: # type: ignore
        """
        Save uploaded files to disk. Do NOT embed here if questions.txt exists and contains URLs.
        Return dict: { filename_lower: file_path }
        """
        saved_files: Dict[str, str] = {}
        FileUploadService.reset_environment()
        for file in files:
            try:
                file_path = os.path.join(UPLOAD_DIR, file.filename)  # type: ignore
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                saved_files[file.filename.lower()] = file_path  # type: ignore
            except Exception:
                logger.exception("Error saving file %s", getattr(file, "filename", "<unknown>"))
        return saved_files

    @staticmethod
    def store_url_content(url: str):
        """Fetch only the exact page at URL and embed its text into DuckDB."""
        try:
            res = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            text_content = soup.get_text(separator=" ", strip=True)
            if text_content:
                metadata = {"source": url, "type": "url"}
                try:
                    embedding = embedding_model.embed_query(text_content)
                except Exception:
                    logger.exception("Embedding URL content failed")
                    embedding = [0.0]
                FileUploadService._store_embedding_row(FileUploadService._new_id(), metadata, text_content[:10000], embedding)
        except Exception as e:
            logger.exception("Error scraping URL %s: %s", url, e)

    @staticmethod
    def _detect_urls_in_text(text: str) -> List[str]:
        #verify url
        urls = re.findall(r"https?://[^\s,;]+", text)
        return urls

    @staticmethod
    def _llm_choose_files(saved_files: Dict[str, str]) -> List[str]:
        """
        Ask the LLM to choose which uploaded files to embed.
        LLM must return a JSON array of filenames to embed (lowercased).
        If LLM fails or returns invalid, fallback to embedding all files.
        """
        try:
           
            inventory = []
            for name, path in saved_files.items():
                if name.endswith("questions.txt"):
                    continue
                ext = Path(path).suffix.lower()
                preview = FileUploadService._read_small_preview(path, ext)
                inventory.append({"filename": name, "ext": ext, "preview": preview})

            prompt = {
                "instruction": (
                    "You are given a list of uploaded files (filename, extension, and a brief preview).\n"
                    "Decide which files are relevant to answer questions that will follow (some files might be images, tables, or text).\n"
                    "Return a JSON array with filenames to embed (use the exact filename key value). "
                    "If many are relevant, include them all. If none are relevant, return an empty array.\n\n"
                    "FILES:\n"
                ),
                "files": inventory
            }
            # Construct textual prompt
            prompt_text = prompt["instruction"] + json.dumps(inventory, ensure_ascii=False, indent=0)
            raw_response = ""
            try:
                resp = llm.invoke(prompt_text)  # type: ignore
                raw_response = getattr(resp, "content", "") or str(resp)
            except Exception:
                logger.exception("LLM choose files invocation failed")
                raw_response = ""

          
            json_match = re.search(r"(\[.*\])", raw_response, re.DOTALL)
            if json_match:
                arr = json.loads(json_match.group(1))
               
                chosen = [a.lower() for a in arr if isinstance(a, str)]
               
                valid_chosen = [c for c in chosen if c in saved_files and not c.endswith("questions.txt")]
                if valid_chosen:
                    logger.info("LLM selected files: %s", valid_chosen)
                    return valid_chosen
                else:
                    logger.info("LLM returned list but no valid matches; falling back to all files")
            else:
              
                lines = [ln.strip().lower() for ln in re.split(r"[,\n]+", raw_response) if ln.strip()]
                valid = [l for l in lines if l in saved_files and not l.endswith("questions.txt")]
                if valid:
                    logger.info("LLM returned newline/comma list: %s", valid)
                    return valid

          
            fallback = [n for n in saved_files.keys() if not n.endswith("questions.txt")]
            logger.info("Fallback embedding all files: %s", fallback)
            return fallback
        except Exception:
            logger.exception("Error in _llm_choose_files; falling back to all files")
            return [n for n in saved_files.keys() if not n.endswith("questions.txt")]

    @staticmethod
    def _embed_file_by_path(file_path: str, filename: str):
        """Choose correct embedding function based on extension."""
        ext = Path(file_path).suffix.lower()
        if ext in [".csv", ".xlsx", ".xls"]:
            FileUploadService._embed_table_file(file_path, filename)
        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
            FileUploadService._embed_image_file(file_path, filename)
        elif ext == ".txt":
            FileUploadService._embed_text_file(file_path, filename)
        else:
            # for unknowns, try as text
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                content = ""
            try:
                embedding = embedding_model.embed_query(content)
            except Exception:
                embedding = [0.0]
            metadata = {"source": filename, "type": ext or "unknown"}
            FileUploadService._store_embedding_row(FileUploadService._new_id(), metadata, content[:10000], embedding)

    @staticmethod
    async def save_and_maybe_embed(files: List[UploadFile]) -> Dict[str, str]:
        """
        Save uploaded files. Do not auto-embed everything here.
        Returns saved_files dict.
        """
        return await FileUploadService.save_files(files) 

    @staticmethod
    async def save_files(files: List[UploadFile]) -> Dict[str, str]:
        """
        Replacement save_files: save files to disk but DO NOT embed automatically.
        Embedding will be decided later by process_questions() which calls _llm_choose_files or store_url_content.
        """
        saved_files: Dict[str, str] = {}
      
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR, exist_ok=True)

       
        for file in files:
            try:
                file_path = os.path.join(UPLOAD_DIR, file.filename)  # type: ignore
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                saved_files[file.filename.lower()] = file_path  # type: ignore
            except Exception:
                logger.exception("Error saving file %s", getattr(file, "filename", "<unknown>"))
        return saved_files

    @staticmethod
    def retrieve_similar_rows(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            q_emb = embedding_model.embed_query(query)
        except Exception:
            logger.exception("Query embedding failed")
            q_emb = [0.0]
        con = duckdb.connect(DUCKDB_PATH)
        try:
            rows = con.execute(f"SELECT id, metadata, content, embedding FROM {TABLE_NAME}").fetchall()
        finally:
            con.close()
        scored = []
        qv = np.array(q_emb, dtype=float)
        for id_, metadata_json, content, embedding_json in rows:
            try:
                emb = np.array(json.loads(embedding_json), dtype=float)
                denom = (np.linalg.norm(qv) * np.linalg.norm(emb))
                score = float(np.dot(qv, emb) / denom) if denom != 0 else 0.0
            except Exception:
                score = 0.0
            try:
                metadata = json.loads(metadata_json)
            except Exception:
                metadata = {"raw": metadata_json}
            scored.append({"id": int(id_), "metadata": metadata, "content": content, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _clean_llm_csv_text(text: str) -> str:
        """
        Extract CSV-like content from LLM response, ignoring extra text/noise.
        """
        csv_block = re.search(r"```(?:csv|text)?\n([\s\S]*?)\n```", text, re.IGNORECASE)
        if csv_block:
            return csv_block.group(1).strip()
        lines = [l for l in text.splitlines() if ',' in l]
        return "\n".join(lines) if lines else text

    @staticmethod
    def _llm_invoke_with_json_response(system_prompt: str, user_prompt: str, max_retries: int = 2) -> (Optional[Dict[str, Any]], str): # type: ignore
        """
        Ask LLM to produce strict JSON with keys:
          - mode: 'text' or 'image'
          - answer: text explanation (string)
          - csv: optional CSV text (string)
          - plot_type: optional ('scatter','histogram','line','bar')
        Returns (parsed_json_or_None, raw_text)
        """
        full_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nImportant: Respond with a single valid JSON object only."
        raw = ""
        for attempt in range(max_retries + 1):
            try:
                response = llm.invoke(full_prompt)  # type: ignore
                raw = getattr(response, "content", "") or str(response)
                json_text_match = re.search(r"(\{[\s\S]*\})", raw)
                json_text = json_text_match.group(1) if json_text_match else raw
                parsed = json.loads(json_text)
                if "mode" in parsed and "answer" in parsed:
                    return parsed, raw
                else:
                    return {"mode": "text", "answer": raw}, raw
            except Exception:
                logger.exception("LLM invocation or JSON parse failed, attempt %s", attempt)
                if attempt == max_retries:
                    return {"mode": "text", "answer": f"Error invoking LLM or parsing JSON. Raw: {raw}"}, raw
                time.sleep(0.5)
        return None, raw

    @staticmethod
    def process_questions(saved_files: Dict[str, str]) -> List[List[Any]]:
        """
        Main flow:
         - find questions.txt in saved_files
         - if questions.txt contains URL(s) => fetch those pages and embed (only those)
         - else => ask LLM which uploaded files to embed; embed chosen files (fallback to all)
         - then proceed to read questions, retrieve from DB, ask LLM to answer, optionally generate plots
        """
        results: List[List[Any]] = []
        try:
          
            q_file_key = next((f for f in saved_files if f.lower().endswith("questions.txt")), None)
            if not q_file_key:
                raise RuntimeError("No 'questions.txt' file found")
            q_path = saved_files[q_file_key]
            with open(q_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.rstrip("\n") for line in f]

          
            text_after_answer_idx = None
            answer_idx = next((i for i, l in enumerate(lines) if "answer" in l.lower()), None)
            if answer_idx is None:
                raise RuntimeError("No line containing 'answer' found in questions.txt")
            text_after_answer_idx = answer_idx + 1
           
            whole_q_text = "\n".join(lines)
            urls = FileUploadService._detect_urls_in_text(whole_q_text)
         
            if urls:
                logger.info("URLs detected in questions.txt: %s", urls)
                for url in urls:
                    try:
                        FileUploadService.store_url_content(url)
                    except Exception:
                        logger.exception("Failed storing url content for %s", url)
            else:
              
                try:
                    chosen_files = FileUploadService._llm_choose_files(saved_files)
                   
                    for fname in chosen_files:
                        path = saved_files.get(fname)
                        if not path:
                            logger.warning("LLM chose missing file %s", fname)
                            continue
                        try:
                            FileUploadService._embed_file_by_path(path, fname)
                        except Exception:
                            logger.exception("Failed embedding chosen file %s", fname)
                except Exception:
                    logger.exception("LLM file choice failed, embedding all non-questions files")
                   
                    for fname, path in saved_files.items():
                        if fname.endswith("questions.txt"):
                            continue
                        try:
                            FileUploadService._embed_file_by_path(path, fname)
                        except Exception:
                            logger.exception("Failed embedding fallback file %s", fname)

           
            question_lines = [l for l in lines[text_after_answer_idx:] if l.strip()]
            questions = []
            current_q = ""
            for line in question_lines:
                if re.match(r"^\d+\.", line.strip()):
                    if current_q:
                        questions.append(current_q.strip())
                    current_q = re.sub(r"^\d+\.\s*", "", line.strip())
                else:
                    current_q += " " + line.strip()
            if current_q:
                questions.append(current_q.strip())

           
            for idx, q in enumerate(questions, start=1):
                try:
                    retrieved = FileUploadService.retrieve_similar_rows(q, top_k=5)
                    context_text = "\n".join([f"- {r['content']}" for r in retrieved]) if retrieved else "No relevant context found."
                    #improvement required
                    system_prompt = (
                        "You are a smart AI assistant. Use the context below to answer the question.\n"
                        "If the question requests a chart (scatter, histogram, line, bar), respond with a JSON containing 'mode':'image', 'answer':<text>, 'csv':<csv-string>, 'plot_type':<type>.\n"
                        "Otherwise respond with 'mode':'text' and 'answer'.\n"
                        "Return ONLY a JSON object."
                    )
                    user_prompt = f"Context (from vector DB):\n{context_text}\n\nQuestion:\n{q}\n\nInstructions: Be concise and prefer CSV data <= 200 rows for plots."

                    parsed_json, raw = FileUploadService._llm_invoke_with_json_response(system_prompt, user_prompt)
                    ans_text = ""
                    if parsed_json is None:
                        ans_text = f"Error: LLM didn't return valid JSON. Raw: {raw}"
                    else:
                        mode = parsed_json.get("mode", "text").lower()
                        if mode == "image":
                            csv_text = parsed_json.get("csv", "") or FileUploadService._clean_llm_csv_text(parsed_json.get("answer", ""))
                            plot_type = parsed_json.get("plot_type", "line")
                            answer_brief = parsed_json.get("answer", "")
                            # Try to create plot
                            try:
                                cleaned_text = csv_text
                                df = pd.read_csv(io.StringIO(cleaned_text))
                                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                fig = None

                                if "scatter" in plot_type.lower() and len(numeric_cols) >= 2:
                                    corr_matrix = df[numeric_cols].corr().abs()
                                    np.fill_diagonal(corr_matrix.values, 0)
                                    col_pair = corr_matrix.stack().idxmax()
                                    x_col, y_col = col_pair  # type: ignore
                                    plt.close('all')
                                    fig, ax = plt.subplots()
                                    ax.scatter(df[x_col], df[y_col])
                                    ax.set_xlabel(x_col)  # type: ignore
                                    ax.set_ylabel(y_col)  # type: ignore
                                    ax.set_title("Scatter plot")
                                elif "histogram" in plot_type.lower() and numeric_cols:
                                    plt.close('all')
                                    fig, ax = plt.subplots()
                                    df[numeric_cols].hist(ax=ax)
                                elif "line" in plot_type.lower() and numeric_cols:
                                    plt.close('all')
                                    fig, ax = plt.subplots()
                                    df[numeric_cols].plot.line(ax=ax)
                                elif "bar" in plot_type.lower() and numeric_cols:
                                    plt.close('all')
                                    fig, ax = plt.subplots()
                                    df[numeric_cols].plot.bar(ax=ax)

                                if fig is not None:
                                    buf = io.BytesIO()
                                    plt.tight_layout()
                                    fig.savefig(buf, format='png')
                                    plt.close(fig)
                                    buf.seek(0)
                                    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
                                    ans_text = f"data:image/png;base64,{encoded}"
                                else:
                                    ans_text = answer_brief or "Requested an image but unable to create plot."
                            except Exception as e:
                                logger.exception("Plot generation failed")
                                ans_text = f"{answer_brief}\nError generating plot: {e}"
                        else:
                            # text mode
                            answer_text = parsed_json.get("answer", "")
                            score = retrieved[0]["score"] if retrieved else 0.0
                            evidence = "\n".join([f"[{r['score']:.3f}] {r['metadata'].get('source','unknown')}: {r['content'][:200]}" for r in retrieved]) if retrieved else ""
                            full_answer = answer_text + ("\n\nEvidence:\n" + evidence if evidence else "")
                            ans_text = full_answer

                    score = retrieved[0]["score"] if retrieved else 0.0
                    results.append([idx, q, score, ans_text])
                except Exception as e:
                    logger.exception("Error processing question %s", q)
                    results.append([idx, q, 0.0, f"Error processing: {e}"])
        except Exception as e:
            logger.exception("Error processing questions: %s", e)
        return results
