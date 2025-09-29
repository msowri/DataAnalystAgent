import os
import shutil
import pytest # type: ignore
import pandas as pd
from PIL import Image
from io import BytesIO
from unittest.mock import patch, AsyncMock
from app.services.fileuploadservice import FileUploadService, UPLOAD_DIR, DB_DIR, DUCKDB_PATH, TABLE_NAME

@pytest.fixture(scope="module", autouse=True)
def setup_environment():
    # Reset environment before tests
    FileUploadService.reset_environment()
    yield
    # Cleanup after tests
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

def test_reset_environment():
    # Environment should create uploads and database directories
    assert os.path.exists(UPLOAD_DIR)
    assert os.path.exists(DB_DIR)

    import duckdb
    con = duckdb.connect(DUCKDB_PATH)
    tables = con.execute("SHOW TABLES").fetchall()
    con.close()
    assert (TABLE_NAME,) in tables

def test_store_file_embedding(tmp_path):
    file_path = tmp_path / "sample.csv"
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df.to_csv(file_path, index=False)

    FileUploadService.store_file_embedding(str(file_path))

    import duckdb
    con = duckdb.connect(DUCKDB_PATH)
    rows = con.execute(f"SELECT content FROM {TABLE_NAME}").fetchall()
    con.close()
    assert len(rows) > 0
    assert "col1" in rows[-1][0]

def test_analyze_csv(tmp_path):
    file_path = tmp_path / "sample.csv"
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df.to_csv(file_path, index=False)

    output = FileUploadService.analyze_csv_or_excel(str(file_path))
    assert "col1" in output
    assert "col2" in output

def test_encode_image(tmp_path):
    file_path = tmp_path / "sample.png"
    img = Image.new('RGB', (10, 10), color='red')
    img.save(file_path)

    encoded = FileUploadService.encode_image(str(file_path))
    assert encoded.startswith("data:image/png;base64,")

@patch("app.services.fileuploadservice.requests.get")
def test_scrape_url(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = "<html><body><p>Hello</p></body></html>"

    result = FileUploadService.scrape_url("http://example.com")
    assert "Hello" in result

def test_process_questions(tmp_path):
    questions_file = tmp_path / "questions.txt"
    questions_file.write_text("What is AI?\n")

    saved_files = {"questions.txt": str(questions_file)}

    with patch("app.services.fileuploadservice.llm.invoke") as mock_llm:
        mock_llm.return_value.content = "Artificial Intelligence is the simulation of human intelligence."
        results = FileUploadService.process_questions(saved_files)

    assert len(results) == 1
    assert "Artificial Intelligence" in results[0]["answer"]

@pytest.mark.asyncio
async def test_save_files(tmp_path):
    from fastapi import UploadFile

    content = b"Hello World"
    file = UploadFile(filename="test.txt", file=BytesIO(content))
    saved_files = await FileUploadService.save_files([file])

    assert "test.txt" in saved_files
    with open(saved_files["test.txt"], "rb") as f:
        data = f.read()
    assert data == content

def test_retrieve_context(tmp_path):
    # Reset environment to clean DuckDB
    FileUploadService.reset_environment()

    # Create two dummy files
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("Hello world from file1")
    file2.write_text("Hello world from file2")

    # Store embeddings
    FileUploadService.store_file_embedding(str(file1))
    FileUploadService.store_file_embedding(str(file2))

    context = FileUploadService.retrieve_context("Hello world")
    assert isinstance(context, list)
    assert len(context) > 0
