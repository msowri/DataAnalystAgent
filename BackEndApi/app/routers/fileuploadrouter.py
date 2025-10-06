from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
from app.services.fileuploadservice import FileUploadService

router = APIRouter(
    prefix="/files",
    tags=["files"]
)

@router.post("/upload-and-process")
async def upload_and_process(files: List[UploadFile] = File(...)):
    """
    Upload multiple files including questions.txt, CSV/XLSX files.   
    """
    try:
        FileUploadService.reset_environment()     
        saved_files: Dict[str, str] = await FileUploadService.save_files(files)    
        results = FileUploadService.process_questions(saved_files)
        return JSONResponse(content={"results": results})
    except HTTPException as error:      
        raise error
    except Exception as e:    
        raise HTTPException(status_code=500, detail=str(e))
