from fastapi import  FastAPI
from fastapi.middleware.cors import  CORSMiddleware
from app.routers import testrouter
from app.routers.fileuploadrouter import router as fileupload_router



app= FastAPI(
title="Backend API",
    description="This API powers the Learning App .",
    version="1.0.0",    
)
@app.get("/")
def read_root():
    return {"status": "running"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

# include routers
app.include_router(testrouter.router, prefix="/api")
app.include_router(fileupload_router, prefix="/api")
