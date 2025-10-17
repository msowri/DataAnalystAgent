from fastapi import  FastAPI
from fastapi.middleware.cors import  CORSMiddleware
from app.routers import testrouter
from app.routers.fileuploadrouter import router as fileupload_router
import uvicorn


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

# if __name__ == "__main__":
#     import os
#     port = int(os.getenv("PORT", 8000))
#     uvicorn.run(
#         "app.main:app",
#         host="0.0.0.0",
#         port=port,
#         reload=False,       # disable reload for production
#         log_level="info"    # useful for debugging
#     )
