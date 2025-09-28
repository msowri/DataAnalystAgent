from fastapi import APIRouter
from app.services.testservice import gethellomessage

router = APIRouter()

@router.get("/message")
def get_message():
    return gethellomessage()