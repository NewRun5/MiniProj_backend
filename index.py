from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import io
import numpy as np


app = FastAPI()



@app.get("/")
def 작명():
     return FileResponse('index.html')

from pydantic import BaseModel
class Model(BaseModel):
     name :str
     phone :int

@app.post("/send")
def 작명(data : Model):
     print(data)
     return '전송완료'