from fastapi import FastAPI, File, Form, UploadFile
import uvicorn
import os
import io
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Tuple

from utils.asr import stt
import time

app = FastAPI()

#############################################################################
print(os.getcwd())
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    For local js, css swagger in Cathay
    :return:
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


##############################################################################

@app.get("/")
def HelloWorld():
    return {"Hello": "World"}


@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    t0 = time.time()
    # 獲取文件名和文件大小
    file_name = file.filename
    file_size = len(await file.read())
    
    t1 = time.time()
    # 返回文件信息
    return JSONResponse(content={"file_name": file_name, 
                                 "file_size": file_size,
                                 "processing_time": f'{t1-t0:.2f} s'})


@app.post("/speech2text/")
async def speech2text(file: UploadFile = File(...)):
    t0 = time.time()
    # 將上傳的音頻文件讀取到內存中
    audio_data = await file.read()
    audio_file = io.BytesIO(audio_data)

    ls_result = stt(audio_file)
    t1 = time.time()
    # 返回轉錄文本
    return JSONResponse(content={"transcription": ls_result, 
                                 "processing_time": f'{t1-t0:.2f} s'})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5110))
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)
