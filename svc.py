from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.asr import stt, convert_to_wav, merge_audio_files
import time
import uvicorn
import os
import io
import filetype  # 用於檢測文件類型

app = FastAPI(
    doc_url="/",
    title="語音辨識ASR API",
)

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


@app.post("/transcribe-file/")
async def transcribe_file(file: UploadFile = File(...)):
    t0 = time.time()
    
    # 讀取文件內容
    audio_data = await file.read()
    
    # 檢查文件類型
    kind = filetype.guess(audio_data)
    if kind is None or not kind.mime.startswith('audio/'):
        raise HTTPException(status_code=400, detail="文件不是有效的錄音檔")
    
    # 將上傳的音頻文件讀取到內存中
    audio_file = io.BytesIO(audio_data)

    ls_result = stt(audio_file)
    t1 = time.time()
    # 返回轉錄文本
    return JSONResponse(content={"transcription": ls_result, 
                                 "processed_time": f'{t1-t0:.2f} s'})


# 管道字典，用于存储不同通道的音频数据
pipelines = {}

@app.post("/transcribe-audio/")
async def transcribe_audio(channel: int = Form(...), file: UploadFile = File(...), action: str = Form(None)):
    t0 = time.time()
    # 初始化管道，如果不存在
    if channel not in pipelines:
        pipelines[channel] = []

    if action == "reset":
        # 重置指定通道的管道
        pipelines[channel] = []
        return JSONResponse(content={"message": f"Channel {channel} has been reset."})

    # 將上傳的音頻文件讀取道內存中
    audio_data = await file.read()
    audio_file = io.BytesIO(audio_data)

    # 檢查文件格式並轉換為WAV
    if file.filename.endswith(".mp3"):
        audio_file = convert_to_wav(audio_file, "mp3")
    elif file.filename.endswith(".wav"):
        pass
    else:
        return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

    # 添加音頻數據到指定通道的管道中
    pipelines[channel].append(audio_file)

    # 合併管道中的所有音頻數據
    combined_audio = merge_audio_files(pipelines[channel])

    # 打印合併音頻的大小
    combined_audio_size = combined_audio.getbuffer().nbytes
    print(f"Combined audio size for channel {channel}: {combined_audio_size} bytes")

    ls_result = stt(combined_audio)
    t1 = time.time()
    # 返回轉錄文本
    return JSONResponse(content={"transcription": ls_result, 
                                 "processed_time": f'{t1-t0:.2f} s'})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5111))
    uvicorn.run(app, log_level='info', port=port)
