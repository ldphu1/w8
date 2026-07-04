import uuid
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import time
import uvicorn
import numpy as np
from contextlib import asynccontextmanager
import traceback

from starlette.concurrency import run_in_threadpool

from ocr import ocr

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models")
    app_state["models"] = ocr()
    print("Loading success")
    app_state["models"].warmup()
    yield
    app_state.clear()

app = FastAPI(title="OCR", lifespan=lifespan)

@app.get("/health", status_code=200)
async def health_check():
    models = app_state.get("models")

    if not models:
        return JSONResponse(
            status_code=503,
            content={
                "status": 503,
                "reason": "Models are not initialized or loading failed."
            },
        )

    try:
        if hasattr(models, "ocr") and hasattr(models, "detect_card") and hasattr(models, "detect_field"):
            return{
                "status": "Healthy",
                "message": "System is ready to inference"
            }
        else:
            raise Exception("Models component are missing")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "reason": f"Model health check failed. Error: {e}"
            }
        )

@app.get("/model-info", status_code=200)
async def get_model_info():
    models = app_state.get("models")
    if not models:
        return JSONResponse(
            status_code=503,
            content={
                "status": "Error",
                "detail": "Models are not initialized."
            }
        )

    try:
        model_details = models.get_info()

        return{
            "status": "Success",
            "data": model_details
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "Error",
                "detail": f"Failed to retrieve models info. Error: {e}",
            }
        )

async def pre_img(img: UploadFile = File(...)):
    img_bytes = await img.read()

    if len(img_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image is too big")

    nparr = np.frombuffer(img_bytes, np.uint8)

    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_cv

@app.post("/inference", status_code=200)
async def inference(file: UploadFile = File(...)):
    req_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        model = app_state.get("models")

        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File type not supported")

        t0 = time.time()
        image = await pre_img(file)
        preprocess_time = round((time.time() - t0) * 1000, 2)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        t1 = time.time()
        results, classes = await run_in_threadpool(model.inference, image)
        inference_time = round((time.time() - t1) * 1000, 2)

        t2 = time.time()
        ocr_lines = {}
        names = model.detect_field.names
        for result, cls in zip(results, classes):
            ocr_lines[names[int(cls)]] = {
                "text": result[0],
                "score": result[1],
            }
        postprocess_time = round((time.time() - t2) * 1000, 2)

        total = preprocess_time + inference_time + postprocess_time

        return JSONResponse(
            status_code=200,
            content={
                "request_id": req_id,
                "status": "Success",
                "predict": ocr_lines,
                "latency": {
                    "preprocess_time": preprocess_time,
                    "inference_time": inference_time,
                    "postprocess_time": postprocess_time,
                    "total": total
                }
            }
        )
    except HTTPException as e:
        raise e

    except Exception as e:
        error_trace = traceback.format_exc()
        inference_time = round((time.time() - t1) * 1000, 2)

        print(f"\n[ERROR] Reques_ID: {req_id}")
        print(f"[ERROR] Traceback: {error_trace}")

        return JSONResponse(
            status_code=500,
            content={
                "request_id": req_id,
                "status": "error",
                "message": "Error occur when execute image.",
                "error_details": str(e),
                "inference_time_ms": inference_time
            }
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
