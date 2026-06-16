import io
import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="Fire & Person Detection API")

# Allow the frontend (any origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load models once at startup ----
FIRE_MODEL_PATH = r"C:\Users\veena\.cache\huggingface\hub\models--SalahALHaismawi--yolov26-fire-detection\snapshots\5397b43994c39c2ed4ef81a6996c6334e2b7a5fa\best.pt"

fire_model = YOLO(FIRE_MODEL_PATH)
# yolov8n is the standard COCO model; class 0 == "person"
person_model = YOLO("yolov8n.pt")

CONF = 0.25


class FrameRequest(BaseModel):
    image: str  # base64-encoded data URL or raw base64


class Detection(BaseModel):
    label: str
    confidence: float
    box: list  # [x1, y1, x2, y2]


class DetectResponse(BaseModel):
    detections: list[Detection]


def decode_base64_image(data: str) -> np.ndarray:
    """Decode a base64 (optionally data-URL) string into a BGR numpy image."""
    if "," in data:
        data = data.split(",", 1)[1]
    img_bytes = base64.b64decode(data)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def run_person_detection(img: np.ndarray) -> list[Detection]:
    detections = []
    results = person_model.predict(source=img, conf=CONF, classes=[0], verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            detections.append(
                Detection(
                    label="person",
                    confidence=round(conf, 3),
                    box=[round(v, 1) for v in xyxy],
                )
            )
    return detections


def run_fire_detection(img: np.ndarray) -> list[Detection]:
    detections = []
    results = fire_model.predict(source=img, conf=CONF, verbose=False)
    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            label = names.get(cls_id, "fire") if isinstance(names, dict) else "fire"
            detections.append(
                Detection(
                    label=label,
                    confidence=round(conf, 3),
                    box=[round(v, 1) for v in xyxy],
                )
            )
    return detections


@app.get("/")
def root():
    return {"status": "ok", "message": "Fire & Person Detection API running"}


@app.post("/detect/person", response_model=DetectResponse)
async def detect_person(file: UploadFile = File(...)):
    img_bytes = await file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return DetectResponse(detections=run_person_detection(img))


@app.post("/detect/fire", response_model=DetectResponse)
async def detect_fire(file: UploadFile = File(...)):
    img_bytes = await file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return DetectResponse(detections=run_fire_detection(img))


@app.post("/detect/all", response_model=DetectResponse)
async def detect_all(file: UploadFile = File(...)):
    """Detect both person and fire in one uploaded image."""
    img_bytes = await file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    dets = run_person_detection(img) + run_fire_detection(img)
    return DetectResponse(detections=dets)


@app.post("/detect/frame", response_model=DetectResponse)
async def detect_frame(req: FrameRequest):
    """Endpoint used by the webcam frontend. Accepts a base64 frame."""
    img = decode_base64_image(req.image)
    if img is None:
        return DetectResponse(detections=[])
    dets = run_person_detection(img) + run_fire_detection(img)
    return DetectResponse(detections=dets)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)