from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import os
import math
from typing import List
import json

# YOLO 모델 로드
model = YOLO("yolov8n.pt")
    
class BoxData(BaseModel):
    id: int
    class_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    
class CreateVideoRequest(BaseModel):
    target_obj_list: List[str]
    box_data: List[BoxData]

app = FastAPI()

@app.post("/uploadfile/")
async def get_inference_data(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        contents = await file.read()
        temp_file.write(contents)

    try:
        cap = cv2.VideoCapture(temp_file.name)
        fps = cap.get(cv2.CAP_PROP_FPS)  # FPS 값을 가져옵니다.
        frame_count = 0
        frame_skip = 10

        label_list = set()
        result_list = []
        object_timeline = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = frame_count / fps  # 현재 프레임의 시간을 초 단위로 계산
            if frame_count % frame_skip == 0:
                try:
                    results = model.track(frame, persist=True)
                    for result in results:
                        boxes = result.boxes.xyxy
                        classes = result.boxes.cls
                        ids = result.boxes.id
                        detections = []
                        for box, cls, obj_id in zip(boxes, classes, ids):
                            x1, y1, x2, y2 = box
                            label = model.model.names[int(cls)]
                            formatted_detection = {
                                "id": int(obj_id),
                                "class": label,
                                "xmin": x1.item(),
                                "xmax": x2.item(),
                                "ymin": y1.item(),
                                "ymax": y2.item(),
                            }
                            label_list.add(label)
                            detections.append(formatted_detection)
                            # 객체의 등장 시간과 퇴장 시간을 기록합니다.
                            obj_id_int = int(obj_id)  # 문자열로 변환하여 사용
                            if obj_id_int not in object_timeline:
                                object_timeline[obj_id_int] = {"start": math.floor(current_time), "end": math.floor(current_time)}
                            else:
                                object_timeline[obj_id_int]["end"] = math.ceil(current_time)
                        result_list.append(detections)
                except cv2.error as e:
                    print(f"OpenCV error during tracking: {e}")
                    continue
            frame_count += 1
        
        cap.release()

    finally:
        if cap.isOpened():
            cap.release()
        os.unlink(temp_file.name)
        
    det_result = {"result_list": result_list, "timeline": object_timeline}
    return {"label_list": list(label_list), "det_result": det_result}

# @app.post("/result-video/")
# async def create_det_video(
#     file: UploadFile = File(...),
#     request: str = Form(...),
# ):
#     try:
#         # JSON 문자열을 Pydantic 모델로 변환
#         parsed_request = CreateVideoRequest(**json.loads(request))
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=400, detail="Invalid JSON format")
    
#     # parsed_request를 사용하여 필요한 작업을 수행합니다.
#     print(parsed_request)
    
#     # 예시: 성공 응답
#     return {"status": "success", "parsed_request": parsed_request}