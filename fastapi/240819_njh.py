from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import os
import io
import math
from typing import List, Dict, Any
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import ffmpeg
from datetime import timedelta



# YOLO 모델 로드
model = YOLO("yolov8n.pt")

app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5500"],  # 또는 ["*"]로 모든 도메인 허용
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 필요한 도메인만 설정
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 시간을 hh:mm:ss:ms 형식으로 변환하는 함수
def format_time(seconds):
    # 밀리세컨드를 계산합니다.
    millis = round((seconds % 1) * 1000)  # 밀리세컨드 계산 및 반올림

    # timedelta를 이용해 시간을 시, 분, 초로 변환합니다.
    time_delta = timedelta(seconds=int(seconds))
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 밀리세컨드를 2자리로 포맷합니다.
    millis = int(millis / 10)  # 밀리세컨드를 2자리로 변환

    # 형식을 맞추기 위해 시, 분, 초를 포맷합니다.
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}.{millis:02}"

    return formatted_time

@app.post("/uploadfile/")
async def get_inference_data(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        contents = await file.read()
        temp_file.write(contents)

    try:
        cap = cv2.VideoCapture(temp_file.name)
        fps = cap.get(cv2.CAP_PROP_FPS)  # FPS 값을 가져옵니다.
        frame_count = 0
        frame_skip = 5

        label_list = set()
        result_list = []
        object_timeline = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = frame_count / fps  # 현재 프레임의 시간을 초 단위로 계산
            formatted_time = format_time(current_time)  # hh:mm:ss:ms 형식으로 변환
            if frame_count % frame_skip == 0:
                try:
                    results = model.track(frame, persist=True)
                    for result in results:
                        boxes = result.boxes.xyxy
                        classes = result.boxes.cls
                        ids = result.boxes.id
                        detections = []
                        if boxes is not None and classes is not None and ids is not None:
                            for box, cls, obj_id in zip(boxes, classes, ids):
                                x1, y1, x2, y2 = box
                                label = model.model.names[int(cls)]
                                formatted_detection = {
                                    "id": int(obj_id),
                                    "class_name": label,
                                    "xmin": x1.item(),
                                    "xmax": x2.item(),
                                    "ymin": y1.item(),
                                    "ymax": y2.item(),
                                }
                                label_list.add(label)
                                detections.append(formatted_detection)
                                # 객체의 등장 시간과 퇴장 시간을 기록합니다 (hh:mm:ss:ms 형식)
                                obj_id_int = int(obj_id)
                                if obj_id_int not in object_timeline:
                                    object_timeline[obj_id_int] = {"label": label, "start": formatted_time, "end": formatted_time}
                                else:
                                    object_timeline[obj_id_int]["end"] = formatted_time
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
        
    # 타임라인 포멧팅
    format_timeline = []

    for key, value in object_timeline.items():
        format_timeline.append({
            'id': key,
            'label': value['label'],
            'start': value['start'],
            'end': value['end']
        })
    
    return {"label_list": list(label_list), "timeline": format_timeline, "det_result": result_list}

@app.post("/result-video/")
async def create_det_video(
    label_filter: str = Form(...),
    det_result: str = Form(...),
    file: UploadFile = File(...)
):
    # 결과값 파싱
    det_result = json.loads(det_result)
    label_filter = json.loads(label_filter)

    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name

    output_path = 'output.mp4'
    
    try:
        # 영상 정보 추출
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            raise RuntimeError("Error opening video file.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 저장 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 프레임 관리
        frame_count = 0
        frame_skip = 1

        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break
            
            # 현재 프레임에 대한 인덱스 계산
            det_result_index = frame_count // frame_skip

            # 인덱스가 det_result 범위 내에 있는지 확인
            if det_result_index < len(det_result):
                for obj in det_result[det_result_index]:
                    if obj['class_name'] in label_filter:
                        x1, x2, y1, y2 = int(obj['xmin']), int(obj['xmax']), int(obj['ymin']), int(obj['ymax'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'id: {obj["id"]} class: {obj["class_name"]}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

    finally:
        if cap.isOpened():
            cap.release()
        os.unlink(temp_file_path)

    # 영상 데이터를 바이너리 형식으로 변환
    with open(output_path, 'rb') as video_file:
        video_bytes = video_file.read()

    # 임시 영상 지우기
    os.remove(output_path)

    return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4", headers={"Content-Disposition": "attachment; filename=output.mp4"})
