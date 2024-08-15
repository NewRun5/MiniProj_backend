from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import io
import numpy as np
from collections import defaultdict
import cv2
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision
import tempfile
import os

# YOLO 모델 로드
model = YOLO("yolov8n.pt")

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
    # 비디오 파일 경로 설정
    # video_path = Image.open(io.BytesIO(contents))
    # 비디오 캡처 객체 생성
    try:
        cap = cv2.VideoCapture(temp_file.name)
        # 객체 추적 기록을 저장할 딕셔너리 생성
        track_history = defaultdict(lambda: [])

        # image = mp.Image(
        #     image_format=mp.ImageFormat.SRGB, data=np.asarray(video_path))
        
        # 비디오가 열려있는 동안 반복
        while cap.isOpened():
            # 프레임 읽기
            success, frame = cap.read()
            if success:
                # YOLO 모델을 사용하여 객체 감지 및 추적
                results = model.track(frame, persist=True)
                # 감지된 객체의 바운딩 박스 좌표 추출
                boxes = results[0].boxes.xywh.cpu()
                # 추적된 객체의 ID 추출
                track_ids = results[0].boxes.id.int().cpu().tolist()
                # 결과를 시각화한 프레임 생성
                annotated_frame = results[0].plot()

                # 각 객체에 대해 반복
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    # 현재 위치를 추적 기록에 추가
                    track.append((float(x), float(y)))
                    # 추적 기록이 30개를 초과하면 가장 오래된 기록 삭제
                    if len(track) > 30:
                        track.pop(0)
                    # 추적 기록을 선으로 그리기 위해 포인트 배열 생성
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    # 추적 경로를 선으로 그리기
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                # 결과 프레임 표시
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
                # 'q' 키를 누르면 루프 종료
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        # 비디오 캡처 객체 해제 및 모든 창 닫기
        cap.release()
        cv2.destroyAllWindows()

    finally:
        # 임시 파일 삭제
        os.unlink(temp_file.name)

    return {"filename": file.filename}