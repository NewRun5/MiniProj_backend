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

        # 프레임 변환을 위한 새로운 리스트 생성
        frame_list = []
        # 현재 작업중인 프레임 기록
        frame_count = 0
        # 속도 향상을 위한 프레임 조절
        frame_skip = 1
        
        while True:
            ret, frame = cap.read()  # 프레임 읽기

            if not ret:
                break  # 더 이상 읽을 프레임이 없으면 종료
            if frame_count % frame_skip == 0:
                # 현재 프레임을 RGB 형식으로 변환 (OpenCV는 기본적으로 BGR 형식을 사용)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 변환된 프레임을 PIL 이미지로 변환
                pil_image = Image.fromarray(frame_rgb)
                # 변환된 이미지를 리스트에 추가
                frame_list.append(pil_image)
            frame_count += 1
            print(frame_count)

        # 자원 회수
        cap.release()
        # cv2.destroyAllWindows()
        result_list = []
        for frame in frame_list:
            # YOLOv8에서는 직접 PIL 이미지를 넣기보다는 numpy 배열을 사용
            frame_np = np.array(frame)
            results = model(frame_np)
            # results[0].boxes.boxes는 numpy 배열로, 각 행은 [x1, y1, x2, y2, score, class] 형식
            detections = []
            for result in results:
                # Boxes 객체에서 필요한 데이터를 직접 접근
                boxes = result.boxes.xyxy  # [x1, y1, x2, y2] 형식의 박스 좌표
                scores = result.boxes.conf  # 신뢰도 점수
                classes = result.boxes.cls  # 클래스 레이블
                
                detections = []
                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    label = model.model.names[int(cls)]
                    detections.append({
                        'label': label,
                        'score': float(score),
                        'box': {'xmin': x1.item(), 'ymin': y1.item(), 'xmax': x2.item(), 'ymax': y2.item()}
                    })
                result_list.append(detections)

            print(len(result_list))

        # 감지된 객체 set 생성
        label_set = set()
        for results in result_list:
            for result in results:
                label_set.add(result['label'])

        # set 출력
        message = '현재 감지된 객체 리스트'
        for label in label_set:
            message += f'| {label}'
        print(message)

        # 제외할 객체 입력
        input_label = ''
        while True:
            input_label = input('제외할 객체를 입력해주세요. (종료는 q 입력)')
            if input_label == 'q':
                break
            if input_label in label_set:
                label_set.remove(input_label)
            message = ''
            for label in label_set:
                message += f'| {label}'
            print('남은 객체' + message)

        frame = 1
        for results in result_list:
            print(f'{frame}프레임')
            message = ''
            for result in results:
                if result['label'] in label_set:
                    print(f"객체명: {result['label']} 일치율: {result['score']:.2f} x:{result['box']['xmin']} ~ {result['box']['xmax']} y: {result['box']['ymin']} ~ {result['box']['ymax']}")
            frame += 1

        # 원본 비디오를 다시 열기
        cap = cv2.VideoCapture(temp_file.name)

        # 비디오 속성 가져오기
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 버림나눗셈으로 결과 인덱스 추론
            result_index = min(frame_count // frame_skip, len(result_list) - 1)
            
            results = result_list[result_index]
            for result in results:
                if result['label'] in label_set:
                    # 박스 그리기
                    x1, y1 = int(result['box']['xmin']), int(result['box']['ymin'])
                    x2, y2 = int(result['box']['xmax']), int(result['box']['ymax'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 레이블과 점수 표시
                    label = f"{result['label']}: {result['score']:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 프레임 번호 표시
            cv2.putText(frame, f"Frame: {frame_count} Box-Frame: {frame_skip}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 프레임을 출력 비디오에 쓰기
            out.write(frame)
            frame_count += 1

        # 객체 고유 ID 관리
        next_object_id = 1
        object_timeline = defaultdict(lambda: {'start': None, 'end': None})
        active_objects = {}  # 현재 화면에 존재하는 객체들 {id: box}

        def calculate_iou(box1, box2):
            """ 두 박스 간의 IoU 계산 """
            x1, y1, x2, y2 = box1['xmin'], box1['ymin'], box1['xmax'], box1['ymax']
            x1_p, y1_p, x2_p, y2_p = box2['xmin'], box2['ymin'], box2['xmax'], box2['ymax']

            inter_x1 = max(x1, x1_p)
            inter_y1 = max(y1, y1_p)
            inter_x2 = min(x2, x2_p)
            inter_y2 = min(y2, y2_p)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x2_p - x1_p) * (y2_p - y1_p)

            iou = inter_area / float(box1_area + box2_area - inter_area)
            return iou

        iou_threshold = 0.5  # IoU threshold for matching objects

        frame = 1
        for results in result_list:
            current_frame_objects = {}
            for result in results:
                if result['label'] in label_set:
                    box = result['box']
                    matched = False
                    
                    for obj_id, obj_box in active_objects.items():
                        iou = calculate_iou(obj_box, box)
                        if iou > iou_threshold:
                            current_frame_objects[obj_id] = box
                            object_timeline[obj_id]['end'] = frame / fps
                            matched = True
                            break

                    if not matched:
                        # 새로운 객체로 간주
                        current_frame_objects[next_object_id] = box
                        object_timeline[next_object_id] = {
                            'label': result['label'],
                            'start': frame / fps,
                            'end': frame / fps
                        }
                        next_object_id += 1

            active_objects = current_frame_objects
            frame += 1

        # 등장 시간과 사라진 시간 출력
        for obj_id, times in object_timeline.items():
            if times['start'] is not None:
                print(f"객체 ID: {obj_id} ({times['label']}) 등장 시작: {times['start']:.2f}초 | 등장 종료: {times['end']:.2f}초")

        # 자원 해제
        cap.release()
        out.release()

    finally:
        # 임시 파일 삭제
        os.unlink(temp_file.name)

    return {"filename": file.filename}