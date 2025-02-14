# -*- coding: utf-8 -*-
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # n(나노), s(스몰), m(미디엄), l(라지), x(엑스라지)


# Load a Image
img_path = "bus.jpg"
results = model(img_path)

# Use the model
# results = model.predict("https://ultralytics.com/images/zidane.jpg")  # ultralytics 패키지 내에 test용으로 이미 존재해 있는 버스 이미지를 이용해 object detection 수행

# 결과 이미지 불러오기
annotated_img = results[0].plot()   # 바운딩 박스 그려진 이미지 불러오기


#print(type(annotated_img)) # <class 'numpy.ndarray'> 여야 정상
#print(annotated_img.shape) # (H, W, 3) 형식이어야 정상
#cv2.imwrite("debug_output.jpg", annotated_img)  # 디버깅용 이미지 저장


# matplolib을 이용한 출력
plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))  #openCV는 BGR 형식이므로 출력하기 위해 RGB로 변환
plt.axis("off") #축 제거
plt.savefig("output_plot.jpg")
