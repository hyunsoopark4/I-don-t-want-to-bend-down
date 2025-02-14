# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # n(나노), s(스몰), m(미디엄), l(라지), x(엑스라지)

# Use the model
results = model.predict("https://ultralytics.com/images/zidane.jpg")  # ultralytics 패키지 내에 test용으로 이미 존재해 있는 버스 이미지를 이용해 object detection 수행