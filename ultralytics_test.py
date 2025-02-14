# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # n(����), s(����), m(�̵��), l(����), x(��������)

# Use the model
results = model.predict("https://ultralytics.com/images/zidane.jpg")  # ultralytics ��Ű�� ���� test������ �̹� ������ �ִ� ���� �̹����� �̿��� object detection ����