# -*- coding: utf-8 -*-
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # n(����), s(����), m(�̵��), l(����), x(��������)


# Load a Image
img_path = "bus.jpg"
results = model(img_path)

# Use the model
# results = model.predict("https://ultralytics.com/images/zidane.jpg")  # ultralytics ��Ű�� ���� test������ �̹� ������ �ִ� ���� �̹����� �̿��� object detection ����

# ��� �̹��� �ҷ�����
annotated_img = results[0].plot()   # �ٿ�� �ڽ� �׷��� �̹��� �ҷ�����


#print(type(annotated_img)) # <class 'numpy.ndarray'> ���� ����
#print(annotated_img.shape) # (H, W, 3) �����̾�� ����
#cv2.imwrite("debug_output.jpg", annotated_img)  # ������ �̹��� ����


# matplolib�� �̿��� ���
plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))  #openCV�� BGR �����̹Ƿ� ����ϱ� ���� RGB�� ��ȯ
plt.axis("off") #�� ����
plt.savefig("output_plot.jpg")
