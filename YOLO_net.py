import cv2
import numpy as np
import time
import sys
import os

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
# конфигурация нейронной сети
config_path = "cfg/yolov3.cfg"
# файл весов сети YOLO
weights_path = "weights/yolov3.weights"
# weights_path = "weights/yolov3-tiny.weights"
# загрузка всех меток классов (объектов)
#labels = open("data/coco.names").read().strip().split("\n")
# генерируем цвета для каждого объекта и последующего построения
#colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

path_name = "F:/key_points/13171.jpg"
image = cv2.imread(path_name)
file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")

h, w = image.shape[:2]
# создать 4D blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

print("image.shape:", image.shape)
print("blob.shape:", blob.shape)

# устанавливает blob как вход сети
net.setInput(blob)
# получаем имена всех слоев
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# прямая связь (вывод) и получение выхода сети
# измерение времени для обработки в секундах
start = time.perf_counter()
layer_outputs = net.forward(ln)
time_took = time.perf_counter() - start
print(f"Потребовалось: {time_took:.2f}s")

