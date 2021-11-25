import torch
import config
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import FaceKeypointResNet50
import time

model = FaceKeypointResNet50(pretrained=True, requires_grad=False).to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load('F:/key_points/model_more_20.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
cap = cv2.VideoCapture("F:/key_points/13171.jpg")
ret, frame = cap.read()
with torch.no_grad():
    image = frame
    image = cv2.resize(image, (224, 224))
    plt.show()
    orig_frame = image.copy()

    orig_h, orig_w, c = orig_frame.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float)
    image = image.unsqueeze(0).to(config.DEVICE)
    outputs = model(image)
    # print(outputs)
    count = 0
    point = []
    x = []
    y = []

    for i in outputs:
        for j in i:
            # print(str(j)[7:len(str(j))-1])
            if count % 3 == 0:
                point.append(float(str(j)[7:len(str(j)) - 1]))
            if count % 3 == 1:
                x.append(float(str(j)[7:len(str(j)) - 1]))
            if count % 3 == 2:
                y.append(float(str(j)[7:len(str(j)) - 1]))
            count += 1
    frame_width = 1200
    frame_height = 800
    orig_frame = cv2.resize(orig_frame, (frame_width, frame_height))
    print(point)
    print(x)
    print(y)
    for i in range(len(point)):
        if (point[i] > 0.5):
            orig_frame = cv2.circle(orig_frame, (int(x[i] * 1200), int(y[i] * 800)), radius=10,
                                    color=(250, 0, 0), thickness=-1)
    plt.imshow(orig_frame)
    plt.show()