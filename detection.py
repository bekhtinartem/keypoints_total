import torchvision.transforms as transforms
import cv2
import numpy
import numpy as np
import torchvision
import numpy
import torch
import argparse
from PIL import Image
import config
from model import FaceKeypointResNet50
import matplotlib.pyplot as plt
local_model = FaceKeypointResNet50(pretrained=True, requires_grad=False).to(config.DEVICE)
checkpoint = torch.load('F:/key_points/model_1more_30.pth', map_location='cpu')
local_model.load_state_dict(checkpoint['model_state_dict'])
local_model.eval()
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval().to(device)
coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])
def get_points(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float)
    image = image.unsqueeze(0).to(config.DEVICE)
    outputs = local_model(image)
    count = 0
    point = []
    x = []
    y = []
    for i in outputs:
        for j in i:
                if count % 3 == 0:
                    point.append(float(str(j)[7:(str(j).find(',')) - 1]))
                if count % 3 == 1:
                    x.append(float(str(j)[7:(str(j).find(',')) - 1]))
                if count % 3 == 2:
                    y.append(float(str(j)[7:(str(j).find(',')) - 1]))
                count += 1

    return point, x, y
def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[0]['labels']

def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        area=image[int(box[1]): int(box[3]), int(box[0]):int(box[2])]
        h, w = area.shape[:2]
        point, x, y = get_points(area)
        for i in range(len(point)):

            if(point[i]>0.5):
                area = cv2.circle(area, (int(x[i] * w), int(y[i] * h)), radius=5,
                                        color=(250, 0, 0), thickness=-1)

        image[int(box[1]): int(box[3]), int(box[0]):int(box[2])]=area
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )

        '''cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)'''
    return image




image = "F:/Dataset/data/antelope/antelope_10026.jpg"

def run(image):
    image=cv2.imread(image)
    plt.imshow(image)
    plt.show()
    boxes, classes, labels = predict(image, model, device, 0.5)
    image = draw_boxes(boxes, classes, labels, image)
    plt.imshow(image)
    plt.show()
import glob
for i in glob.glob("F:/Dataset/data/antelope/**"):
    run(i)


