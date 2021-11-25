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
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
new_poln=[]
new_delta=[]
for i in range(0, 37):
    new_poln.append([])
    new_delta.append([])



model.eval()
t_p=[]
t_x=[]
t_y=[]
classes=[]
delta_classes=[]
poln_res_classes=[]
f=open("F:/key_points/test_data_all_points.csv", 'r')
count=0
number_class=-1
start=True
t0=time.time()
delta=[]
poln_1=[]
for s in f:
    if s!="end":
        poln=0
        count+=1
        #s=s[1:]
        name=s[0:s.find(";")]

        start=False
        s=s[s.find(";")+1:]
        k=s.split(";")
        k.pop()
        #print(len(k))
        point1=[]
        x1=[]
        y1=[]
        for i in range(len(k)):
            if(i%3==0):
               point1.append(float(k[i]))
            if (i % 3 == 1):
                x1.append(float(k[i]))
            if (i % 3 == 2):
                y1.append(float(k[i]))
        t_p.append(point1)
        t_x.append(x1)
        t_y.append(y1)
        name=name.capitalize()
        #print(name)
        id=("F:/Dataset/data_for_testing/" + name)
        #print(id)
        cap = cv2.VideoCapture(id)



        for i in range(1):
            # capture each frame of the video
            ret, frame = cap.read()
            if True:
                with torch.no_grad():
                    image = frame
                    h, w = image.shape[:2]
                    image = cv2.resize(image, (224, 224))
                    #plt.show()
                    orig_frame = image.copy()

                    orig_h, orig_w, c = orig_frame.shape
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image / 255.0
                    image = np.transpose(image, (2, 0, 1))
                    image = torch.tensor(image, dtype=torch.float)
                    image = image.unsqueeze(0).to(config.DEVICE)
                    outputs = model(image)
                    #print(outputs)
                    count=0
                    point=[]
                    x=[]
                    y=[]

                    for i in outputs:
                        for j in i:
                            #print(str(j)[7:len(str(j))-1])
                            if count%3==0:
                                point.append(float(str(j)[7:len(str(j))-1]))
                            if count%3==1:
                                x.append(float(str(j)[7:len(str(j))-1]))
                            if count %3==2:
                                y.append(float(str(j)[7:len(str(j))-1]))
                            count+=1



                outputs = outputs.cpu().detach().numpy()
                keypoints = outputs
                frame_width = w
                frame_height = h
                orig_frame = cv2.resize(orig_frame, (frame_width, frame_height))

                countt=0
                delta_x=0.0
                delta_y=0.0
                countn=0
                for i in range(len(point)):
                    delta_x = 0.0
                    delta_y = 0.0
                    x[i]*=w
                    x[i]=int(x[i])
                    y[i]*=h
                    y[i]=int(y[i])
                    if(point1[i]==1):
                        countt+=1
                        countn+=1
                        if(point[i]>0.5):

                            poln+=1
                            delta_x+=abs(x[i]-x1[i])
                            delta_y+=abs(y[i]-y1[i])
                            new_delta[i].append(1.0*(delta_x+delta_y)/(2*(h+w)) *100)
                            new_poln[i].append(1)
                        else:
                            new_poln[i].append(0)
                    else:
                        countn += 1
                        if (point[i] < 0.5):
                            poln += 1
                            new_poln[i].append(1)
                        else:
                            new_poln[i].append(0)

                #print(number_class)
                poln_1.append(1.0*poln/countn)
                delta.append(1.0*(delta_x+delta_y)/(2*(h+w)*countt) *100)
                #print(1.0*poln/countn)
                #print(1.0*(delta_x+delta_y)/(2*(h+w)*countt) *100)


print()
print()
for i in range(len(new_poln)):
    c=0
    sum=0
    cd=0
    sd=0
    for j in range(len(new_poln[i])):
        c+=1
        sum+=new_poln[i][j]
    for j in range(len(new_delta[i])):
        cd+=1
        sd+=new_delta[i][j]
    print(1.0*sum/c*100 , "   ", 1.0*sd/cd)

