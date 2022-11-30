import numpy
import cv2
import torch
import os
import PIL
from torchvision import models, transforms


#ResNet-101, GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = models.resnet101(pretrained=True)
#Switch to GPU or CPU execution
resnet  = resnet.to(device)
#Switch model to interference mode
resnet.eval()

#Labels
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

#Preprocessing function
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

#Webcam configuration section
webcam = cv2.VideoCapture(0)

#Configuring Width and Height of video device
frameWidth = 640
frameHeight = 480
   
#Capturing Video from Webcam
webcam.set(3, frameWidth)
webcam.set(4, frameHeight)

while(True):
    #Capture frame-by-frame
    ret, frame = webcam.read()
    #Convert image from video frame into PIL array, normalize image
    frame_c = PIL.Image.fromarray(frame)
    frame_t = preprocess(frame_c)
    #Unsqueeze
    batch_t = torch.unsqueeze(frame_t, 0)
    out = resnet(batch_t)
    indices, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(labels[index[0]], percentage[index[0]].item())
    object = labels[index[0]] + str(percentage[index[0]].item())

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.putText(frame,object,(50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    #cv2.rectangle(frame,(400,150),(900,550), (250,0,0), 2)
    cv2.imshow("Webcam feed",frame)
    # Press Q to break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
webcam.release()
cv2.destroyAllWindows()