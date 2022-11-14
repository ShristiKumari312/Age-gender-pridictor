import cv2
import math
import argparse
def highlightface(net,frame,conf_threshold=0.7):
    frameopencvdnn = frame.copy()
    frameheight=frameopencvdnn.shape[0]
    framewith=frameopencvdnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameopencvdnn,1.0,(300,300),[104,117,123],True,False)
    net.setInput(blob)
    detection=net.forward()
    faceboxes=[]
    for i in range (detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detection[0,0,i,3]*framewith)
            y1=int(detection[0,0,i,4]*frameheight)
            x2=int(detection[0,0,i,5]*framewith)
            y2=int(detection[0,0,i,6]*frameheight)
            faceboxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameopencvdnn,(x1,y1),(x2,y2),(0,250,0),int(round(frameheight/150)),8)
    return frameopencvdnn,faceboxes

parser=argparse.ArgumentParser()
parser.add_argument('--image')
args=parser.parse_args()
faceproto= "opencv_face_detector.pbtxt"
facemodel= "opencv_face_detector_uint8.pb"
ageproto="age_deploy.prototxt"
agemodel="age_net.caffemodel"
genderproto="gender_deploy.prototxt"
gendermodel="gender_net.caffemodel"
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
facenet=cv2.dnn.readNet(facemodel,faceproto)
agenet=cv2.dnn.readNet(agemodel,ageproto)
gendernet=cv2.dnn.readNet(gendermodel,genderproto)
video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1)<0:
    hasframe,frame=video.read()
    if not hasframe:
        cv2.waitKey()
        break
    resulting,faceboxes=highlightface(facenet,frame)
    if not faceboxes:
        print ("No face deducted")
    for facebox in faceboxes:
        face=frame[max(0,facebox[1]-padding):min(facebox[3]+padding,frame.shape[0]-1),max(0,facebox[0]-padding):min(facebox[2]+padding,frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(frame, 1.0, (227,227),MODEL_MEAN_VALUES,swapRB=False)
        gendernet.setInput(blob)
        genderpredict=gendernet.forward()
        gender=genderList[genderpredict[0].argmax()]
        print(f'gender:{gender}')
        agenet.setInput(blob)
        agepredict=agenet.forward()
        age=ageList[agepredict[0].argmax()]
        print(f'age:{age[1:-1]}')
        cv2.putText(resulting,f'{gender},{age}',(facebox[0],facebox[1]-10),cv2.FONT_HERSHEY_DUPLEX,0.8,(0, 255, 255),2,cv2.LINE_AA)
        cv2.imshow('video',resulting)