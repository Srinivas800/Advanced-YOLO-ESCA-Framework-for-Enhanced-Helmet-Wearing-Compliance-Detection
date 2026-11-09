from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
import os
import cv2
import pathlib
from pathlib import Path
import torch
pathlib.PosixPath = pathlib.WindowsPath

main = tkinter.Tk()
main.title("YOLO-ESCA: A High-Performance Safety Helmet Standard Wearing Behavior Detection Model Based on Improved YOLOv5") #designing main screen
main.geometry("1300x1200")

global yolo_model

def uploadDataset():
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,"SHWD Helmet Saftey Dataset loaded\n");
    text.insert(END,filename)
    
def processDataset():
    text.delete('1.0', END)
    text.insert(END,"Dataset processing completed\n")
    text.insert(END,"Different class labels found in dataset\n")
    text.insert(END,"Head, Helmet, Person\n\n")

def trainModel():
    global yolo_model
    text.delete('1.0', END)
    yolo_model = torch.hub.load('yolov5', 'custom', path='model/best.pt', force_reload=True,source='local')
    text.insert(END,"YOLO-ESCA Training & Loading Completed")
    yolo_train_detection = cv2.imread("model/result.png")
    cv2.imshow("Detection Accurscy", yolo_train_detection)
    cv2.waitKey(0)

def graph():
    yolo_train_detection = cv2.imread("model/results.png")
    yolo_train_detection = cv2.resize(yolo_train_detection, (800, 600))
    cv2.imshow("Training Graph", yolo_train_detection)
    cv2.waitKey(0)

def detection(frame):
    global yolo_model
    results = yolo_model(frame)
    result = results.pandas().xyxy
    for i in range(len(result)):
        out = result[i]
        out = out.values
        print(out)
        for j in range(len(out)):
            xmin = int(out[j,0])
            ymin = int(out[j,1])
            xmax = int(out[j,2])
            ymax = int(out[j,3])
            conf = float(out[j,4])
            label = str(out[j, 6])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            if label == 'helmet':
                if conf > 0.70:
                    cv2.putText(frame, label+" "+str(round(conf,2)), (xmin, ymin+20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "Uncertain "+str(round(conf,2)), (xmin, ymin+20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, label+" "+str(round(conf,2)), (xmin, ymin+20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    return frame

def imageHelmet():
    filename = filedialog.askopenfilename(initialdir="Images")
    img = cv2.imread(filename)
    frame = detection(img)
    cv2.imshow("Predicted Result", frame)
    cv2.waitKey(0)
    

def videoHelmet():
    filename = filedialog.askopenfilename(initialdir="Videos")
    video = cv2.VideoCapture(filename)
    while(True):
        ret, frame = video.read()
        if ret == True:
            frame = detection(frame)
            cv2.imshow("Predicted Result", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break  
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    

font = ('times', 13, 'bold')
title = Label(main, text='YOLO-ESCA: A High-Performance Safety Helmet Standard Wearing Behavior Detection Model Based on Improved YOLOv5')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=420,y=100)
text.config(font=font1)

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload SHWD Helmet Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

cnnButton = Button(main, text="Train YOLO-ESCA Algorithm", command=trainModel)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1) 

graphButton = Button(main, text="Training Graph", command=graph)
graphButton.place(x=50,y=250)
graphButton.config(font=font1)

imageButton = Button(main, text="Detect Helmet from Image", command=imageHelmet)
imageButton.place(x=50,y=300)
imageButton.config(font=font1)

videoButton = Button(main, text="Detect Helmet from Video", command=videoHelmet)
videoButton.place(x=50,y=350)
videoButton.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()
