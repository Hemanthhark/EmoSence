from tkinter import font
from tensorflow.keras.models import model_from_json
from tkinter.filedialog import askopenfile

import cv2
import numpy as np
import webbrowser
from logging import root
import tkinter

from tkinter import *
import tkinter as tk


root=Tk()
root.geometry("700x600")


root.title("FACE EMOTION BASED MUSIC RECOMMANDER")
#img = PhotoImage(file="2.png")
#label = Label(
      #root,
      #image=img)
#label.place(x=0,y=0,width=600,height=500)

tit=Label(text="FACE EMOTION BASED MUSIC RECOMMANDER",bg='dark olive green',font=('times',16,'bold'))
tit.place(x=100,y=10)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_weights.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def video_detect(video_link = 0):
    cap=cv2.VideoCapture(video_link)  

    while True:  
        ret,test_img=cap.read()# captures frame and returns boolean value and captured image  
        if not ret:  
            continue  
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  

        try:
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x,y,w,h) in faces_detected:  
                cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)  
                roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
                roi_gray=cv2.resize(roi_gray,(48,48))  
                img = roi_gray.reshape((1,48,48,1))
                img = img /255.0

                max_index = np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]

                global predicted_emotion 
                predicted_emotion = emotions[max_index]  

                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        except:
            pass    

        resized_img = cv2.resize(test_img, (1000, 700))  
        cv2.imshow('Facial emotion analysis ',resized_img)  



        if cv2.waitKey(10) == ord('s'):#wait until 's' key is pressed  
            cap.release()  
            cv2.destroyAllWindows()
            if predicted_emotion=="neutral":       
                # url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0kSZqQE2zKYSisa5Vd33-WI"
                url = "https://open.spotify.com/track/7tYKa4wd7gL5LwcxidBPkG?si=f350bec799b04fe3"
                webbrowser.open(url)
            elif predicted_emotion=="sad":
                url = "https://open.spotify.com/track/7tYKa4wd7gL5LwcxidBPkG?si=f350bec799b04fe3"

                webbrowser.open(url)
            elif predicted_emotion=="happy":
                url = "https://open.spotify.com/track/7tYKa4wd7gL5LwcxidBPkG?si=f350bec799b04fe3"

                webbrowser.open(url)
            elif predicted_emotion=="angry":
                url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0kIvJ_FDr8u8EEnPSoTXPcR"
                webbrowser.open(url)
            elif predicted_emotion=="surprise":
                url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0nkF4xN7YbmNFtkQ6Qn6bf3"
                webbrowser.open(url)
            elif predicted_emotion=="disgust":
                url = ": https://youtube.com/playlist?list=PL8Nkf7hoNm0m11La5RgqiugRQIHp7aVFl"
                webbrowser.open(url)
            elif predicted_emotion=="fear":
                url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0mqhWoAnmYeg5bnSVQZ380q"
                webbrowser.open(url)
            break
        
        
    

def image_detect():
    file = askopenfile(filetypes =[('file selector', '*.jpg')])
    print(str(file.name))
    c_img = cv2.imread(file.name)
    gray_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x,y,w,h) in faces_detected:  
        cv2.rectangle(c_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)  
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img = roi_gray.reshape((1,48,48,1))
        img = img /255.0

        max_index = np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]

                  
        predicted_emotion_image = emotions[max_index]  
        print(predicted_emotion_image)

        cv2.putText(c_img, predicted_emotion_image, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    resized_img = cv2.resize(c_img, (1000, 700))  
    cv2.imshow('Facial emotion analysis ',resized_img)
    cv2.imwrite("All_Emotions_Detection.jpg", resized_img)
    if cv2.waitKey(0) == ord('s'):
        cv2.destroyAllWindows()
    
    if predicted_emotion_image=="neutral":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0kSZqQE2zKYSisa5Vd33-WI"
        webbrowser.open(url)
    elif predicted_emotion_image=="sad":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0ldVRSUhH8UauQ2Z4BvZVis"
        webbrowser.open(url)
    elif predicted_emotion_image=="happy":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0m4vO2_P69Phxe4undoNYKZ"
        webbrowser.open(url)
    elif predicted_emotion_image=="angry":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0kIvJ_FDr8u8EEnPSoTXPcR"
        webbrowser.open(url)
    elif predicted_emotion_image=="surprise":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0nkF4xN7YbmNFtkQ6Qn6bf3"
        webbrowser.open(url)
    elif predicted_emotion_image=="disgust":
        url = ": https://youtube.com/playlist?list=PL8Nkf7hoNm0m11La5RgqiugRQIHp7aVFl"
        webbrowser.open(url)
    elif predicted_emotion_image=="fear":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0mqhWoAnmYeg5bnSVQZ380q"
        webbrowser.open(url)


def changeOnHover(button, colorOnHover, colorOnLeave):
  
    # adjusting backgroung of the widget
    # background on entering widget
    button.bind("<Enter>", func=lambda e: button.config(
        background=colorOnHover))
  
    # background color on leving widget
    button.bind("<Leave>", func=lambda e: button.config(
        background=colorOnLeave))






b1=Button(root,text="video_test",command=video_detect,width=20,bg='black',fg='white',height=2,font=('times',12,'bold'))
b1.place(x=150,y=300)
changeOnHover(b1, "gray", "black")
b2=Button(root,text="image_test",command=image_detect,bg='black',fg='white',width=20,height=2,font=('times',12,'bold'))
b2.place(x=400,y=300)
changeOnHover(b2, "gray", "black")


root.mainloop()


#image_detect("test9.jpg")
#video_detect()