import cv2
import os
import tensorflow
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from tkinter import *
from tkinter import messagebox




root = Tk()
root.title("Detection_System")


#txt = Entry(window, width = 20)
#txt.grid(row = 4, column = 4)
l1 = Label(root, text = "Driver's Name ", font = 12)
l1.grid(row = 1, column = 1)
t1 = Entry(root, width = 25,bd = 5)
t1.grid(column = 2, row = 1, padx = 10, pady = 10)

l3 = Label(root, text = "Vehicle Number Plate ", font = 12)
l3.grid(row = 2, column = 1)
t3 = Entry(root, width = 25,bd = 5)
t3.grid(column = 2, row = 2, padx = 10, pady = 10)

l2 = Label(root, text = "Vehicle Name ", font = 12)
l2.grid(row = 3, column = 1)
t2 = Entry(root, width = 25,bd = 5)
t2.grid(column = 2, row = 3, padx = 10, pady = 10)


#location of sound file


#location of haar cascades files


def detect():

    if (t1.get() == "" or t2.get() == "" or t3.get() == ""):
        messagebox.showinfo('Result', 'Please fill the gaps')
    else:


    #opening camera or simply capturing video

    ###########def detect():
        face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
        leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
        reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
        lbl = ['Close', 'Open']
        model = load_model('models/cnncat2.h5')
        path = os.getcwd()
    #video capture
        cap = cv2.VideoCapture(0)
        count = 0
        Blink = 0
        mixer.init()
        sound = mixer.Sound('alarm.wav')
        thicc = 2
        rpred = [99]
        lpred = [99]
        while (True):

            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            ret, frame = cap.read()
            height, width = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.3, minSize=(100, 100))
            left_eye = leye.detectMultiScale(gray)
            right_eye = reye.detectMultiScale(gray)

            cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
            # returns array of detection with cordinates
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

            for (x, y, w, h) in right_eye:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 2)
                r_eye = frame[y:y + h, x:x + w]
                count = count + 1
                r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye, (24, 24))
                r_eye = r_eye / 255
                r_eye = r_eye.reshape(24, 24, -1)
                r_eye = np.expand_dims(r_eye, axis=0)
                rpred = model.predict_classes(r_eye)
                if (rpred[0] == 1):
                    lbl = 'Open'
                if (rpred[0] == 0):
                    lbl = 'Closed'
                break

            for (x, y, w, h) in left_eye:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 2)
                l_eye = frame[y:y + h, x:x + w]
                count = count + 1
                l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye, (24, 24))
                l_eye = l_eye / 255
                l_eye = l_eye.reshape(24, 24, -1)
                l_eye = np.expand_dims(l_eye, axis=0)
                lpred = model.predict_classes(l_eye)
                if (lpred[0] == 1):
                    lbl = 'Open'
                if (lpred[0] == 0):
                    lbl = 'Closed'
                break

            if (lpred[0]==0 and rpred[0]==0):
                Blink = Blink + 1
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                # if(rpred[0]==1 or lpred[0]==1):
            else:
                Blink = Blink - 1
                cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if (Blink < 0):
                Blink = 0
            cv2.putText(frame, 'Count:' + str(Blink), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                # setting eye blink and giving score
            if (Blink > 14):
                # person is feeling sleepy so we buzzer the alarm
                cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
                try:
                    sound.play()

                except:  # isplaying = False
                    pass
                if (thicc < 11):
                    thicc = thicc + 2
                else:
                    thicc = thicc - 2
                    if (thicc < 2):
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
            cv2.imshow("Drowsiness Driver's face detection", frame)


            if cv2.waitKey(10) & 0xFF == ord('q'):
                break



        cap.release()
        cv2.destroyAllWindows()
myButton = Button(root, text="Start Detecting!", font = ("Algerian",15), bg ='red', fg = 'green', command=detect)
myButton.grid(row = 100, column = 2)



root.geometry("500x200")
root.mainloop()