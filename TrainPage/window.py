from tkinter import *
from tkinter.filedialog import *
from PIL import Image, ImageTk
import cv2
import os
import sys
import time
from threading import Thread
import numpy as np


def save_file(model, username):
    b0.config(state=DISABLED)
    file_name = f'{username}_facemodel'
    file = asksaveasfile(initialfile=file_name,
                         initialdir=folder_path,
                         filetypes=[('YML Document', '*.yml')],
                         defaultextension=[('YML Document', '*.yml')])
    model.save(file.name)
    file.close()
    alert.set("Model Saved!")
    img2 = ImageTk.PhotoImage(file=os.path.join(folder_path, "Resources/img2.png"))
    b0.img2 = img2
    b0.config(state=NORMAL, command=lambda: window.destroy(), image=img2)


def proceed():
    name = entry0.get()
    if name and name.strip():
        b0.config(state=DISABLED)
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(features, labels)
        alert.set("Training Completed!")
        img3 = ImageTk.PhotoImage(file=os.path.join(folder_path, "Resources/img3.png"))
        b0.img3 = img3
        b0.config(state=NORMAL,
                  command=lambda: save_file(face_recognizer, name),
                  image=img3)
    else:
        alert.set("Please fill up your Name!")


def count_func():
    b0.config(state=DISABLED)
    alert.set("Position your face in the Camera!\nTurn your head, so we get to know you!")
    global features, labels
    features = []
    labels = []
    count = StringVar()
    countdown = Label(window, textvariable=count,
                      font=("Calibri", "45"),
                      bg="white",
                      fg="black")
    countdown.place(x=330, y=300)
    for i in range(3, 0, -1):
        count.set(str(i))
        time.sleep(1)
    count.set("GO")
    for i in range(1, 100 + 1):
        face_cropped = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(face_cropped, (200, 200))
        features.append(resized_face)
        labels.append(0)
        count.set(str(i))
        time.sleep(0.1)

    features = np.array(features)
    labels = np.array(labels)

    count.set("DONE")
    alert.set("Photo Samples Taken!")
    img1 = ImageTk.PhotoImage(file=os.path.join(folder_path, "Resources/img1.png"))
    b0.img1 = img1
    b0.config(command=proceed, image=img1, state=NORMAL)


def show_frame():
    global gray, x, y, w, h
    if vid.isOpened():
        cap = vid.read()[1]
        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        face_points = haar_cascade.detectMultiScale(gray, scaleFactor = 1.25,
                                                    minNeighbors = 5)

        for (x, y, w, h) in face_points:
            cv2.rectangle(cap, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
        cv2image = cv2image[int((cv2image.shape[0] / 2) - 200):
                            int((cv2image.shape[0] / 2) + 200),
                            int((cv2image.shape[1] / 2) - 250):
                            int((cv2image.shape[1] / 2) + 250)]
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        webcam.imgtk = imgtk
        webcam.configure(image=imgtk)
        webcam.after(20, show_frame)


if getattr(sys, 'frozen', False):
    folder_path = os.path.dirname(sys.executable)
else:
    folder_path = os.path.dirname(__file__)

window = Tk()
window.title("Training")
icon = PhotoImage(file=os.path.join(folder_path, "Resources/icon.png"))
window.iconphoto(False, icon)
window.geometry("1152x700")
window.resizable(False, False)
window.configure(bg="#ffffff")

vid = cv2.VideoCapture(0)
haar_cascade = cv2.CascadeClassifier(os.path.join(folder_path, "Resources/haar_face.xml"))

canvas = Canvas(
    window,
    bg="#ffffff",
    height=700,
    width=1152,
    bd=0,
    highlightthickness=0,
    relief="ridge")
canvas.place(x=0, y=0)

background_img = PhotoImage(file=os.path.join(folder_path, "Resources/background.png"))
background = canvas.create_image(
    576.0, 350.0,
    image=background_img)

entry0_img = PhotoImage(file=os.path.join(folder_path, "Resources/img_textBox0.png"))
entry0_bg = canvas.create_image(
    401.0, 631.0,
    image=entry0_img)

entry0 = Entry(
    bd=0,
    bg="#f3f3f3",
    highlightthickness=0,
    font=("Calibri 25"))

entry0.place(
    x=231.0, y=606,
    width=340.0,
    height=48)

img0 = PhotoImage(file=os.path.join(folder_path, "Resources/img0.png"))
b0 = Button(
    image=img0,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: Thread(target=count_func).start(),
    relief="flat",
    cursor="hand2",
    state=NORMAL)

b0.place(
    x=797, y=314,
    width=223,
    height=83)

alert = StringVar()

alert_label = Label(
    textvariable=alert,
    fg="black",
    bg="white",
    font=("Rancho-Regular", int(20.0)))

alert_label.place(x=650, y=484.5)

webcam = Label(window)
webcam.place(x=100, y=140)

show_frame()
window.mainloop()
