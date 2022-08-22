from tkinter import *
from tkinter.filedialog import *
from PIL import Image, ImageTk
import cv2
import os
import sys


def select_model():
    global vid, haar_cascade, face_recognizer, webcam, score, state, state_label
    file = askopenfile(initialdir=folder_path,
                       filetypes=[('YML Document', '*.yml')])
    if file is not None:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(file.name)
        vid = cv2.VideoCapture(0)
        haar_cascade = cv2.CascadeClassifier(os.path.join(folder_path, "Resources/haar_face.xml"))
        webcam = Label(window)
        webcam.place(x=250, y=150)
        score = StringVar()
        score_label = Label(window, textvariable=score, bg="black", fg="white",
                            font=("Courier", 30))
        score_label.place(x=60, y=330)
        state = StringVar()
        state_label = Label(window, textvariable=state, bg="black", fg="white",
                            font=("Courier", 30))
        state_label.place(x=930, y=310)
        b0.destroy()
        show_frame()


def show_frame():
    if vid.isOpened():
        cap = vid.read()[1]
        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        face_points = haar_cascade.detectMultiScale(gray,
                                                    scaleFactor = 1.25,
                                                    minNeighbors = 5)

        for (x, y, w, h) in face_points:
            cv2.rectangle(cap, (x, y), (x + w, y + h), (255, 0, 0), 2)
            faces_roi = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(faces_roi, (200, 200))
            label, confidence = face_recognizer.predict(resized_face)
            if confidence < 500:
                accuracy = float("{0:.2f}".format((100 * (1 - (confidence) / 300))))
                score.set(str(accuracy))
                if accuracy >= 80:
                    state.set("Device\nUnlocked")
                    state_label.configure(fg="green")
                else:
                    state.set("Device\nLocked")
                    state_label.configure(fg="red")

        cv2image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

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
window.title("Prediction")
icon = PhotoImage(file=os.path.join(folder_path, "Resources/icon.png"))
window.iconphoto(False, icon)
window.geometry("1152x700")
window.resizable(False, False)
window.configure(bg="#ffffff")

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

img0 = PhotoImage(file=os.path.join(folder_path, "Resources/img0.png"))
b0 = Button(
    image=img0,
    borderwidth=0,
    highlightthickness=0,
    command=select_model,
    cursor="hand2",
    relief="flat")

b0.place(
    x=475, y=275,
    width=223,
    height=83)

img1 = PhotoImage(file=os.path.join(folder_path, "Resources/img1.png"))
b1 = Button(
    image=img1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: window.destroy(),
    cursor="hand2",
    relief="flat")

b1.place(
    x=1020, y=626,
    width=132,
    height=74)

window.mainloop()
