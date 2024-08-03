# from sklearn.neighbors import KNeighborsClassifier

# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime
# from win32com.client import Dispatch


# def speak(str1):
#     speak = Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)


# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# if not os.path.exists('data/'):
#     os.makedirs('data/')

# with open('data/names.pkl', 'rb') as f:
#     LABELS = pickle.load(f)

# with open('data/faces_data.pkl', 'rb') as f:
#     FACES = pickle.load(f)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)
# imgBackground = cv2.imread("background.png")








# COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

# def check_if_exists(value):
#     try:
#         with open("Votes.csv", "r") as csvfile:
#             reader = csv.reader(csvfile)
#             for row in reader:
#                 if row and row[0] == value:
#                     return True
#     except FileNotFoundError:
#         print("File not found or unable to open the CSV file.")
#     return False

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
#     output = None  # Initialize output to a default value
#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w]
#         resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
#         output = knn.predict(resized_img)
#         ts = time.time()
#         date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist = os.path.isfile("Votes.csv")
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
#         cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
#         cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
#         attendance = [output[0], timestamp]
        
#     imgBackground[370:370 + 480, 225:225 + 640] = frame
#     cv2.imshow('frame')
#     k = cv2.waitKey(1)
    
#     if output is not None:
#         voter_exist = check_if_exists(output[0])
#         if voter_exist:
#             speak("YOU HAVE ALREADY VOTED")
#             break

#         if k == ord('1'):
#             speak("YOUR VOTE HAS BEEN RECORDED")
#             time.sleep(2)
#             if exist:
#                 with open("Votes.csv", "a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     attendance = [output[0], "BJP", date, timestamp]
#                     writer.writerow(attendance)
#             else:
#                 with open("Votes.csv", "a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     writer.writerow(COL_NAMES)
#                     attendance = [output[0], "BJP", date, timestamp]
#                     writer.writerow(attendance)
#             speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
#             break

#         if k == ord('2'):
#             speak("YOUR VOTE HAS BEEN RECORDED")
#             time.sleep(5)
#             if exist:
#                 with open("Votes.csv", "a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     attendance = [output[0], "CONGRESS", date, timestamp]
#                     writer.writerow(attendance)
#             else:
#                 with open("Votes.csv", "a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     writer.writerow(COL_NAMES)
#                     attendance = [output[0], "CONGRESS", date, timestamp]
#                     writer.writerow(attendance)
#             speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
#             break

#         if k == ord('3'):
#             speak("YOUR VOTE HAS BEEN RECORDED")
#             time.sleep(5)
#             if exist:
#                 with open("Votes.csv", "a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     attendance = [output[0], "AAP", date, timestamp]
#                     writer.writerow(attendance)
#             else:
#                 with open("Votes.csv", "a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     writer.writerow(COL_NAMES)
#                     attendance = [output[0], "AAP", date, timestamp]
#                     writer.writerow(attendance)
#             speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
#             break

#         if k == ord('4'):
#             speak("YOUR VOTE HAS BEEN RECORDED")
#             time.sleep(5)
#             if exist:
#                 with open("Votes.csv", "a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     attendance = [output[0], "NOTA", date, timestamp]
#                     writer.writerow(attendance)
#             else:
#                 with open("Votes.csv", "a") as csvfile:
#                     writer = csv.writer(csvfile)
#                     writer.writerow(COL_NAMES)
#                     attendance = [output[0], "NOTA", date, timestamp]
#                     writer.writerow(attendance)
#             speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
#             break

# video.release()
# cv2.destroyAllWindows()






















import tkinter as tk
from tkinter import messagebox
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from PIL import Image, ImageTk

# Function to handle voice output
def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

# Initialize the main window
root = tk.Tk()
root.title("Voting System")

# Initialize the camera
video = cv2.VideoCapture(0)
if not video.isOpened():
    messagebox.showerror("Error", "Could not open the camera.")
    root.destroy()
    exit()

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if not os.path.exists('data/'):
    os.makedirs('data/')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

def check_if_exists(value):
    try:
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except FileNotFoundError:
        print("File not found or unable to open the CSV file.")
    return False

def capture_photo():
    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            messagebox.showerror("Error", "Failed to capture image from camera.")
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            cv2.imshow('frame', frame)

            voter_exist = check_if_exists(output[0])
            if voter_exist:
                speak("YOU HAVE ALREADY VOTED")
                messagebox.showinfo("Already Voted", "You have already voted.")
                return None

            return output[0], date, timestamp

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

def submit_vote(candidate):
    if not candidate:
        messagebox.showwarning("No Selection", "Please select a candidate to vote for.")
        return

    voter_info = capture_photo()
    if voter_info is None:
        return

    voter_name, date, timestamp = voter_info
    speak("YOUR VOTE HAS BEEN RECORDED")
    time.sleep(2)
    exist = os.path.isfile("Votes.csv")
    if exist:
        with open("Votes.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            attendance = [voter_name, candidate, date, timestamp]
            writer.writerow(attendance)
    else:
        with open("Votes.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(COL_NAMES)
            attendance = [voter_name, candidate, date, timestamp]
            writer.writerow(attendance)
    speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
    messagebox.showinfo("Vote Submitted", f"Your vote for {candidate} has been submitted!")
    root.destroy()

def start_voting():
    voter_info = capture_photo()
    if voter_info is None:
        return
    
    # Create GUI components for voting options
    candidate_var = tk.StringVar()
    candidate_var.set(None)

    candidates = ["Party 1", "Party 2", "Party 3", "NOTA"]
    option_buttons = []
    for candidate in candidates:
        btn = tk.Radiobutton(root, text=candidate, variable=candidate_var, value=candidate)
        btn.pack()
        option_buttons.append(btn)

    # Submit vote button
    submit_button = tk.Button(root, text="Submit Vote", command=lambda: submit_vote(candidate_var.get()))
    submit_button.pack()

# Start the voting process
start_voting()

# Start the main loop
root.mainloop()

# Release the camera if it is still open
if video.isOpened():
    video.release()
cv2.destroyAllWindows()
