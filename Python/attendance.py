import face_recognition
import os
import cv2
import numpy as np
import pickle
from datetime import datetime

# ---------- Paths and Global Setup ----------
path = 'Images'
encoding_file = 'encodings.pkl'
images = []
classNames = []

# ---------- Load Images from Folder ----------
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# ---------- Encode Faces ----------
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

# ---------- Load or Generate Encodings ----------
if os.path.exists(encoding_file):
    print("Loading encodings from file...")
    with open(encoding_file, 'rb') as f:
        encodeListKnown, classNames = pickle.load(f)
    print("Encodings loaded successfully.")
else:
    print("Encodings not found. Generating...")
    encodeListKnown = findEncodings(images)
    with open(encoding_file, 'wb') as f:
        pickle.dump((encodeListKnown, classNames), f)
    print("Encodings saved to file.")

# ---------- Mark Attendance ----------
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        existing_entries = [line.strip().split(',') for line in myDataList]
        nameList = [entry[0] for entry in existing_entries]

        now = datetime.now()
        timeString = now.strftime('%H:%M:%S')
        dateString = now.strftime('%Y-%m-%d')

        already_marked = any(entry[0] == name and entry[-1] == dateString for entry in existing_entries)

        if not already_marked:
            f.writelines(f'\n{name},{timeString},{dateString}')
            print(f"Marked attendance for {name} at {timeString} on {dateString}")
        else:
            print(f"Attendance already marked for {name} today.")

# ---------- Face Recognition from Webcam ----------
registered_this_session = set()
cap = cv2.VideoCapture(0)
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", 800, 600)

while True:
    success, img = cap.read()
    if not success:
        print("Could not access webcam.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if True not in matches:
            print("New Face Detected")

            face_id = tuple(encodeFace)
            if face_id in registered_this_session:
                continue

            new_face = input("Please enter your full name: ")

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            face_crop = img[y1:y2, x1:x2]

            save_path = os.path.join(path, f'{new_face}.jpg')
            cv2.imwrite(save_path, face_crop)
            print(f"Saved cropped face to {save_path}")

            # Update images and classNames
            images.append(face_crop)
            classNames.append(new_face)

            # Recompute encodings and save
            encodeListKnown = findEncodings(images)
            with open(encoding_file, 'wb') as f:
                pickle.dump((encodeListKnown, classNames), f)
            print("Encodings updated and saved.")

            # Encode cropped image again to confirm match
            registered_img = cv2.imread(save_path)
            registered_img = cv2.cvtColor(registered_img, cv2.COLOR_BGR2RGB)
            reg_enc = face_recognition.face_encodings(registered_img)
            if reg_enc and face_recognition.compare_faces([reg_enc[0]], encodeFace)[0]:
                markAttendance(new_face.upper())
            else:
                print("Could not confirm match for new face.")

            registered_this_session.add(face_id)
            continue

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


