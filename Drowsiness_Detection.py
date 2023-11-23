from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import datetime

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5]) 
    B = distance.euclidean(eye[2], eye[4]) 
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

flag = 0
drowsy_started = None
drowsy_ended = None
drowsiness_count = 0

file_name = "drowsiness_timestamps.txt"
file = open(file_name, "a")

leftEye = None
rightEye = None

while True:
    ret, frame = cap.read()

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        if leftEAR < thresh or rightEAR < thresh:
            flag += 1

            if flag >= frame_check and drowsy_started is None:
                drowsy_started = datetime.datetime.now()
        else:
            if flag >= frame_check:
                drowsy_ended = datetime.datetime.now()
                total_time_drowsy = drowsy_ended - drowsy_started

                if total_time_drowsy.seconds >= 2:
                    drowsiness_count += 1
                    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f"\n\nDate: {current_date}\n")
                    file.write(f"Drowsiness count: {drowsiness_count}\n")
                    hours = total_time_drowsy.seconds // 3600
                    minutes = (total_time_drowsy.seconds % 3600) // 60
                    seconds = total_time_drowsy.seconds % 60
                    total_time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

                    file.write(f"Time interval of drowsiness: {total_time_str}\n")
                
                flag = 0
                drowsy_started = None
                drowsy_ended = None

    cv2.putText(frame, "Detecting eyes...", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if leftEye is not None:
        for (i, j) in zip(range(0, len(leftEye)), range(len(leftEye) - 1)):
            cv2.line(frame, tuple(leftEye[i]), tuple(leftEye[j]), (0, 255, 0), 1)
    if rightEye is not None:
        for (i, j) in zip(range(0, len(rightEye)), range(len(rightEye) - 1)):
            cv2.line(frame, tuple(rightEye[i]), tuple(rightEye[j]), (0, 255, 0), 1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

file.close()
cv2.destroyAllWindows()
cap.release()
