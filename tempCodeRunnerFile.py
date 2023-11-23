
from scipy.spatial import distance # used to calculate the Euclidean distance between points.

from imutils import face_utils # used to get the indices of the left and right eyes from the 68 facial landmarks 
from pygame import mixer  # used to play the alarm sound
import imutils  # used to resize the frames to speed up the processing
import dlib  # t's used for face detection and facial landmark prediction
import cv2  # used to capture the frames from webcam


# Initialize the Pygame mixer and load a music file
mixer.init()
mixer.music.load("music.wav") 

# Define a function to calculate the eye aspect ratio


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5]) # Euclidean distance between the vertical eye landmarks
    B = distance.euclidean(eye[2], eye[4]) # Euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3]) # Euclidean distance between the left-most and right-most eye landmarks or endpoint of each eye landmark
    ear = (A + B) / (2.0 * C)
    return ear



  
#The Eye Aspect Ratio (EAR) is a measure used in computer vision and facial landmark analysis to quantify the openness or closure of the eyes. 
# It is commonly employed in tasks such as drowsiness detection systems, where monitoring eye behavior is essential.

# Set threshold and frame check values for drowsiness detection
thresh = 0.25 #When the EAR falls below this value, the code starts counting frames.
frame_check = 20 # represents the number of consecutive frames in which the EAR must be below the threshold (thresh) to trigger an alert. 

# Initialize face detector and shape predictor using dlib
detect = dlib.get_frontal_face_detector() 
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat") # a pre-trained model file that contains information about facial landmarks.

# Define the indices for left and right eyes in the facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"] 
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Open a video capture object
cap = cv2.VideoCapture(0)

# Initialize a flag variable
flag = 0

# Start an infinite loop for video capture and processing
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=450)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #Grayscale images have one channel compared to the three channels (B, G, R) in a color image. This reduces the dimensionality of the data.

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)

    # Loop over each detected face
    for subject in subjects:
        # Predict facial landmarks
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the eye aspect ratio for each eye
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratios of both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Convex hulls for left and right eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # Draw contours around the eyes
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the threshold
        if ear < thresh:
            flag += 1
            print(flag)

            # If the condition persists for a certain number of frames, trigger an alert
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Play an alert sound
                mixer.music.play()
        else:
            # Reset the flag if eyes are open
            flag = 0

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for user input to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video capture object and close all windows
cv2.destroyAllWindows()
cap.release()
