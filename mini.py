import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import dlib

# Extract frames from the video
cap = cv2.VideoCapture('frame_video1.mp4')

# Initialize an empty list to store the frames
frames = []

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If we reached the end of the video, break the loop
    if not ret:
        break

    # Convert the frame to grayscale and add it to the list
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(frame)

# Release the video capture
cap.release()

# Detect faces in the frames using Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize an empty list to store the face landmarks
face_landmarks = []

for frame in frames:
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    # Extract the face landmarks for each face
    for (x, y, w, h) in faces:
        landmarks = np.array([[p.x, p.y] for p in predictor(frame, dlib.rectangle(x, y, x + w, y + h)).parts()])
        face_landmarks.append(landmarks)

# Cluster the face landmarks using DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(face_landmarks)

# Print the number of clusters
print(len(np.unique(db.labels_)))

# Visualize the clusters by plotting the face landmarks on a scatter plot
colors = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(db.labels_))))

for k, col in zip(np.unique(db.labels_), colors):
    class_member_mask = (db.labels_ == k)
    xy = face_landmarks[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col)