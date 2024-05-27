import cv2
import face_recognition
import os
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load known faces
known_face_encodings = []
known_face_names = []

CACHE_FILE = "face_encodings_cache.pkl"


def save_cache(cache_file, encodings, names):
    with open(cache_file, 'wb') as f:
        pickle.dump((encodings, names), f)


def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None, None


def load_known_faces(directory, cache_file):
    global known_face_encodings, known_face_names

    cached_encodings, cached_names = load_cache(cache_file)
    if cached_encodings is not None and cached_names is not None:
        known_face_encodings = cached_encodings
        known_face_names = cached_names
        print(f"Loaded {len(known_face_names)} faces from cache")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
                print(f"Loaded {filename}, total known faces: {len(known_face_names)}")
            else:
                print(f"No faces found in {filename}")

    save_cache(cache_file, known_face_encodings, known_face_names)
    print(f"Saved {len(known_face_names)} faces to cache")


load_known_faces("f_faces_extracted", CACHE_FILE)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Metrics initialization
TP = 0
FP = 0
TN = 0
FN = 0
y_true = []
y_scores = []

# Define the expected name (the name to be recognized)
expected_name = "fabio"

# Define the threshold for face distance
threshold = 0.6

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = "Unknown"

            # Check if the face distance is below the threshold
            if face_distances[best_match_index] < threshold:
                name = known_face_names[best_match_index]
                if expected_name in name:
                    TP += 1
                    y_true.append(1)
                    y_scores.append(1 - face_distances[best_match_index])
                else:
                    FP += 1
                    y_true.append(0)
                    y_scores.append(1 - face_distances[best_match_index])
            else:
                FN += 1
                y_true.append(1)
                y_scores.append(1 - face_distances[best_match_index])

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        label = name if name != "Unknown" else "Unknown Face"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

# Calculate metrics
accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
FAR = FP / (FP + TN) if (FP + TN) != 0 else 0
FRR = FN / (TP + FN) if (TP + FN) != 0 else 0
FMR = FP / (FP + TN) if (FP + TN) != 0 else 0
FNMR = FN / (TP + FN) if (TP + FN) != 0 else 0

print(f"Accuracy: {accuracy}")
print(f"FAR: {FAR}")
print(f"FRR: {FRR}")
print(f"FMR: {FMR}")
print(f"FNMR: {FNMR}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
