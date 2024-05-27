import cv2
import face_recognition
import os
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse

# Constants
KNOWN_CACHE_FILE = "known_face_encodings_cache.pkl"
STRANGER_CACHE_FILE = "stranger_face_encodings_cache.pkl"
THRESHOLD = 0.5

# Metrics
TP = 0
FP = 0
TN = 0
FN = 0
total_faces_detected = 0
imposters_detected = 0
real_faces_detected = 0
y_true = []
y_scores = []


def save_cache(cache_file, encodings, names):
    with open(cache_file, 'wb') as f:
        pickle.dump((encodings, names), f)


def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None, None


def load_faces(directory, cache_file):
    face_encodings, face_names = load_cache(cache_file)
    if face_encodings is not None and face_names is not None:
        print(f"Loaded {len(face_names)} faces from cache")
        return face_encodings, face_names

    face_encodings = []
    face_names = []

    for filename in os.listdir(directory):
        if filename.endswith((".jpeg", ".png", ".jpg")):
            image_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                face_encodings.append(encodings[0])
                face_names.append(os.path.splitext(filename)[0])
                print(f"Loaded {filename}, total faces: {len(face_names)}")
            else:
                print(f"No faces found in {filename}")

    save_cache(cache_file, face_encodings, face_names)
    print(f"Saved {len(face_names)} faces to cache")
    return face_encodings, face_names


def process_frame(frame, known_face_encodings, known_face_names, stranger_face_encodings, stranger_face_names,
                  expected_name, threshold):
    global TP, FP, TN, FN, total_faces_detected, imposters_detected, real_faces_detected, y_true, y_scores

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    total_faces_detected += len(face_encodings)
    face_names = []
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        name = "Unknown"
        if face_distances[best_match_index] < threshold:
            name = known_face_names[best_match_index]
            if expected_name in name:
                TP += 1
                real_faces_detected += 1
                y_true.append(1)
                y_scores.append(1 - face_distances[best_match_index])
            else:
                FP += 1
                imposters_detected += 1
                y_true.append(0)
                y_scores.append(1 - face_distances[best_match_index])
        else:
            # Check against stranger faces
            face_distances = face_recognition.face_distance(stranger_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < threshold:
                name = "Stranger"
                FP += 1
                imposters_detected += 1
                y_true.append(0)
                y_scores.append(1 - face_distances[best_match_index])
            else:
                FN += 1
                real_faces_detected += 1
                y_true.append(1)
                y_scores.append(1 - face_distances[best_match_index])

        face_names.append(name)

    return face_locations, face_names


def draw_results(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        color = (0, 0, 255) if name == "Unknown" or name == "Stranger" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = name if name != "Unknown" else "Unknown Face"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame


def video_stream(video_source, known_face_encodings, known_face_names, stranger_face_encodings, stranger_face_names,
                 expected_name, threshold):
    video_capture = cv2.VideoCapture(video_source)
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if process_this_frame:
            face_locations, face_names = process_frame(frame, known_face_encodings, known_face_names,
                                                       stranger_face_encodings, stranger_face_names, expected_name,
                                                       threshold)
            frame = draw_results(frame, face_locations, face_names)

        process_this_frame = not process_this_frame

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def calculate_metrics():
    global TP, FP, TN, FN, total_faces_detected, imposters_detected, real_faces_detected, y_true, y_scores

    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
    FAR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FRR = FN / (TP + FN) if (TP + FN) != 0 else 0
    FMR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FNMR = FN / (TP + FN) if (TP + FN) != 0 else 0

    print(f"Accuracy: {accuracy}")
    print(f"False Acceptance Rate (FAR): {FAR}")
    print(f"False Rejection Rate (FRR): {FRR}")
    print(f"False Match Rate (FMR): {FMR}")
    print(f"False Non-Match Rate (FNMR): {FNMR}")

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

    # Additional statistics
    print(f"Total faces detected: {total_faces_detected}")
    print(f"Imposters detected: {imposters_detected}")
    print(f"Real faces detected: {real_faces_detected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Use live video feed')
    parser.add_argument('--video_path', type=str, help='Path to recorded video')
    parser.add_argument('--reference_images', type=str, default='reference_faces', help='Path to reference images')
    parser.add_argument('--stranger_images', type=str, default='stranger_faces', help='Path to stranger images')
    args = parser.parse_args()

    known_face_encodings, known_face_names = load_faces(args.reference_images, KNOWN_CACHE_FILE)
    stranger_face_encodings, stranger_face_names = load_faces(args.stranger_images, STRANGER_CACHE_FILE)

    expected_name = "fabio"  # This could be parameterized as well
    threshold = THRESHOLD  # This could be parameterized as well

    if args.live:
        video_stream(0, known_face_encodings, known_face_names, stranger_face_encodings, stranger_face_names,
                     expected_name, threshold)
    elif args.video_path:
        video_stream(args.video_path, known_face_encodings, known_face_names, stranger_face_encodings,
                     stranger_face_names, expected_name, threshold)
    else:
        print("Please provide a video source: --live for live video or --video_path <path> for a recorded video")
        exit()

    calculate_metrics()


if __name__ == '__main__':
    main()
