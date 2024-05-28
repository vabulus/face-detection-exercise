import argparse
import os
import pickle
import face_recognition
import numpy as np

# Constants
KNOWN_CACHE_FILE = "known_face_encodings_cache.pkl"
STRANGER_CACHE_FILE = "stranger_face_encodings_cache.pkl"

# Metrics
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


def compute_similarity_scores(base_encoding, known_face_encodings, stranger_face_encodings):
    global y_true, y_scores

    for known_encoding in known_face_encodings:
        distance = np.linalg.norm(base_encoding - known_encoding)
        y_true.append(1)
        y_scores.append(distance)

    for stranger_encoding in stranger_face_encodings:
        distance = np.linalg.norm(base_encoding - stranger_encoding)
        y_true.append(0)
        y_scores.append(distance)


def determine_optimal_threshold(y_true, y_scores):
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_threshold = 0
    best_score = 0
    best_metrics = (0, 0, 0, 0)  # (TP, TN, FP, FN)

    for threshold in thresholds:
        y_pred = [1 if score < threshold else 0 for score in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        if accuracy > best_score:
            best_score = accuracy
            best_threshold = threshold
            best_metrics = (tp, tn, fp, fn)

    return best_threshold, best_score, best_metrics


def load_base_image(base_image_path):
    image = face_recognition.load_image_file(base_image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0]
    else:
        print(f"No face found in the base image: {base_image_path}")
        exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_images', type=str, default='known_faces', help='Path to reference images')
    parser.add_argument('--stranger_images', type=str, default='stranger_faces', help='Path to stranger images')
    parser.add_argument('--base_image', type=str, required=True, help='Path to the base image for comparison')
    args = parser.parse_args()

    base_encoding = load_base_image(args.base_image)
    known_face_encodings, known_face_names = load_faces(args.reference_images, KNOWN_CACHE_FILE)
    stranger_face_encodings, stranger_face_names = load_faces(args.stranger_images, STRANGER_CACHE_FILE)

    compute_similarity_scores(base_encoding, known_face_encodings, stranger_face_encodings)
    optimal_threshold, best_score, best_metrics = determine_optimal_threshold(y_true, y_scores)

    tp, tn, fp, fn = best_metrics
    print(f"Determined Optimal Threshold: {optimal_threshold} with Accuracy: {best_score}")
    print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")


if __name__ == '__main__':
    main()
