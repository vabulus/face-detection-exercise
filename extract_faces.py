# import cv2
# import face_recognition
# import os
#
# # Load your reference image and create an encoding
# reference_image_path = "f_reference_image.png"
# reference_image = face_recognition.load_image_file(reference_image_path)
# reference_encoding = face_recognition.face_encodings(reference_image)[0]
#
#
# def extract_and_save_my_face(image_path, output_dir, reference_encoding, tolerance=0.7):
#     # Load the image
#     image = face_recognition.load_image_file(image_path)
#     # Find all face locations and encodings in the image
#     face_locations = face_recognition.face_locations(image)
#     face_encodings = face_recognition.face_encodings(image, face_locations)
#
#     # Load the image with OpenCV for saving purposes
#     image_cv = cv2.imread(image_path)
#
#     # Check if no faces were found in the image
#     if not face_encodings:
#         print(f"No faces found in {image_path}")
#         return
#
#     # Loop through each face found in the image
#     for i, (top, right, bottom, left) in enumerate(face_locations):
#         face_encoding = face_encodings[i]
#
#         # Compare the face encoding to the reference encoding
#         matches = face_recognition.compare_faces([reference_encoding], face_encoding, tolerance=tolerance)
#         if matches[0]:
#             # Extract the face
#             face_image = image_cv[top:bottom, left:right]
#
#             # Save the face as a new file
#             face_filename = os.path.join(output_dir,
#                                          f"{os.path.splitext(os.path.basename(image_path))[0]}_my_face_{i}.jpg")
#             cv2.imwrite(face_filename, face_image)
#             print(f"Extracted and saved my face from {image_path} as {face_filename}")
#         else:
#             print(f"Face in {image_path} did not match the reference image (index {i})")
#
#
# def process_images(input_dir, output_dir, reference_encoding, tolerance=0.7):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     print(f"Listing all files in directory: {input_dir}")
#     all_files = os.listdir(input_dir)
#     print(f"Found {len(all_files)} files")
#
#     for filename in all_files:
#         if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(
#                 ".JPG"):
#             image_path = os.path.join(input_dir, filename)
#             print(f"Processing {image_path}")
#             extract_and_save_my_face(image_path, output_dir, reference_encoding, tolerance)
#
#
# input_images_directory = "f_faces_raw"
# output_faces_directory = "f_faces_extracted"
#
# process_images(input_images_directory, output_faces_directory, reference_encoding)
