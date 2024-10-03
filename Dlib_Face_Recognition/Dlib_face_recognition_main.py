import dlib
import cv2
import numpy as np
import os

# Load the models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('resources/dlib_face_recognition_resnet_model_v1.dat')

# Function to get face encodings
def get_face_encoding(image):
    dets = detector(image, 1)
    if len(dets) == 0:
        return None
    shape = predictor(image, dets[0])
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

# Load known faces and their encodings
def load_known_faces(directory):
    known_encodings = {}
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(person_dir, filename)
                    image = cv2.imread(img_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    encoding = get_face_encoding(image_rgb)
                    if encoding is not None:
                        if person_name not in known_encodings:
                            known_encodings[person_name] = []
                        known_encodings[person_name].append(encoding)
                    else:
                        print(f"No face detected in {filename}.")
    return known_encodings

# Function to compare encodings
def is_match(unknown_encoding, known_encodings, tolerance=0.6):
    for name, encodings in known_encodings.items():
        for known_encoding in encodings:
            distance = np.linalg.norm(known_encoding - unknown_encoding)
            if distance < tolerance:
                return name  # Return the name of the matched face
    return None  # No match found

# Function to save unknown faces
def save_unknown_faces(images, name, save_directory):
    person_dir = os.path.join(save_directory, name)
    os.makedirs(person_dir, exist_ok=True)
    for i, image in enumerate(images):
        image_filename = os.path.join(person_dir, f"{name}_{i + 1}.jpg")
        cv2.imwrite(image_filename, image)
        print(f"Saved unknown face as: {image_filename}")

# Load known faces
known_faces_directory = "known_faces"  # Change to your directory containing known faces
known_encodings = load_known_faces(known_faces_directory)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Process the frame for face recognition
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    unknown_encoding = get_face_encoding(image_rgb)

    if unknown_encoding is not None:
        # Check if it matches any known faces
        matched_name = is_match(unknown_encoding, known_encodings)
        if matched_name:
            cv2.putText(frame, f"Match found: {matched_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(matched_name, "Detected")
        else:
            cv2.putText(frame, "No match found.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Ask user if they want to save the unknown face
            cv2.putText(frame, "Press 's' to save or 'q' to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Webcam", frame)
            

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # If 's' key is pressed
                person_name = input("Enter the name for the new person: ").strip()
                images_to_save = []
                
                # Capture multiple images
                num_images = 30  # Number of images to capture
                print(f"Capturing {num_images} images for {person_name}.")
                for _ in range(num_images):
                    ret, capture_frame = cap.read()
                    if ret:
                        images_to_save.append(capture_frame)
                        cv2.putText(capture_frame, f"Capturing image {_ + 1}/{num_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Capturing Images", capture_frame)
                        cv2.waitKey(1000)  # Capture an image every second
                    else:
                        print("Failed to grab frame during capture.")

                save_unknown_faces(images_to_save, person_name, known_faces_directory)
            elif key == ord('q'):  # If 'q' key is pressed
                break
    else:
        cv2.putText(frame, "No face detected.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the webcam feed
    
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
