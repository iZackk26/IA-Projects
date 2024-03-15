import os
import cv2

input_directory = "train/spoof"
output_directory = "output_spoof/"

# Crear el directorio de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def detect_and_crop_faces(image_path, output_directory):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Read the input image
    img = cv2.imread(image_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        # Crop the face region
        face_cropped = img[y : y + h, x : x + w]
        # Save the cropped face
        output_path = os.path.join(
            output_directory, f"face_{i}_{os.path.basename(image_path)}"
        )
        cv2.imwrite(output_path, face_cropped)



def process_images(input_directory, output_directory):
    # Enumerate all image files in the input directory
    image_files = [
        f
        for f in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, f))
    ]
    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(input_directory, image_file)
        # Detect and crop faces in the current image
        detect_and_crop_faces(image_path, output_directory)


# Procesar las imÃ¡genes del directorio de entrada y guardar las caras detectadas en el directorio de salida
process_images(input_directory, output_directory)
