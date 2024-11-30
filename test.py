import cv2
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Load image
image = cv2.imread("test2.jpg")
height, width, _ = image.shape
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform facial landmark detection
result = face_mesh.process(rgb_image)

# Get the landmarks for the eyes
if result.multi_face_landmarks:
    for face_landmarks in result.multi_face_landmarks:
        # Left eye (landmarks 33 to 133)
        left_eye_x = [int(landmark.x * width) for i, landmark in enumerate(face_landmarks.landmark) if i in range(33, 134)]
        left_eye_y = [int(landmark.y * height) for i, landmark in enumerate(face_landmarks.landmark) if i in range(33, 134)]
        
        # Bounding box for left eye
        left_x_min = min(left_eye_x)
        left_x_max = max(left_eye_x)
        left_y_min = min(left_eye_y)
        left_y_max = max(left_eye_y)

        # Draw bounding box
        cv2.rectangle(image, (left_x_min, left_y_min), (left_x_max, left_y_max), (255, 0, 0), 2)

        # Similarly, you can extract the right eye (landmarks 133 to 362)
        
# Save the image with bounding boxes
cv2.imwrite("mediapipe_eye_labeled.jpg", image)
