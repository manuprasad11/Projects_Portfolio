import cv2

def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    # Check if cascade loaded
    if face_classifier.empty():
        print("Error loading cascade classifier.")
        return

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error accessing webcam.")
        return

    id = 2
    img_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        face = face_cropped(frame)
        if face is not None:
            img_id += 10
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped face", face)

        if cv2.waitKey(1) == 13 or img_id == 100:  # Press Enter to stop
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting complete")

generate_dataset()
