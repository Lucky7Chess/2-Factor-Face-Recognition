import face_recognition

# Loads jpg files into numpy arrays
user_image = face_recognition.load_image_file("barack_A.jpg")
webcam_image = face_recognition.load_image_file("barack_B.jpg")

# Only compares first "found" face in each image
user_face_encoding = face_recognition.face_encodings(user_image)[0]
webcam_face_encoding = face_recognition.face_encodings(webcam_image)[0]

results = face_recognition.compare_faces([webcam_face_encoding], user_face_encoding)

print(results[0])