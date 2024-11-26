import cv2
from deepface import DeepFace


models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet",
]


def detect_spoofing(frame):
    extracted_faces = DeepFace.extract_faces(
        frame,
        anti_spoofing=True,
        enforce_detection=False,
    )

    for i in range(len(extracted_faces)):
        print(extracted_faces[i])

        # face = extracted_faces[i]['face']
        facial_area = extracted_faces[i]['facial_area']
        face_confidence = extracted_faces[i]['confidence']

        is_real = extracted_faces[i]['is_real']
        anti_spoof_score = extracted_faces[i]['antispoof_score']

        # get region with detected face
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']

        # draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 153, 0), 2)

        # display face confidence
        text = f"Face confidence: {face_confidence * 100:.0f}%"
        text_x = x + 10
        text_y = y + h + 30
        cv2.putText(frame, text, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 153, 0), 2)

        # display anti spoof score
        text = f"Real face: {is_real} ({anti_spoof_score * 100:.2f}%)"
        text_y = y + h + 60
        cv2.putText(frame, text, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 153, 0), 2)


def analyze_faces(frame):
    face_analysis = DeepFace.analyze(
        frame,
        actions=["emotion", "age", "gender", "race"],
        enforce_detection=False
    )

    for i in range(len(face_analysis)):
        print(face_analysis[i])

        emotion = face_analysis[i]['dominant_emotion']
        age = int(face_analysis[i]['age'])
        gender = face_analysis[i]['dominant_gender']
        race = face_analysis[i]['dominant_race']

        facial_area = face_analysis[i]['region']

        # get region with detected face
        x = facial_area['x']
        y = facial_area['y']
        # w = facial_area['w']
        h = facial_area['h']

        # # display analysis info
        texts = f"Emotion: {emotion}\nAge: {age}\nGender: {gender}\nRace: {race}".split('\n')
        j = 0
        for text in texts:
            text_x = x + 10
            text_y = y + h + 90 + 30 * j

            # shadow
            cv2.putText(frame, text, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
            # text
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 153, 0), 2)

            j += 1
