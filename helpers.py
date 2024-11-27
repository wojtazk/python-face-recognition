import cv2
from deepface import DeepFace

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


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
model = DeepFace.build_model(models[0])


def detect_spoofing(frame):
    spoofing_analysis = DeepFace.extract_faces(
        frame,
        anti_spoofing=True,
        enforce_detection=False,
    )

    return spoofing_analysis


def draw_spoofing(frame, spoofing_analysis, border=(0, 0, 0, 0)):
    for i in range(len(spoofing_analysis)):
        # face = extracted_faces[i]['face']
        facial_area = spoofing_analysis[i]['facial_area']
        face_confidence = spoofing_analysis[i]['confidence']

        is_real = spoofing_analysis[i]['is_real']
        anti_spoof_score = spoofing_analysis[i]['antispoof_score']

        # get region with detected face
        x = facial_area['x'] + border[2]
        y = facial_area['y'] + border[0]
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
        text_y += 30
        cv2.putText(frame, text, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 153, 0), 2)


def analyze_faces(frame):
    face_analysis = DeepFace.analyze(
        frame,
        actions=["emotion", "age", "gender", "race"],
        enforce_detection=False
    )

    return face_analysis


def draw_face_analysis(frame, face_analysis, border=(0, 0, 0, 0)):
    for i in range(len(face_analysis)):
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
        text_x = x + 10 + border[2]
        for text in texts:
            text_y = y + h + 90 + 30 * j + border[0]

            # shadow
            cv2.putText(frame, text, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
            # text
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 153, 0), 2)

            j += 1
