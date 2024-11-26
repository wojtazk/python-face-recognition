import cv2
from deepface import DeepFace

from helpers import opencv_decorator_camera, models

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


# loading model into memory
DeepFace.build_model(models[0])


def analyze_face_camera(frame=None) -> None:
    if frame is None:
        return

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
        cv2.putText(frame, text, (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 153, 0), 2)

        # display anti spoof score
        text = f"Real face: {is_real} ({anti_spoof_score * 100:.2f}%)"
        cv2.putText(frame, text, (x + 10, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 153, 0), 2)

    # analyze face
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
            cv2.putText(frame, text, (x + 10, y + h + 90 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 153, 0), 2)
            j += 1


if __name__ == '__main__':
    opencv_decorator_camera(analyze_face_camera)
