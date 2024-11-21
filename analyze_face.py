import cv2
from deepface import DeepFace

from helpers import opencv_decorator, models


# loading model into memory
DeepFace.build_model(models[0])


def analyze_face(frame = None) -> None:
    if frame is None:
        return

    extracted_faces = DeepFace.extract_faces(
        frame,
        anti_spoofing=True
    )
    print(extracted_faces[0])

    face = extracted_faces[0]['face']
    facial_area = extracted_faces[0]['facial_area']
    face_confidence = extracted_faces[0]['confidence']

    is_real = extracted_faces[0]['is_real']
    anti_spoof_score = extracted_faces[0]['antispoof_score']

    # get region with detected face
    x = facial_area['x']
    y = facial_area['y']
    w = facial_area['w']
    h = facial_area['h']

    # draw rectangle around the face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 153, 0), 2)

    # display face confidence
    text = f"Face confidence: {face_confidence * 100:.0f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 153, 0), 2)

    # display anti spoof score
    text = f"Real face: {is_real}, confidence: {anti_spoof_score * 100:.2f}%"
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 153, 0), 2)

    # analyze face
    face_analysis = DeepFace.analyze(
        frame,
        actions=["emotion", "age", "gender", "race"],
        enforce_detection=False
    )[0]
    print(face_analysis)

    emotion = face_analysis['dominant_emotion']
    age = int(face_analysis['age'])
    gender = face_analysis['dominant_gender']
    race = face_analysis['dominant_race']

    # display analysis info
    text = f"Emotion: {emotion}, Age: {age}, Gender: {gender}, Race: {race}"
    cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 255, 102), 2)


if __name__ == '__main__':
    opencv_decorator(analyze_face)