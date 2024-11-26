import cv2
from deepface import DeepFace

import uuid

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


img_path = '/home/wojtazk/Desktop/obi-wan_1.jpg'

frame = cv2.imread(img_path)

# frame = cv2.resize(frame, (640, 480))  # resize the frame

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
        text_x = x + 10
        text_y = y + h + 90 + 30 * j

        # shadow
        cv2.putText(frame, text, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
        # text
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 153, 0), 2)

        j += 1


# spawn window
window_id = str(uuid.uuid4())
cv2.namedWindow(window_id, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# interpolate the image
frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
# add padding to the frame
frame = cv2.copyMakeBorder(frame, 2000, 2000, 2000, 2000, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
while True:
    cv2.imshow(window_id, frame)
    # get pressed key
    pressed_key = cv2.waitKey(1) & 0xFF
    # Press 'q' to quit
    if pressed_key == ord('q'):
        break
    # quit when pressing the exit button
    if cv2.getWindowProperty(window_id, cv2.WND_PROP_VISIBLE) < 1:
        break
