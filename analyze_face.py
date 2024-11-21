import cv2
from deepface import DeepFace

cv2.startWindowThread()
cap = cv2.VideoCapture(0)  # 0 - initialize video capture from the default camera

window_name = 'Hello There'


while True:
    ret, frame = cap.read()  # read the frame

    # frame = cv2.resize(frame, (640, 480))  # resize the frame
    frame = cv2.flip(frame, 1)  # flip the frame horizontally

    # show the camera output
    cv2.imshow(window_name, frame)

    # get pressed key
    pressed_key = cv2.waitKey(1) & 0xFF
    # Press 'q' to quit
    if pressed_key == ord('q'):
        break
    # quit when pressing the exit button
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    # Press 'f' to pay respect
    if pressed_key == ord('f'):
        result = DeepFace.analyze(frame, actions=["emotion", "age", "gender", "race"], enforce_detection=False, )[0]

        emotion = result['dominant_emotion']
        age = int(result['age'])
        gender = result['dominant_gender']
        race = result['dominant_race']

        text = f"Emotion: {emotion}, Age: {age}, Gender: {gender}, Race: {race}"

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 255, 102), 2)

        while True:
            cv2.imshow(window_name, frame)

            # get pressed key
            pressed_key = cv2.waitKey(1) & 0xFF
            # Press 'q' to quit
            if pressed_key == ord('q'):
                break
            # quit when pressing the exit button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

cap.release()
cv2.destroyAllWindows()
