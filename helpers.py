import cv2


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


def opencv_decorator(func):
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

            func(frame)  # call passed function

            while True:
                cv2.imshow(window_name, frame)

                # Press 'q' to stop showing the prediction
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

    return func
