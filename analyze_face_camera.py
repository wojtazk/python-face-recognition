import cv2

from helpers import detect_spoofing, analyze_faces

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


if __name__ == '__main__':
    cv2.startWindowThread()
    cap = cv2.VideoCapture(0)  # 0 - initialize video capture from the default camera

    window_name = 'Hello There'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 600, 600)

    while True:
        ret, frame = cap.read()  # read the frame from the camera

        # frame = cv2.resize(frame, (640, 480))  # resize the frame
        frame = cv2.flip(frame, 1)  # flip the frame horizontally

        # pad the frame
        # frame = cv2.copyMakeBorder(frame, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(0, 0, 0))

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

            # detect spoofing, draw rectangle and spoofing info
            detect_spoofing(frame)

            # add padding to the frame
            frame = cv2.copyMakeBorder(frame, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

            # analyze faces and draw results
            analyze_faces(frame)

            while True:
                cv2.imshow(window_name, frame)

                # Press 'q' to stop showing the prediction
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
