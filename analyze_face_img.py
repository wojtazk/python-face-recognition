import cv2
import uuid

from helpers import detect_spoofing, analyze_faces

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

images = [
    '/home/wojtazk/Desktop/obi-wan_1.jpg',
    '/home/wojtazk/Desktop/stary_obi_wan.jpg',
    '/home/wojtazk/Desktop/obi_wan_animowany.png',
    '/home/wojtazk/Desktop/lego_obi_wan.jpg',
]


if __name__ == '__main__':
    for image in images:
        frame = cv2.imread(image)

        # frame = cv2.resize(frame, (640, 480))  # resize the frame

        # detect spoofing, draw rectangle and spoofing info
        detect_spoofing(frame)

        # analyze faces and draw results
        analyze_faces(frame)

        # define window
        cv2.startWindowThread()
        window_id = str(uuid.uuid4())

        cv2.namedWindow(window_id, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_id, 600, 600)

        # add padding to the frame
        frame = cv2.copyMakeBorder(frame, 1000, 1000, 1000, 1000, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
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
