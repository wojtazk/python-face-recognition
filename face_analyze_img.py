import cv2
import os

from helpers import detect_spoofing, analyze_faces, draw_spoofing, draw_face_analysis


IMG_DIRECTORY = '/home/wojtazk/Desktop/biometria_zdjecia'
images = sorted(IMG_DIRECTORY + '/' + file for file in os.listdir(IMG_DIRECTORY))

# border: top, bottom, left, right
border = (1000, 1000, 1000, 1000)


if __name__ == '__main__':
    for image in images:
        frame = cv2.imread(image)

        # frame = cv2.resize(frame, (640, 480))  # resize the frame

        # detect spoofing
        spoofing_analysis = detect_spoofing(frame)
        print()
        for face in spoofing_analysis:
            del face['face']
            print(face)

        # analyze faces
        face_analysis = analyze_faces(frame)
        for face in face_analysis:
            print(face)

        # define window
        cv2.startWindowThread()
        window_id = str(image)

        cv2.namedWindow(window_id, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_id, 600, 600)

        # add padding to the frame
        frame = cv2.copyMakeBorder(frame, *border, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

        # draw spoofing info and face analysis
        draw_spoofing(frame, spoofing_analysis, border)
        draw_face_analysis(frame, face_analysis, border)
        cv2.imshow(window_id, frame)

    while True:
        # get pressed key
        pressed_key = cv2.waitKey(1) & 0xFF
        # Press 'q' to quit
        if pressed_key == ord('q'):
            break
        # quit when pressing the exit button
        # if cv2.getWindowProperty(window_id, cv2.WND_PROP_VISIBLE) < 1:
        #     break

    cv2.destroyAllWindows()
