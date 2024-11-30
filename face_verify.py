import cv2
from deepface import DeepFace
import numpy as np

from helpers import detect_spoofing, draw_spoofing


IMG1_PATH = '/home/wojtazk/Desktop/biometria_zdjecia/anakin_skywalker_animowany.jpg'
IMG2_PATH = '/home/wojtazk/Desktop/biometria_zdjecia/anakin_skywalker.png'

# border: top, bottom, left, right
border = (2000, 2000, 2000, 2000)


if __name__ == '__main__':
    result = DeepFace.verify(
        img1_path=IMG1_PATH,
        img2_path=IMG2_PATH,
    )
    print(result)

    ###########################################
    frame_img1 = cv2.imread(IMG1_PATH)
    frame_img2 = cv2.imread(IMG2_PATH)

    img1_dimensions = frame_img1.shape

    # detect spoofing
    spoofing_analysis_img1 = detect_spoofing(frame_img1)
    face1 = spoofing_analysis_img1[0]['face']

    spoofing_analysis_img2 = detect_spoofing(frame_img2)
    face2 = spoofing_analysis_img2[0]['face']

    # define window
    cv2.startWindowThread()
    window_id = 'Verification'

    cv2.namedWindow(window_id, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_id, 600, 600)

    # add padding to the frame
    frame_img1 = cv2.copyMakeBorder(frame_img1, *border, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    frame_img2 = cv2.copyMakeBorder(frame_img2, *(0, 100, 10, 100), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    # draw spoofing info and face analysis
    draw_spoofing(frame_img1, spoofing_analysis_img1, border)
    draw_spoofing(frame_img2, spoofing_analysis_img2, border=(0, 100, 10, 100))

    ########################################
    # draw images side by side
    frame = frame_img1

    # # draw original images side by side
    h_img1, w_img1, _ = img1_dimensions
    h_img2, w_img2, _ = frame_img2.shape

    frame[
        (border[0]):(border[0] + h_img2),
        (border[2] + w_img1):(border[2] + w_img1 + w_img2)
    ] = frame_img2

    ########################################
    # draw extracted faces side by side
    # convert faces to uint
    face1 = (face1 * 255).astype(np.uint8)
    face2 = (face2 * 255).astype(np.uint8)

    # convert from BGR to RGB
    face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
    face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)

    # face1 = cv2.copyMakeBorder(face1, *(0, 0, 10, 0), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    face2 = cv2.copyMakeBorder(face2, *(0, 0, 10, 0), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    h_f1, w_f1, _ = face1.shape
    h_f2, w_f2, _ = face2.shape

    frame[
        600:(600 + h_f1),
        600:(600 + w_f1)
    ] = face1

    frame[
        600:(600 + h_f2),
        (600 + w_f1):(600 + w_f1 + w_f2)
    ] = face2

    # add verification text
    text = f'Verified: {result['verified']}'
    cv2.putText(frame, text, (600 + 30, 600 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 153, 0), 2)

    cv2.imshow(window_id, frame)
    while True:
        # get pressed key
        pressed_key = cv2.waitKey(1) & 0xFF
        # Press 'q' to quit
        if pressed_key == ord('q'):
            break
        # quit when pressing the exit button
        if cv2.getWindowProperty(window_id, cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()
