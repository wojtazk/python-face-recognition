import cv2
from deepface import DeepFace
from helpers import detect_spoofing, draw_spoofing


IMG_PATH = '/home/wojtazk/Desktop/Screenshot_20241125_195446.png'
DB_PATH = '/home/wojtazk/Desktop/biometria_zdjecia'

# border: top, bottom, left, right
border = (1000, 1000, 1000, 1000)


if __name__ == '__main__':
    matches = DeepFace.find(
        img_path=IMG_PATH,
        db_path=DB_PATH,
    )

    match_dict = matches[0].to_dict()
    print(match_dict)

    img_paths = []
    for i in range(len(match_dict['identity'])):
        img_paths.append(match_dict['identity'][i])

    print(img_paths)

    #################################################
    # show original
    # noinspection DuplicatedCode
    frame = cv2.imread(IMG_PATH)
    spoofing_analysis = detect_spoofing(frame)
    print()
    for face in spoofing_analysis:
        del face['face']
        print(face)

    cv2.startWindowThread()
    window_id = f'ORIGINAL - {IMG_PATH}'

    cv2.namedWindow(window_id, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_id, 600, 600)

    # add padding to the frame
    frame = cv2.copyMakeBorder(frame, *border, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    # draw spoofing info and face analysis
    draw_spoofing(frame, spoofing_analysis, border)

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

    #################################################
    # show matches
    for image in img_paths:
        # noinspection DuplicatedCode
        frame = cv2.imread(image)

        # detect spoofing
        spoofing_analysis = detect_spoofing(frame)
        print()
        for face in spoofing_analysis:
            del face['face']
            print(face)

        # define window
        cv2.startWindowThread()
        window_id = f'MATCH - {image}'

        cv2.namedWindow(window_id, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_id, 600, 600)

        # add padding to the frame
        frame = cv2.copyMakeBorder(frame, *border, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

        # draw spoofing info and face analysis
        draw_spoofing(frame, spoofing_analysis, border)

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
