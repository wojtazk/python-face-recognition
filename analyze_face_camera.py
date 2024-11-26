from helpers import opencv_decorator_camera, detect_spoofing, analyze_faces

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


def analyze_face_camera(frame=None) -> None:
    if frame is None:
        return

    # detect spoofing, draw rectangle and spoofing info
    detect_spoofing(frame)

    # analyze faces and draw results
    analyze_faces(frame)


if __name__ == '__main__':
    opencv_decorator_camera(analyze_face_camera)
