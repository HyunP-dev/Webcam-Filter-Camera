import dlib
import math

faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(
    "landmarks/shape_predictor_5_face_landmarks.dat")


def get_rotation(img, face=None):
    landmarks = landmarkDetector(img,
                                 faceDetector(img, 0)[0] if face == None else face)
    rightEye = landmarks.part(2)
    leftEye = landmarks.part(0)
    deg = math.atan((leftEye.y - rightEye.y) / (leftEye.x - rightEye.x))
    return -deg*180/math.pi
