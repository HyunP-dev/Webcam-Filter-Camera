import cv2

# github:gachonyws/face-mask-detector
facenet = cv2.dnn.readNet('face_detector/deploy.prototxt',
                          'face_detector/res10_300x300_ssd_iter_140000.caffemodel')


def detect_face(image) -> list:
    """
    Detect faces from image

    Args:
        image: image

    Returns:
        list: series of [(x1, y1), (x2, y2)]
    """
    blob = cv2.dnn.blobFromImage(image, size=(300, 300))
    facenet.setInput(blob)
    detections = facenet.forward()
    h, w = image.shape[:2]
    result = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue
        else:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            result.append([(x1, y1), (x2, y2)])
    return result


def get_eyes_from_landmark(landmark):
    if landmark.num_parts is not 68:
        return None
    left = [landmark.part(i) for i in range(36, 42)]
    right = [landmark.part(i) for i in range(42, 48)]
    return [left, right]


def get_eye_aspect_ratio(eye):
    def d(p1, p2):
        return ((p1.x-p2.x)**2 + (p1.y-p2.y)**2)**0.5
    return (d(eye[1], eye[5])+d(eye[2], eye[5]))/(2*d(eye[0]-eye[3]))