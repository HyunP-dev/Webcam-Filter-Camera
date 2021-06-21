import cv2
import numpy as np
import dlib

sunglass = cv2.imread("images/sunglasses.png")
ears_orig = cv2.imread("images/6570766_preview.png")

predictor = dlib.shape_predictor("landmarks/shape_predictor_68_face_landmarks.dat")

def image_to_mask(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
    return cv2.merge((mask, mask, mask))

def put_on_sunglass(image, face, transparency=.3):
    sunglass_height, sunglass_weight = sunglass.shape[:2]

    right_lateral_canthus_x = np.int(sunglass_weight*.15)
    left_lateral_canthus_x = np.int(sunglass_weight*.85)
    lateral_canthus_y = np.int(sunglass_height*.4)

    lateral_canthus_src = [(right_lateral_canthus_x, lateral_canthus_y),
                           (left_lateral_canthus_x, lateral_canthus_y)]

    landmarks = predictor(image, face)
    lateral_canthus_dst = [(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(45).x, landmarks.part(45).y)]
    
    xform = cv2.estimateAffinePartial2D(np.array([lateral_canthus_src]), np.array([lateral_canthus_dst]))[0]
    transformed_sunglass = cv2.warpAffine(sunglass, xform, image.shape[:2][::-1])
    transformed_sunglass_mask = cv2.warpAffine(image_to_mask(sunglass), xform, image.shape[:2][::-1])
    transformed_sunglass = (transformed_sunglass&image_to_mask(transformed_sunglass))

    return ((image&~transformed_sunglass_mask)+((1-transparency)*transformed_sunglass+(transparency)*(image&transformed_sunglass_mask)).astype('uint8'))


def get_euclidean_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5


def draw_landmark_number(photo, landmark):
    result = photo.copy()
    for i in range(landmark.num_parts):
        x = landmark.part(i).x
        y = landmark.part(i).y

        cv2.putText(result, str(i), (x, y),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
    return result


def determine_ears_size(landmark):
    p1 = (landmark.part(0).x, landmark.part(0).y)
    p2 = (landmark.part(16).x, landmark.part(16).y)
    distance = int(get_euclidean_distance(p1, p2))

    ratio = distance / ears_orig.shape[1]
    dim = (distance, int(ears_orig.shape[0] * ratio))
    return dim


from math import sin, radians

def determine_ears_location(landmark, ears, rotation):
    p1 = (landmark.part(0).x, landmark.part(0).y)
    p2 = (landmark.part(16).x, landmark.part(16).y)

    if rotation >=0:
        loc_x = p1[0]-int(ears.shape[1]*sin(radians(rotation)))
    else:
        loc_x = p1[0]-int(0.5*ears.shape[1]*sin(radians(rotation)))

    if rotation >= 0:
        loc_y = p1[1]-ears.shape[1]
    else:
        loc_y = p2[1]-ears.shape[1]

    return (loc_x, loc_y)

from toolkit.measurement import get_rotation
from toolkit.transformation import rotate_image, paste_image

def put_on_neko_ears(frame, face):
    ears = ears_orig.copy()

    # Extract Landmarks
    landmark = predictor(frame, face)

    # Resize
    resized_ears = cv2.resize(ears, determine_ears_size(landmark),
                              interpolation=cv2.INTER_NEAREST)

    # Get angle of rotation
    rot = get_rotation(frame, face)

    # Rotate
    modified_ears = rotate_image(resized_ears, rot)

    # Determine ears location
    loc = determine_ears_location(landmark, modified_ears, rot)

    # Composing images
    return paste_image(frame, modified_ears, loc)