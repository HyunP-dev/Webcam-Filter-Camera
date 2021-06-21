import cv2
import numpy as np

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def paste_image(img, to, location):
    """
    Args:
        img: 이미지
        to: img위에 덮을 투명 이미지.
        location: 덮을 위치.
    """
    result = img.copy()
    
    mask_img = np.zeros_like(to, dtype='uint8')
    mask_img[np.where(to[:,:,:]!=0)]=255
    mask_img = cv2.bitwise_not(mask_img)

    orig_rows, orig_cols, orig_channels = img.shape
    mask_rows, mask_cols, mask_channels = mask_img.shape
    loc_x, loc_y = location
    
    temp = result[loc_y:loc_y+mask_rows, loc_x:loc_x+mask_cols, :]
    temp = cv2.bitwise_and(temp, mask_img)
    temp = cv2.add(temp, to)
    result[loc_y:loc_y+mask_rows, loc_x:loc_x+mask_cols, :] = temp
    return result