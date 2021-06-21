import cv2

def sketchify(img, ksize=5):
    # Convert image to grayscale 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
    # Apply median filter to the grayscale image 
    img_gray = cv2.medianBlur(img_gray, 7) 
 
    # Detect edges in the image and threshold it 
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize) 
    ret, sketch = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV) 
 
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return sketch_bgr

def cartoonize(img) :
    num_iter = 1 # bilateral filter 수행 횟수
    ds_factor = 2 # down_sampling factor. 속도빠른 연산 위해
    sigma_color = 10
    sigma_space = 8
    # Resize the image to a smaller size for faster computation 
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor)
    for _ in range(num_iter): # Apply bilateral filter the image multiple times 
        img_small = cv2.bilateralFilter(img_small, 0, sigma_color, sigma_space) 
    
    img_bilateral = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor) 
 
    # Add the thick boundary lines to the image
    sketch_bgr = sketchify(img)
    sketch_01 = sketch_bgr / 255
    cartoon = img_bilateral * sketch_01
    cartoon = cartoon.astype('uint8')

    return cartoon 