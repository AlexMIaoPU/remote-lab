import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

'''
This module contains functions for processing images, including resistor horizontal alignment,
'''


def image_horitzontal_alignment(image):
    """
    This function takes a singled out resistor image as input, convert it to binary and uses
    method proposed by Yung-Sheng Chen and Jeng-Yau Wang. Reading resistor based on image processing. In
    2015 International Conference on Machine Learning and Cybernetics (ICMLC), 2015.

    by finding the line passing through the 
    region center of mass about the lowest moment of inertia. 
    """

    # Convert RGB to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Otsu's thresholding to get binary image
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get coordinates of foreground pixels
    y_indices, x_indices = np.nonzero(binary)
    A = len(x_indices)
    if A == 0:
        raise ValueError("No foreground pixels found.")

    Sx = np.sum(x_indices)
    Sy = np.sum(y_indices)
    Sxx = np.sum(x_indices ** 2)
    Syy = np.sum(y_indices ** 2)
    Sxy = np.sum(x_indices * y_indices)

    Mxy = Sxy - (Sx * Sy) / A
    Mxx = Sxx - (Sx ** 2) / A
    Myy = Syy - (Sy ** 2) / A

    theta = 0.5 * np.arctan2(2 * Mxy, Mxx - Myy)

    print("Theta:", theta)
    
    # Rotate the image to align horizontally and preserve all pixels
    (h, w) = image.shape[:2]
    angle_deg = np.degrees(theta)
    # Calculate the new bounding dimensions
    abs_cos = abs(np.cos(np.radians(angle_deg)))
    abs_sin = abs(np.sin(np.radians(angle_deg)))
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to account for translation
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot_mat[0, 2] += (new_w - w) / 2
    rot_mat[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return rotated

def crop_to_narrowest_y_range(image):
    """
    Finds the column where the vertical span (y_max - y_min) of non-background pixels is minimized,
    and crops the image to that y-range across the entire width.
    Returns the cropped image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    min_height = None
    best_y_min = 0
    best_y_max = h
    
    # Find the range of y pixels for the resistor body
    rows = np.any(gray > 0, axis=1)
    y_indices = np.where(rows)[0]
    if len(y_indices) == 0:
        raise ValueError("No foreground pixels found for cropping.")
    y_min, y_max = y_indices[0], y_indices[-1] + 1  # +1 for inclusive slicing

    height = y_max - y_min
    heigh_threshold = 0.5 * height

    for col in range(w):
        col_pixels = gray[:, col]
        y_indices = np.where(col_pixels > 0)[0]
        if len(y_indices) == 0:
            continue
        y_min, y_max = y_indices[0], y_indices[-1]
        height = y_max - y_min + 1
        if ((min_height is None) or (height < min_height)) and (height > heigh_threshold):
            min_height = height
            best_y_min = y_min
            best_y_max = y_max

    cropped = image[best_y_min:best_y_max+1, :, :]
    return cropped

def remove_resistor_body(image):
    """
    This function removes the resistor body colour by sampling the 4 corners of the image and 
    takes the average of the 4 as the resistor body colour. It then subtracts the whole image
    by this average colour and applies a thresholding to set all pixels to 0 or 1 with the resistor
    body colour set to 0 and the rest of the image to 1. 
    """
    # Get the average color of the 4 corners
    h, w = image.shape[:2]
    top_left = image[0, 0]
    top_right = image[0, w - 1]
    bottom_left = image[h - 1, 0]
    bottom_right = image[h - 1, w - 1]

    avg_color = np.mean([top_left, top_right, bottom_left, bottom_right], axis=0)

    print("Average color:", avg_color)

    # Subtract the average color from the image using numpy broadcasting
    subtracted_image = np.absolute(image.astype(np.float32) - avg_color.astype(np.float32))
    subtracted_image = subtracted_image.astype(np.uint8)

    # Convert to grayscale for thresholding
    #gray = cv2.cvtColor(subtracted_image, cv2.COLOR_RGB2GRAY)
    #_, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return subtracted_image.astype(np.uint8)



def lococate_colour_bands(binary_image):
    '''
    Extract the colour bands location from the binary image.
    
    '''

def image_histogram_equalization(image):
    """
    Convert the image to hsv color space and apply histogram equalization to the h channel.
    Then convert back to RGB color space.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # Apply histogram equalization to the L channel
    #h_channel_eq = cv2.equalizeHist(h_channel)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    h_channel_eq = clahe.apply(h_channel)

    # Merge the channels back
    hsv_eq = cv2.merge((h_channel_eq, s_channel, v_channel))

    # Convert back to RGB color space
    rgb_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

    return (hsv_eq, rgb_eq)

def remove_glare(image):

    #COLOR 
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=40.0,tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    lab = cv2.merge((l_eq, a, b))
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


    #INPAINT + CLAHE
    gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray , 210, 255, cv2.THRESH_BINARY)[1]

    # Mask refinement using morphological operations
    kernel = np.ones((3,3), np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.dilate(mask,kernel,iterations = 1)

    result2 = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    plt.imshow(mask)
    plt.axis('off')
    plt.show()

    return result2




def main():
    """
    Test function for image_horitzontal_alignment.
    Loads an example image, aligns it, and displays the result.
    """

    # Path to the test image (update if needed)
    img_path = r"c:/Users/Alex Miao/Documents/dev/remote-lab/res3.png"
    if not os.path.exists(img_path):
        print(f"Test image not found: {img_path}")
        return

    # Load image in RGB
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Failed to load image.")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Align and display
    #rotated = image_horitzontal_alignment(img_rgb)
    #cropped = crop_to_narrowest_y_range(rotated)
    #binary = remove_resistor_body(cropped)

    #binary = remove_resistor_body(img_rgb)

    # Apply histogram equalization
    _, image_rgb = image_histogram_equalization(img_rgb)

    remove_glare_image_bgr = remove_glare(img_bgr)
    
    plt.imshow(cv2.cvtColor(remove_glare_image_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    print("Image has been aligned and displayed.")

if __name__ == "__main__":
    main()

