from PIL import Image
import numpy as np
from google.colab.patches import cv2_imshow


def extract_instance(item_mask, im):
    # Get the true bounding box of the mask (not the same as the bbox prediction)
    segmentation = np.where(item_mask == True)
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))
    print(x_min, x_max, y_min, y_max)

    # Create a cropped image from just the portion of the image we want
    cropped = Image.fromarray(im[y_min:y_max, x_min:x_max, :], mode='RGB')

    # Create a PIL image out of the mask
    mask = Image.fromarray((item_mask * 255).astype('uint8'))

    # Crop the mask to match the cropped image
    cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

    # Load in a background image and choose a paste position
    background = Image.new('RGB', cropped.size)
    #paste_position = (int(cropped.size[0]/2), int(cropped.size[1]/2))

    paste_position = (0, 0)

    # Create a new foreground image as large as the composite and paste the cropped image on top
    new_fg_image = Image.new('RGB', cropped.size)
    new_fg_image.paste(cropped, paste_position)

    # Create a new alpha mask as large as the composite and paste the cropped mask
    new_alpha_mask = Image.new('L', cropped.size, color = 0)
    new_alpha_mask.paste(cropped_mask, paste_position)

    # Compose the foreground and background using the alpha mask
    composite = Image.composite(new_fg_image, background, new_alpha_mask)

    # Display the image
    cv2_imshow(np.array(composite))

    return composite