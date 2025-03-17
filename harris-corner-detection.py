"""
Harris Corner Detection Implementation

This script implements the Harris Corner Detection algorithm from scratch
and compares it with OpenCV's implementation.
"""

import cv2
import os
import numpy as np
from PIL import Image

def load_images(image_folder_path):
    """
    Load images from a folder and convert them to grayscale.
    
    Parameters:
    image_folder_path (str): Path to the folder containing images
    
    Returns:
    tuple: (image_file_names, original_images, gray_images)
    """
    image_file_names = sorted(os.listdir(image_folder_path))
    original_images = []
    gray_images = []
    for image_name in image_file_names:
        image_path = os.path.join(image_folder_path, image_name)

        # Check if the image is a GIF
        if image_name.lower().endswith('.gif'):
            gif_image = Image.open(image_path)
            gif_rgb = gif_image.convert('RGB')
            new_image_path = os.path.splitext(image_path)[0] + '.jpg'
            gif_rgb.save(new_image_path)
            image = cv2.imread(new_image_path)
        else:
            image = cv2.imread(image_path)

        original_images.append(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray_image)

    return image_file_names, original_images, gray_images


def padding(image):
    """
    Add zero padding around the image.
    
    Parameters:
    image (ndarray): Input image
    
    Returns:
    ndarray: Padded image
    """
    padded_image = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.float32)
    padded_image[1:-1, 1:-1] = image
    return padded_image


def convolution(padded_image, kernel):
    """
    Perform convolution operation on a padded image with a given kernel.
    
    Parameters:
    padded_image (ndarray): Padded input image
    kernel (ndarray): Convolution kernel
    
    Returns:
    ndarray: Convolved image
    """
    I = np.zeros((padded_image.shape[0] - 2, padded_image.shape[1] - 2))
    for i in range(padded_image.shape[0] - 2):
        for j in range(padded_image.shape[1] - 2):
            area = padded_image[i:i + 3, j:j + 3]
            I[i, j] = np.sum(area * kernel)
    return I


def harris_corner_detection(original_image, gray_image, k=0.04, threshold_factor=0.01):
    """
    Implement Harris Corner Detection algorithm from scratch.
    
    Parameters:
    original_image (ndarray): Original color image
    gray_image (ndarray): Grayscale image
    k (float): Harris detector free parameter
    threshold_factor (float): Factor to determine threshold from max response
    
    Returns:
    ndarray: Image with highlighted corners
    """
    image = original_image.copy()
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = kernel_x.T
    window = np.ones((3, 3)) / 9

    padded_image = padding(gray_image)

    Ix = convolution(padded_image, kernel_x)
    Iy = convolution(padded_image, kernel_y)

    Ixx = Ix**2
    Ixy = Ix * Iy
    Iyy = Iy**2

    Ixx = padding(Ixx)
    Ixy = padding(Ixy)
    Iyy = padding(Iyy)

    Sxx = convolution(Ixx, window)
    Sxy = convolution(Ixy, window)
    Syy = convolution(Iyy, window)

    det_M = Sxx * Syy - Sxy**2
    trace_M = Sxx + Syy
    R = det_M - k * trace_M**2

    threshold = threshold_factor * R.max()
    image[R > threshold] = [0, 0, 255]
    
    return image


def display_images(image1, image2, title1, title2):
    """
    Display two images side by side for comparison.
    
    Parameters:
    image1 (ndarray): First image
    image2 (ndarray): Second image
    title1 (str): Title for the first image
    title2 (str): Title for the second image
    """
    result = np.hstack((image1, image2))
    cv2.imshow(f"{title1} vs {title2}", result)
    cv2.waitKey(0)


def main():
    """Main function to run Harris Corner Detection"""
    # Replace with your image folder path
    image_folder_path = "./images/Question 1"
    
    print("Loading images...")
    file_names, original_images, gray_images = load_images(image_folder_path)
    
    for i, (file_name, original_image, gray_image) in enumerate(
        zip(file_names, original_images, gray_images)):
        if i == 0:  # Skip the first image if it's not part of the test set
            continue
            
        print(f"Processing image: {file_name}")
        
        # Custom Harris Corner Detection
        print('Using scratch implementation...')
        custom_result = harris_corner_detection(original_image.copy(), gray_image)
        
        # OpenCV Harris Corner Detection
        print('Using OpenCV implementation...')
        opencv_result = original_image.copy()
        dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        opencv_result[dst > 0.01 * dst.max()] = [0, 0, 255]
        
        # Display or save results
        cv2.imwrite(f"harris_custom_{file_name}", custom_result)
        cv2.imwrite(f"harris_opencv_{file_name}", opencv_result)
        
        print(f"Results saved for {file_name}")
        print('-' * 80)
        
        # Uncomment to display images (if running with GUI)
        # display_images(custom_result, opencv_result, "Custom Implementation", "OpenCV Implementation")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
