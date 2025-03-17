import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_images(left_img_path, right_img_path):
    """
    Load stereo image pair and convert to grayscale
    
    Parameters:
    left_img_path (str): Path to the left image
    right_img_path (str): Path to the right image
    
    Returns:
    tuple: Original and grayscale versions of both images
    """
    img1 = cv2.imread(left_img_path)
    img2 = cv2.imread(right_img_path)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not load images at {left_img_path} or {right_img_path}")
        
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    return img1, img2, img1_gray, img2_gray

def detect_features(img1_gray, img2_gray):
    """
    Detect SIFT features in both images
    
    Parameters:
    img1_gray (ndarray): Grayscale left image
    img2_gray (ndarray): Grayscale right image
    
    Returns:
    tuple: Keypoints and descriptors for both images
    """
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)
    
    return keypoints1, descriptors1, keypoints2, descriptors2

def feature_match(descriptors1, descriptors2, ratio_threshold=0.75):
    """
    Match features using ratio test (Lowe's algorithm)
    
    Parameters:
    descriptors1 (ndarray): Feature descriptors from first image
    descriptors2 (ndarray): Feature descriptors from second image
    ratio_threshold (float): Threshold for ratio test (default: 0.75)
    
    Returns:
    list: List of tuples containing indices of matching features
    """
    matches = []
    for i, descriptor1 in enumerate(descriptors1):
        best_match_idx = -1
        second_best_match_idx = -1
        best_distance = float('inf')
        second_best_distance = float('inf')

        for j, descriptor2 in enumerate(descriptors2):
            distance = np.sqrt(np.sum((descriptor1 - descriptor2) ** 2))
            if distance < best_distance:
                second_best_distance = best_distance
                second_best_match_idx = best_match_idx
                best_distance = distance
                best_match_idx = j
            elif distance < second_best_distance:
                second_best_distance = distance
                second_best_match_idx = j

        if best_distance < ratio_threshold * second_best_distance:
            matches.append((i, best_match_idx))

    return matches

def draw_epipolar_lines(img1, img2, keypoints1, keypoints2, matches, F):
    """
    Draw epipolar lines on both images based on feature matches
    
    Parameters:
    img1 (ndarray): Left image
    img2 (ndarray): Right image
    keypoints1 (list): Keypoints from left image
    keypoints2 (list): Keypoints from right image
    matches (list): List of matched feature indices
    F (ndarray): Fundamental matrix
    
    Returns:
    tuple: Images with epipolar lines drawn
    """
    img1_with_lines = img1.copy()
    img2_with_lines = img2.copy()
    
    for match in matches:
        idx1, idx2 = match
        pt1 = keypoints1[idx1].pt
        pt2 = keypoints2[idx2].pt
        pt1_homo = np.array([pt1[0], pt1[1], 1]).reshape(3, 1)
        pt2_homo = np.array([pt2[0], pt2[1], 1]).reshape(3, 1)

        # Compute epipolar lines
        line_in_img1 = np.dot(F.T, pt2_homo).reshape(3)
        line_in_img2 = np.dot(F, pt1_homo).reshape(3)

        # Draw epipolar lines on images
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Line in image 1
        height, width = img1.shape[:2]
        x0, y0 = 0, int(-line_in_img1[2] / line_in_img1[1])
        x1, y1 = width, int(-(line_in_img1[0] * width + line_in_img1[2]) / line_in_img1[1])
        img1_with_lines = cv2.line(img1_with_lines, (x0, y0), (x1, y1), color, 1)

        # Line in image 2
        height, width = img2.shape[:2]
        x0, y0 = 0, int(-line_in_img2[2] / line_in_img2[1])
        x1, y1 = width, int(-(line_in_img2[0] * width + line_in_img2[2]) / line_in_img2[1])
        img2_with_lines = cv2.line(img2_with_lines, (x0, y0), (x1, y1), color, 1)
    
    return img1_with_lines, img2_with_lines

def sample_points_along_epipolar_line(img1, img2, keypoints1, keypoints2, matches, F, num_points=10):
    """
    Sample points along epipolar lines and visualize correspondences
    
    Parameters:
    img1 (ndarray): Left image
    img2 (ndarray): Right image
    keypoints1 (list): Keypoints from left image
    keypoints2 (list): Keypoints from right image
    matches (list): List of matched feature indices
    F (ndarray): Fundamental matrix
    num_points (int): Number of points to sample along each line
    
    Returns:
    tuple: Four images showing points on epipolar lines
    """
    # Select one match
    match = matches[0]
    idx1, idx2 = match
    pt1 = keypoints1[idx1].pt
    pt2 = keypoints2[idx2].pt
    pt1_homo = np.array([pt1[0], pt1[1], 1]).reshape(3, 1)
    pt2_homo = np.array([pt2[0], pt2[1], 1]).reshape(3, 1)

    # Part 1: Points on line in image 1
    line_in_img1 = np.dot(F.T, pt2_homo).reshape(3)
    height, width = img1.shape[:2]
    x0, y0 = 0, int(-line_in_img1[2] / line_in_img1[1])
    x1, y1 = width, int(-(line_in_img1[0] * width + line_in_img1[2]) / line_in_img1[1])
    
    img1_with_line = img1.copy()
    img1_with_line = cv2.line(img1_with_line, (x0, y0), (x1, y1), (0, 255, 255), 1)
    
    # Calculate points uniformly along the line on img1
    x_coords = np.linspace(x0, x1, num_points)
    y_coords = np.linspace(y0, y1, num_points)
    points_on_line_img1 = np.column_stack((x_coords.astype(int), y_coords.astype(int)))
    
    for point in points_on_line_img1:
        cv2.circle(img1_with_line, point, radius=4, color=(0, 255, 0), thickness=-1)
    
    # Compute corresponding epipolar lines on img2
    epipolar_lines_img2 = np.dot(F, np.vstack((points_on_line_img1.T, np.ones(num_points))))
    
    # Compute intersection points on img2
    img2_with_points = img2.copy()
    points_on_line_img2 = []
    
    for i in range(num_points):
        line = epipolar_lines_img2[:, i]
        x_max = img2.shape[1]
        y_max = int(-(line[0] * x_max + line[2]) / line[1])
        y_max = min(max(0, y_max), img2.shape[0] - 1)
        points_on_line_img2.append((x_max, y_max))
        cv2.circle(img2_with_points, (x_max, y_max), radius=10, color=(0, 0, 255), thickness=-1)
    
    # Part 2: Points on line in image 2
    line_in_img2 = np.dot(F, pt1_homo).reshape(3)
    height, width = img2.shape[:2]
    x0, y0 = 0, int(-line_in_img2[2] / line_in_img2[1])
    x1, y1 = width, int(-(line_in_img2[0] * width + line_in_img2[2]) / line_in_img2[1])
    
    img2_with_line = img2.copy()
    img2_with_line = cv2.line(img2_with_line, (x0, y0), (x1, y1), (0, 255, 255), 1)
    
    # Calculate points uniformly along the line on img2
    x_coords = np.linspace(x0, x1, num_points)
    y_coords = np.linspace(y0, y1, num_points)
    points_on_line_img2_sample = np.column_stack((x_coords.astype(int), y_coords.astype(int)))
    
    for point in points_on_line_img2_sample:
        cv2.circle(img2_with_line, point, radius=4, color=(0, 255, 0), thickness=-1)
    
    # Compute corresponding epipolar lines on img1
    epipolar_lines_img1 = np.dot(F.T, np.vstack((points_on_line_img2_sample.T, np.ones(num_points))))
    
    # Compute intersection points on img1
    img1_with_points = img1.copy()
    points_on_line_img1_sample = []
    
    for i in range(num_points):
        line = epipolar_lines_img1[:, i]
        x_max = img1.shape[1]
        y_max = int(-(line[0] * x_max + line[2]) / line[1])
        y_max = min(max(0, y_max), img1.shape[0] - 1)
        points_on_line_img1_sample.append((x_max, y_max))
        cv2.circle(img1_with_points, (x_max, y_max), radius=10, color=(0, 0, 255), thickness=-1)
    
    return img1_with_line, img2_with_points, img2_with_line, img1_with_points

def save_results(output_dir, *images_with_names):
    """
    Save result images to output directory
    
    Parameters:
    output_dir (str): Directory to save output images
    *images_with_names: Variable number of (image, name) tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img, name in images_with_names:
        output_path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(output_path, img)
        print(f"Saved {output_path}")

def display_results(img1_with_lines, img2_with_lines, 
                   img1_with_line, img2_with_points,
                   img2_with_line, img1_with_points):
    """
    Display all result images using matplotlib
    
    Parameters:
    Various result images from previous functions
    """
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Epipolar Lines on Left Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Epipolar Lines on Right Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(img1_with_line, cv2.COLOR_BGR2RGB))
    plt.title('Sampled Points on Left Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(img2_with_points, cv2.COLOR_BGR2RGB))
    plt.title('Corresponding Points on Right Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(img2_with_line, cv2.COLOR_BGR2RGB))
    plt.title('Sampled Points on Right Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(img1_with_points, cv2.COLOR_BGR2RGB))
    plt.title('Corresponding Points on Left Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Define paths
    left_img_path = 'images/Question 2 and 3 Images/000000.png'
    right_img_path = 'images/Question 2 and 3 Images/000023.png'
    output_dir = 'results/epipolar'
    
    # Define fundamental matrix
    F = np.array([[3.34638533e-07, 7.58547151e-06, -2.04147752e-03],
                 [-5.83765868e-06, 1.36498636e-06, 2.67566877e-04],
                 [1.45892349e-03, -4.37648316e-03, 1.00000000e+00]])
    
    # Load images
    print("Loading images...")
    img1, img2, img1_gray, img2_gray = load_images(left_img_path, right_img_path)
    
    # Detect features
    print("Detecting features...")
    keypoints1, descriptors1, keypoints2, descriptors2 = detect_features(img1_gray, img2_gray)
    print(f"Found {len(keypoints1)} keypoints in left image and {len(keypoints2)} in right image")
    
    # Match features
    print("Matching features...")
    matches = feature_match(descriptors1, descriptors2)
    print(f"Found {len(matches)} good matches")
    
    # Draw epipolar