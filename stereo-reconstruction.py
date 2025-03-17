"""
Stereo 3D Reconstruction Implementation

This script implements stereo 3D reconstruction to find the disparity map,
depth map, and 3D point cloud representation of a scene.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_stereo_images(left_image_path, right_image_path):
    """
    Load stereo image pair and convert to grayscale.
    
    Parameters:
    left_image_path (str): Path to left stereo image
    right_image_path (str): Path to right stereo image
    
    Returns:
    tuple: (left_color, right_color, left_gray, right_gray)
    """
    left_color = cv2.imread(left_image_path)
    right_color = cv2.imread(right_image_path)
    left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
    
    return left_color, right_color, left_gray, right_gray


def block_matching(img_left, img_right, block_size=3, max_disparity=128):
    """
    Compute disparity map using block matching algorithm.
    
    Parameters:
    img_left (ndarray): Left grayscale image
    img_right (ndarray): Right grayscale image
    block_size (int): Size of the matching block
    max_disparity (int): Maximum disparity value to search for
    
    Returns:
    ndarray: Disparity map
    """
    height, width = img_left.shape
    half_block_size = block_size // 2
    disparity_map = np.zeros_like(img_left, dtype=np.float32)

    for y in range(half_block_size, height - half_block_size):
        for x in range(half_block_size, width - half_block_size - max_disparity):
            template = img_left[y - half_block_size:y + half_block_size + 1, 
                               x - half_block_size:x + half_block_size + 1]
            best_match_score = float('inf')
            best_disparity = 0

            for d in range(max_disparity):
                if x - d - half_block_size < 0:
                    break
                target = img_right[y - half_block_size:y + half_block_size + 1, 
                                  x - d - half_block_size:x - d + half_block_size + 1]
                if target.shape != template.shape:
                    break
                score = np.sum((template - target) ** 2)
                if score < best_match_score:
                    best_match_score = score
                    best_disparity = d

            disparity_map[y, x] = best_disparity

    return disparity_map


def compute_depth_map(disparity_map, focal_length, baseline):
    """
    Compute depth map from disparity map.
    
    Parameters:
    disparity_map (ndarray): Disparity map
    focal_length (float): Focal length of the camera in pixels
    baseline (float): Baseline distance between cameras in mm
    
    Returns:
    ndarray: Depth map
    """
    # Avoid division by zero
    disparity_map_safe = disparity_map.copy()
    disparity_map_safe[disparity_map_safe == 0] = 0.000001
    
    # Convert baseline to meters for depth in meters
    baseline_meters = baseline / 1000.0
    
    # Compute depth map
    depth_map = (focal_length * baseline_meters) / disparity_map_safe
    
    return depth_map


def generate_point_cloud(depth_map, focal_length):
    """
    Generate 3D point cloud from depth map.
    
    Parameters:
    depth_map (ndarray): Depth map
    focal_length (float): Focal length of the camera in pixels
    
    Returns:
    ndarray: 3D point cloud
    """
    height, width = depth_map.shape
    point_cloud = np.zeros((height, width, 3))
    
    for y in range(height):
        for x in range(width):
            depth = depth_map[y, x]
            X = (x - width / 2) * depth / focal_length
            Y = (y - height / 2) * depth / focal_length
            Z = depth
            point_cloud[y, x] = [X, Y, Z]
    
    return point_cloud


def visualize_point_cloud(point_cloud):
    """
    Visualize 3D point cloud.
    
    Parameters:
    point_cloud (ndarray): 3D point cloud
    """
    X = point_cloud[:, :, 0].flatten()
    Y = point_cloud[:, :, 1].flatten()
    Z = point_cloud[:, :, 2].flatten()
    
    # Create mask for valid points (remove very distant points)
    mask = (Z < 100) & (Z > 0)
    
    # Plot the point cloud
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[mask], Y[mask], Z[mask], s=0.5, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Point Cloud')
    plt.savefig('point_cloud.png')
    plt.show()


def visualize_results(disparity_map, depth_map):
    """
    Visualize disparity map and depth map.
    
    Parameters:
    disparity_map (ndarray): Disparity map
    depth_map (ndarray): Depth map
    """
    plt.figure(figsize=(12, 6))
    
    # Normalize disparity map for better visualization
    disparity_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply colormap to disparity map
    disparity_color = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
    
    # Normalize depth map for better visualization
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply colormap to depth map
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    # Plot disparity map
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB))
    plt.title('Disparity Map')
    plt.axis('off')
    
    # Plot depth map
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
    plt.title('Depth Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('disparity_depth_maps.png')
    plt.show()


def main():
    """Main function to run stereo reconstruction"""
    # Define image paths and camera parameters
    left_image_path = "./images/Question 2 and 3 Images/bikeL.png"
    right_image_path = "./images/Question 2 and 3 Images/bikeR.png"
    
    # Camera intrinsic parameters
    K1 = np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]])
    K2 = np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])
    baseline = 177.288  # in mm
    
    # Load stereo images
    print("Loading stereo images...")
    left_color, right_color, left_gray, right_gray = load_stereo_images(left_image_path, right_image_path)
    
    # Compute disparity map
    print("Computing disparity map...")
    disparity_map = block_matching(left_gray, right_gray, block_size=3, max_disparity=128)
    
    # Compute depth map
    print("Computing depth map...")
    depth_map = compute_depth_map(disparity_map, K1[0][0], baseline)
    
    # Visualize disparity and depth maps
    print("Visualizing disparity and depth maps...")
    visualize_results(disparity_map, depth_map)
    
    # Generate and visualize 3D point cloud
    print("Generating 3D point cloud...")
    point_cloud = generate_point_cloud(depth_map, K1[0][0])
    
    print("Visualizing 3D point cloud...")
    visualize_point_cloud(point_cloud)
    
    print("Stereo reconstruction completed!")


if __name__ == "__main__":
    main()
