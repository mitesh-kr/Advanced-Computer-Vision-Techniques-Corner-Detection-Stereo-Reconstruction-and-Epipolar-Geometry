#!/usr/bin/env python3
"""
Inference script for running computer vision algorithms on custom images.
This script provides a command-line interface to run any of the three implemented
computer vision techniques on user-provided images.
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from harris_corner_detection import load_images as load_harris_images, harris_corner_detection
from stereo_reconstruction import block_matching, generate_point_cloud
from epipolar_geometry import load_images, detect_features, feature_match, draw_epipolar_lines, sample_points_along_epipolar_line, display_results

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_harris_corner_detection(args):
    """Run Harris Corner Detection on input images"""
    print("Running Harris Corner Detection...")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, "harris")
    ensure_directory(output_dir)
    
    # Load images
    if os.path.isdir(args.input):
        image_folder = args.input
        file_names, original_images, gray_images = load_harris_images(image_folder)
        
        for i, (file_name, original_image, gray_image) in enumerate(zip(file_names, original_images, gray_images)):
            print(f"Processing image: {file_name}")
            
            # Custom implementation
            print("Using custom implementation...")
            custom_result = harris_corner_detection(original_image.copy(), gray_image)
            
            # OpenCV implementation for comparison
            print("Using OpenCV implementation...")
            opencv_result = original_image.copy()
            dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            opencv_result[dst > 0.01 * dst.max()] = [0, 0, 255]
            
            # Save results
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_custom.jpg"), custom_result)
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_opencv.jpg"), opencv_result)
            
            # Compare results side by side
            comparison = np.hstack((custom_result, opencv_result))
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_comparison.jpg"), comparison)
            
    else:
        print(f"Error: Input path {args.input} is not a directory. Harris Corner Detection requires a directory of images.")
        return
    
    print(f"Results saved to {output_dir}")

def run_stereo_reconstruction(args):
    """Run Stereo 3D Reconstruction on input stereo image pair"""
    print("Running Stereo 3D Reconstruction...")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, "stereo")
    ensure_directory(output_dir)
    
    # Check if we have a left and right image
    if args.right_image is None:
        print("Error: Stereo reconstruction requires both left and right images.")
        print("Use --right_image to specify the right stereo image.")
        return
    
    # Load images
    I1 = cv2.imread(args.input)
    I2 = cv2.imread(args.right_image)
    
    if I1 is None or I2 is None:
        print("Error: Could not load input images.")
        return
    
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    
    # Use default camera matrices if not provided
    K1 = np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]])
    K2 = np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])
    baseline = 177.288  # in mm
    
    # Compute disparity map
    print("Computing disparity map...")
    disparity_map = block_matching(I1_gray, I2_gray, block_size=args.block_size, max_disparity=args.max_disparity)
    
    # Save disparity map
    plt.figure(figsize=(10, 8))
    plt.imshow(disparity_map, cmap='jet')
    plt.colorbar(label='Disparity')
    plt.title('Disparity Map')
    plt.savefig(os.path.join(output_dir, "disparity_map.png"))
    
    # Compute depth map
    print("Computing depth map...")
    depth_map = (K1[0][0] * baseline/1000) / (disparity_map + 0.000001)
    
    # Save depth map
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Depth (m)')
    plt.title('Depth Map')
    plt.savefig(os.path.join(output_dir, "depth_map.png"))
    
    # Generate point cloud
    print("Generating 3D point cloud...")
    point_cloud = generate_point_cloud(depth_map, K1[0][0])
    
    # Save point cloud visualization
    X = point_cloud[:,:,0].flatten()
    Y = point_cloud[:,:,1].flatten()
    Z = point_cloud[:,:,2].flatten()
    
    # Filter out extreme values
    mask = (Z > 0) & (Z < 100)
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use a stride to reduce number of points for faster rendering
    stride = max(1, len(X) // 10000)
    ax.scatter(X[::stride], Y[::stride], Z[::stride], s=1, c=Z[::stride], cmap='viridis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')
    plt.savefig(os.path.join(output_dir, "point_cloud.png"))
    
    print(f"Results saved to {output_dir}")

def run_epipolar_geometry(args):
    """Run Epipolar Geometry Visualization on input image pair"""
    print("Running Epipolar Geometry Visualization...")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, "epipolar")
    ensure_directory(output_dir)
    
    # Check if we have a second image
    if args.right_image is None:
        print("Error: Epipolar geometry requires both images.")
        print("Use --right_image to specify the second image.")
        return
    
    # Use default fundamental matrix if not provided
    F = np.array([[3.34638533e-07, 7.58547151e-06, -2.04147752e-03],
                 [-5.83765868e-06, 1.36498636e-06, 2.67566877e-04],
                 [1.45892349e-03, -4.37648316e-03, 1.00000000e+00]])
    
    # Load images
    img1, img2, img1_gray, img2_gray = load_images(args.input, args.right_image)
    
    # Detect features
    print("Detecting features...")
    keypoints1, descriptors1, keypoints2, descriptors2 = detect_features(img1_gray, img2_gray)
    print(f"Found {len(keypoints1)} keypoints in left image and {len(keypoints2)} in right image")
    
    # Match features
    print("Matching features...")
    matches = feature_match(descriptors1, descriptors2)
    print(f"Found {len(matches)} good matches")
    
    if len(matches) == 0:
        print("Error: No good matches found between images.")
        return
    
    # Draw epipolar lines
    print("Drawing epipolar lines...")
    img1_with_lines, img2_with_lines = draw_epipolar_lines(
        img1.copy(), img2.copy(), keypoints1, keypoints2, matches, F
    )
    
    # Sample points along epipolar lines
    print("Sampling points along epipolar lines...")
    img1_with_line, img2_with_points, img2_with_line, img1_with_points = sample_points_along_epipolar_line(
        img1.copy(), img2.copy(), keypoints1, keypoints2, matches, F, num_points=args.num_points
    )
    
    # Save results
    cv2.imwrite(os.path.join(output_dir, "left_epipolar_lines.jpg"), img1_with_lines)
    cv2.imwrite(os.path.join(output_dir, "right_epipolar_lines.jpg"), img2_with_lines)
    cv2.imwrite(os.path.join(output_dir, "left_sampled_points.jpg"), img1_with_line)
    cv2.imwrite(os.path.join(output_dir, "right_corresponding_points.jpg"), img2_with_points)
    cv2.imwrite(os.path.join(output_dir, "right_sampled_points.jpg"), img2_with_line)
    cv2.imwrite(os.path.join(output_dir, "left_corresponding_points.jpg"), img1_with_points)
    
    # Create combined visualization
    fig = plt.figure(figsize=(20, 12))
    
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
    plt.savefig(os.path.join(output_dir, "epipolar_visualization.png"))
    
    print(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run computer vision algorithms on custom images")
    
    parser.add_argument("algorithm", type=str, choices=["harris", "stereo", "epipolar"],
                        help="Which algorithm to run (harris, stereo, or epipolar)")
    
    parser.add_argument("input", type=str,
                        help="Input image path (or directory for harris)")
    
    parser.add_argument("--right_image", type=str, default=None,
                        help="Right image path (required for stereo and epipolar)")
    
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results (default: results)")
    
    # Harris Corner Detection parameters
    parser.add_argument("--window_size", type=int, default=3,
                        help="Window size for Harris Corner Detection (default: 3)")
    
    parser.add_argument("--k", type=float, default=0.04,
                        help="k parameter for Harris Corner Detection (default: 0.04)")
    
    # Stereo Reconstruction parameters
    parser.add_argument("--block_size", type=int, default=15,
                        help="Block size for stereo matching (default: 15)")
    
    parser.add_argument("--max_disparity", type=int, default=128,
                        help="Maximum disparity for stereo matching (default: 128)")
    
    # Epipolar Geometry parameters
    parser.add_argument("--num_points", type=int, default=10,
                        help="Number of points to sample along epipolar lines (default: 10)")
    
    args = parser.parse_args()
    
    # Run the selected algorithm
    if args.algorithm == "harris":
        run_harris_corner_detection(args)
    elif args.algorithm == "stereo":
        run_stereo_reconstruction(args)
    elif args.algorithm == "epipolar":
        run_epipolar_geometry(args)

if __name__ == "__main__":
    main()
