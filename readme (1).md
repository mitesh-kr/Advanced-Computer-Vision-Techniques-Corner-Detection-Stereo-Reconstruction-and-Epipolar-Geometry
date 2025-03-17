# Advanced Computer Vision Techniques: Corner Detection, Stereo Reconstruction, and Epipolar Geometry

This repository contains implementations of three fundamental computer vision algorithms:

1. Harris Corner Detection
2. Stereo 3D Reconstruction
3. Epipolar Geometry Visualization

## Project Structure

```
.
├── README.md
├── .gitignore
├── harris_corner_detection.py    # Question 1: Harris Corner Detection
├── stereo_reconstruction.py      # Question 2: Stereo 3D Reconstruction
├── epipolar_geometry.py          # Question 3: Epipolar Lines Visualization
├── images/                       # Directory for test images (not included in repo)
│   ├── Question 1/               # Images for Harris Corner Detection
│   └── Question 2 and 3 Images/  # Images for Stereo Reconstruction and Epipolar Geometry
└── results/                      # Output directory for generated images
    ├── harris/                   # Harris corner detection results
    ├── stereo/                   # Stereo reconstruction results
    └── epipolar/                 # Epipolar geometry results
```

## Setup and Requirements

### Dependencies

- Python 3.7+
- NumPy
- OpenCV (cv2)
- Matplotlib
- PIL (Python Imaging Library)

Install dependencies using:

```bash
pip install numpy opencv-python matplotlib pillow
```

### Dataset

Download the dataset from the provided Google Drive link and place the images in the `images/` directory:
[Dataset Link](https://drive.google.com/drive/folders/1la4hwF_n4g7T25d2gyCF1ob3HJOir3Th?usp=sharing)

## Implementation Details

### 1. Harris Corner Detection

The implementation in `harris_corner_detection.py` performs corner detection using the Harris Corner Detection algorithm from scratch without using OpenCV's built-in functions for the core algorithm.

#### Features:
- Custom implementation of convolution operations
- Adjustable window size and threshold parameters
- Comparison with OpenCV's implementation
- Visualization of detected corners

#### Usage:
```bash
python harris_corner_detection.py
```

### 2. Stereo 3D Reconstruction

The implementation in `stereo_reconstruction.py` performs stereo reconstruction using a pair of stereo images.

#### Features:
- Block matching algorithm for disparity calculation
- Depth map generation
- 3D point cloud visualization
- Uses camera intrinsic matrices from the bike.txt file

#### Usage:
```bash
python stereo_reconstruction.py
```

### 3. Epipolar Geometry Visualization

The implementation in `epipolar_geometry.py` visualizes epipolar lines between two images taken from different viewpoints.

#### Features:
- SIFT feature detection and matching
- Epipolar line visualization using the fundamental matrix
- Sampling of points along epipolar lines
- Visualization of corresponding points between images

#### Usage:
```bash
python epipolar_geometry.py
```

## Results

### Harris Corner Detection

The algorithm detects corners in the input images and compares the results with OpenCV's implementation. Sample output images are saved in the `results/harris/` directory.

### Stereo 3D Reconstruction

The algorithm generates a disparity map, depth map, and 3D point cloud from the stereo image pair. Results are saved in the `results/stereo/` directory.

### Epipolar Geometry Visualization

The algorithm visualizes epipolar lines and corresponding points between two images. Results are saved in the `results/epipolar/` directory.

## Algorithm Details

### Harris Corner Detection

1. Compute x and y derivatives of the image using Sobel operators
2. Compute products of derivatives at each pixel
3. Apply Gaussian filtering to the products
4. Compute Harris response R = det(M) - k * trace(M)^2
5. Apply thresholding to identify corners

### Stereo 3D Reconstruction

1. Perform block matching to find correspondences between left and right images
2. Calculate disparity map based on pixel displacements
3. Convert disparity to depth using camera parameters
4. Generate 3D point cloud from depth map

### Epipolar Geometry Visualization

1. Extract SIFT features from both images
2. Match features using ratio test
3. Draw epipolar lines using the fundamental matrix
4. Sample points along epipolar lines
5. Visualize corresponding points between images

## References

- Harris, C., & Stephens, M. (1988). A combined corner and edge detector. In Alvey vision conference (Vol. 15, No. 50, pp. 10-5244).
- Hartley, R., & Zisserman, A. (2003). Multiple view geometry in computer vision. Cambridge university press.
- Szeliski, R. (2010). Computer vision: algorithms and applications. Springer Science & Business Media.
