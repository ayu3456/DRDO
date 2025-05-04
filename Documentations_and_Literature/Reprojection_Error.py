import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Simulate 3D object points (e.g., checkerboard corners)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Step 2: Simulate corresponding 2D image points (with slight noise)
# Assume a simple pinhole camera projection (without distortion for simplicity)
fx, fy = 800, 800
cx, cy = 320, 240
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)

rvec = np.array([[0.1], [0.2], [0.1]])  # rotation vector
# Change the data type to np.float32 to make sure all elements are floats.
tvec = np.array([[0], [0], [5]], dtype=np.float32)        # translation vector

# Project 3D points to 2D using known camera parameters
image_points, _ = cv2.projectPoints(objp, rvec, tvec, K, None)

# Simulate detected points (with some error)
detected_points = image_points + np.random.normal(0, 0.5, image_points.shape)

# Step 3: Compute reprojection error
reprojection_error = np.linalg.norm(image_points - detected_points, axis=2).mean()
print("Mean Reprojection Error (pixels):", reprojection_error)

# Step 4: Visualize
plt.figure(figsize=(6, 6))
plt.scatter(image_points[:, 0, 0], image_points[:, 0, 1], label='Projected Points')
plt.scatter(detected_points[:, 0, 0], detected_points[:, 0, 1], label='Detected Points')
for i in range(len(image_points)):
    plt.plot([image_points[i, 0, 0], detected_points[i, 0, 0]],
             [image_points[i, 0, 1], detected_points[i, 0, 1]], 'r-', alpha=0.5)
plt.legend()
plt.title("Reprojection Error Visualization")
plt.gca().invert_yaxis()
plt.grid()
plt.show()
