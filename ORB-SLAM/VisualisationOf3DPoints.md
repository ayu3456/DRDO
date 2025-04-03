# Visualization of 3D points

## **Steps to Visualize 3D Points**

      1️⃣Capture **two images** from different angles.

2️⃣ Detect and match **feature points**.

3️⃣ Estimate **camera pose** using the **essential matrix**.

4️⃣ Perform **triangulation** to recover 3D points.

5️⃣ **Plot** the 3D points using **Matplotlib**.

## **Updated Code with 3D Visualization**

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images
img1 = cv2.imread("view1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("view2.jpg", cv2.IMREAD_GRAYSCALE)

# ORB Feature Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Feature Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract Matched Points
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Camera Intrinsics (Assumed)
K = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]])

# Compute the Essential Matrix
E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Recover Camera Pose (R, t)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# Projection Matrices for Both Cameras
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # First Camera
P2 = np.hstack((R, t))  # Second Camera

# Triangulate Points
points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)

# Convert Homogeneous 4D to 3D
points_3D = points_4D[:3] / points_4D[3]

# Plot 3D Points
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(points_3D[0], points_3D[1], points_3D[2], c="blue", marker="o")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("3D Points from Triangulation")

plt.show()
```

## **Expected Output**

- A **3D scatter plot** displaying the **reconstructed points** from the two images.
- The points represent the **3D structure of the scene**.