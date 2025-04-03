# **Triangulation in 3D Vision**

## **What is Triangulation?**

Triangulation is the process of finding the **3D position** of a point using its **projections in multiple images**. If you have two images of the same scene taken from different positions, you can compute the depth of key points.

### **Where is it Used?**

      ✅ **SLAM (Simultaneous Localization and Mapping)**

✅ **3D Reconstruction** (Photogrammetry, AR, VR)

✅ **Robot Navigation**

✅ **Autonomous Vehicles (Depth Estimation)**

## **How Does It Work? (Mathematical Approach)**

### **🔹 Given:**

- **Two camera matrices** P1P_1P1 and P2P_2P2
- **Two image points** x1x_1x1 and x2x_2x2
    
    ![Screenshot 2025-04-03 at 12.34.13 PM.png](attachment:32c76e05-978d-4f99-ad7b-3c0927207617:Screenshot_2025-04-03_at_12.34.13_PM.png)
    

## **Types of Triangulation Methods**

### **🔹 1. Linear Least Squares Triangulation**

- Constructs an **over-determined** system and solves using **Singular Value Decomposition (SVD).**
- Works well with **small errors** in feature detection.

### **🔹 2. Midpoint Method**

- Finds the **closest point** between two **epipolar lines** in space.
- Simple but **not very accurate**.

### **🔹 3. Nonlinear Triangulation (Optimal Method)**

- Uses **iterative optimization** (e.g., **Levenberg-Marquardt**) to **minimize reprojection error**.
- Most accurate but **computationally expensive**.

## **Implementation in OpenCV (Python Code)**

We will triangulate a **3D point** using **two images and their camera poses**.

### **🔹 Steps:**

      1️⃣ Capture **two images** of the same scene from different angles.

2️⃣ Detect **feature points** (e.g., using ORB).

3️⃣ Match keypoints **between two images**.

4️⃣ Compute the **essential matrix** to estimate **camera pose**.

5️⃣ Use **triangulation** to recover **3D coordinates**.

## **Code: Triangulation in OpenCV**

```python
import cv2
import numpy as np

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

# Print 3D Coordinates of First 5 Points
print(points_3D.T[:5])

```

## **Summary**

- **Triangulation** is used to **convert 2D image points into 3D world coordinates**.
- It requires **camera poses, feature points, and stereo images**.
- **Linear & Nonlinear methods** exist (OpenCV uses **Linear**).
- Used in **3D reconstruction, SLAM, and autonomous navigation**.