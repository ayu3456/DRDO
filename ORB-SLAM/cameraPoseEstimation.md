# Camera Pose Estimation

### **What is Camera Pose Estimation?**

Camera Pose Estimation determines the **position** and **orientation** of the camera in 3D space relative to the scene. Essentially, it answers:

📸 **"Where is the camera located, and where is it looking?"**

### **Why is it Important?**

- It allows us to **track camera motion** in SLAM.
- Helps in reconstructing **3D scenes** from multiple 2D images.
- Needed for **AR applications**, **robotics**, and **autonomous navigation**.

## **How Does Camera Pose Estimation Work?**

### **1️⃣ Find Corresponding Features**

- First, detect and match keypoints (which you've already done).

### **2️⃣ Estimate the Essential Matrix**

- The **Essential Matrix (E)** encodes the camera motion (rotation + translation) given two images from a calibrated camera.

### **3️⃣ Decompose the Essential Matrix**

- Extract **Rotation (R)** and **Translation (t)** to determine the **camera pose**.

### **4️⃣ Validate the Pose**

- Ensure points follow the correct geometry (using triangulation).

## **Let's Implement It!**

Now, let’s write a Python script to estimate camera pose from two images. You'll need OpenCV.

### **Step 1: Install Dependencies**

```bash
pip install opencv-python numpy
```

### **Step 2: Write the Code**

Create a new Python file: **`camera_pose_estimation.py`**

```python
import cv2
import numpy as np

# Load two consecutive images
img1 = cv2.imread("image1.jpg", 0)
img2 = cv2.imread("image2.jpg", 0)

# Detect ORB keypoints and descriptors
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)  # Sort by quality

# Extract matched keypoints
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Camera intrinsic matrix (Assuming fx=fy=700, cx=cy=320)
K = np.array([[700, 0, 320],
              [0, 700, 240],
              [0, 0, 1]])

# Estimate the Essential Matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Recover Pose (Rotation & Translation)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

print("Rotation Matrix (R): \n", R)
print("Translation Vector (t): \n", t)

```

---

## **How to Run?**

1. **Take two images** of the same scene with a small change in position (like you did before).
2. **Save them as** `image1.jpg` and `image2.jpg`.
3. **Run the script:**
    
    ```bash
    python3 camera_pose_estimation.py
    ```
    

---

## **What Does This Do?**

✅ Detects features

✅ Matches keypoints

✅ Estimates the **Essential Matrix**

✅ Extracts **Rotation (R)** and **Translation (t)**

🔹 The **rotation (R)** tells us how the camera **rotated**.

🔹 The **translation (t)** tells us how the camera **moved** in 3D space.

---