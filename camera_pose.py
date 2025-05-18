import cv2
import numpy as np
from feature_detection import detect_features
import os

def estimate_camera_pose(img1, img2, K):
    # Detect features and get matched points
    pts1, pts2, _ = detect_features(img1, img2)
    
    # Find Essential Matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover pose
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    
    return R, t, mask

if __name__ == "__main__":
    # Load images
    img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        exit()
    
    # Camera matrix (example values - should be calibrated for real use)
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])
    
    # Estimate camera pose
    R, t, mask = estimate_camera_pose(img1, img2, K)
    
    # Save results to file
    os.makedirs('results', exist_ok=True)
    with open('results/camera_pose.txt', 'w') as f:
        f.write('Rotation Matrix:\n')
        f.write(str(R) + '\n')
        f.write('\nTranslation Vector:\n')
        f.write(str(t) + '\n')
        f.write(f"\nNumber of inliers: {np.sum(mask)}\n")
    
    print("Rotation Matrix:")
    print(R)
    print("\nTranslation Vector:")
    print(t)
    print(f"\nNumber of inliers: {np.sum(mask)} (saved to results/camera_pose.txt)") 