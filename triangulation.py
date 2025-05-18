import cv2
import numpy as np
from feature_detection import detect_features
from camera_pose import estimate_camera_pose
import os

def triangulate_points(pts1, pts2, K, R, t):
    # Projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera
    P2 = K @ np.hstack((R, t))  # Second camera
    
    # Triangulate points (reshape to (2, N))
    pts1_flat = pts1.reshape(-1, 2).T
    pts2_flat = pts2.reshape(-1, 2).T
    points_4d = cv2.triangulatePoints(P1, P2, pts1_flat, pts2_flat)
    
    # Convert to 3D points and flatten
    points_3d = (points_4d[:3] / points_4d[3]).T  # shape (N, 3)
    
    return points_3d

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
    
    # Detect features and get matched points
    pts1, pts2, _ = detect_features(img1, img2)
    
    # Estimate camera pose
    R, t, mask = estimate_camera_pose(img1, img2, K)
    
    # Triangulate points
    points_3d = triangulate_points(pts1, pts2, K, R, t)
    
    # Save results to file
    os.makedirs('results', exist_ok=True)
    with open('results/triangulation.txt', 'w') as f:
        f.write(f"3D Points shape: {points_3d.shape}\n")
        f.write("\nFirst 5 3D points:\n")
        f.write(str(points_3d[:5]) + '\n')
    
    print("3D Points shape:", points_3d.shape)
    print("\nFirst 5 3D points:")
    print(points_3d[:5]) 