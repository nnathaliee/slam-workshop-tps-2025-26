import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from bundle_adjustment import (
    run_bundle_adjustment, 
    pose_matrix_to_params,
    params_to_pose_matrix,
    project_point,
    K
)

class KITTIBundleAdjustment:
    """
    Bundle Adjustment pipeline for KITTI sequence data.
    """
    
    def __init__(self, data_path, K_matrix):
        """
        Args:
            data_path: Path to KITTI sequence directory
            K_matrix: Camera intrinsic matrix
        """
        self.data_path = Path(data_path)
        self.K = K_matrix
        self.images = []
        self.poses = []
        self.observations = []
        self.points_3d = {}
        
    def load_images(self):
        """Load all images from the sequence."""
        image_dir = self.data_path / "image_0"
        if not image_dir.exists():
            image_dir = self.data_path
            
        image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.images.append(img)
        
        print(f"Loaded {len(self.images)} images")
        return self.images
    
    def detect_and_match_features(self, max_features=1000):
        """
        Detect and match features between consecutive frames.
        
        Args:
            max_features: Maximum number of features to detect
        
        Returns:
            Dictionary of matches between frame pairs
        """
        if len(self.images) < 2:
            print("Not enough images for matching")
            return {}
        
        # Use ORB detector (SIFT would be better but ORB is free)
        detector = cv2.ORB_create(nfeatures=max_features)
        
        # Detect keypoints and compute descriptors for all images
        keypoints_list = []
        descriptors_list = []
        
        print("Detecting features...")
        for i, img in enumerate(self.images):
            kp, desc = detector.detectAndCompute(img, None)
            keypoints_list.append(kp)
            descriptors_list.append(desc)
            print(f"Image {i}: {len(kp)} keypoints")
        
        # Match features between consecutive frames
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches_dict = {}
        
        print("\nMatching features between frames...")
        for i in range(len(self.images) - 1):
            if descriptors_list[i] is None or descriptors_list[i+1] is None:
                continue
                
            # Match descriptors
            matches = matcher.knnMatch(descriptors_list[i], descriptors_list[i+1], k=2)
            
            # Apply ratio test (Lowe's ratio test)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            matches_dict[(i, i+1)] = {
                'matches': good_matches,
                'kp1': keypoints_list[i],
                'kp2': keypoints_list[i+1]
            }
            
            print(f"Frames {i}-{i+1}: {len(good_matches)} good matches")
        
        return matches_dict, keypoints_list
    
    def triangulate_points(self, matches_dict, poses):
        """
        Triangulate 3D points from 2D matches and camera poses.
        
        Args:
            matches_dict: Dictionary of matches between frame pairs
            poses: List of camera poses (4x4 matrices)
        
        Returns:
            Dictionary of 3D points and list of observations
        """
        points_3d = {}
        observations = []
        point_counter = 0
        
        # Projection matrices
        P_list = []
        for pose in poses:
            R = pose[:3, :3]
            t = pose[:3, 3:4]
            P = self.K @ np.hstack([R, t])
            P_list.append(P)
        
        # Track which 2D points correspond to which 3D points
        point_2d_to_3d = {}
        
        for (i, j), match_info in matches_dict.items():
            if i >= len(poses) or j >= len(poses):
                continue
            
            matches = match_info['matches']
            kp1 = match_info['kp1']
            kp2 = match_info['kp2']
            
            # Get matched point coordinates
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            # Triangulate
            pts_4d_hom = cv2.triangulatePoints(P_list[i], P_list[j], pts1.T, pts2.T)
            pts_3d = pts_4d_hom[:3] / pts_4d_hom[3]
            
            for k, m in enumerate(matches):
                # Check if this 2D point already has a 3D point
                key1 = (i, m.queryIdx)
                key2 = (j, m.trainIdx)
                
                if key1 in point_2d_to_3d:
                    point_idx = point_2d_to_3d[key1]
                elif key2 in point_2d_to_3d:
                    point_idx = point_2d_to_3d[key2]
                else:
                    # Create new 3D point
                    point_idx = point_counter
                    point_counter += 1
                    points_3d[point_idx] = pts_3d[:, k]
                
                # Record this 2D-3D correspondence
                point_2d_to_3d[key1] = point_idx
                point_2d_to_3d[key2] = point_idx
                
                # Add observations
                observations.append((i, point_idx, pts1[k, 0], pts1[k, 1]))
                observations.append((j, point_idx, pts2[k, 0], pts2[k, 1]))
        
        # Remove duplicate observations
        observations = list(set(observations))
        
        print(f"\nTriangulated {len(points_3d)} 3D points")
        print(f"Total observations: {len(observations)}")
        
        return points_3d, observations
    
    def run_optimization(self, initial_poses, use_robust_loss=True):
        """
        Run the full Bundle Adjustment pipeline.
        
        Args:
            initial_poses: List of initial camera poses
            use_robust_loss: Whether to use Huber loss
        """
        # Store poses
        self.poses = initial_poses
        
        # Detect and match features
        matches_dict, keypoints = self.detect_and_match_features()
        
        # Triangulate 3D points
        self.points_3d, self.observations = self.triangulate_points(
            matches_dict, self.poses
        )
        
        if len(self.observations) < 10:
            print("Not enough observations for Bundle Adjustment!")
            return None, None
        
        # Run Bundle Adjustment
        print("\n" + "="*70)
        print("Running Bundle Adjustment")
        print("="*70)
        
        optimized_poses, optimized_points, summary = run_bundle_adjustment(
            self.observations,
            self.poses,
            self.points_3d,
            self.K,
            use_robust_loss=use_robust_loss,
            max_iterations=50
        )
        
        return optimized_poses, optimized_points
    
    def visualize_results(self, optimized_poses, optimized_points):
        """
        Visualize the optimization results.
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Camera trajectory (top view)
        ax1 = fig.add_subplot(131)
        
        # Initial trajectory
        initial_positions = np.array([pose[:3, 3] for pose in self.poses])
        ax1.plot(initial_positions[:, 0], initial_positions[:, 2], 
                'b.-', label='Initial', linewidth=2)
        
        # Optimized trajectory
        if optimized_poses is not None:
            opt_positions = np.array([
                optimized_poses[i][:3, 3] for i in sorted(optimized_poses.keys())
            ])
            ax1.plot(opt_positions[:, 0], opt_positions[:, 2], 
                    'r.-', label='Optimized', linewidth=2)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Z (m)')
        ax1.set_title('Camera Trajectory (Top View)')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot 2: 3D point cloud
        ax2 = fig.add_subplot(132, projection='3d')
        
        if optimized_points is not None:
            points = np.array([optimized_points[i] for i in sorted(optimized_points.keys())])
            
            # Filter out points too far from cameras (outliers)
            distances = np.linalg.norm(points - initial_positions[0], axis=1)
            valid_points = points[distances < 50]
            
            ax2.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 
                       c=valid_points[:, 2], cmap='viridis', s=1)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title('3D Point Cloud')
        
        # Plot 3: Reprojection errors
        ax3 = fig.add_subplot(133)
        
        if optimized_poses is not None and optimized_points is not None:
            errors = []
            for cam_idx, point_idx, u_obs, v_obs in self.observations:
                if cam_idx in optimized_poses and point_idx in optimized_points:
                    pose = optimized_poses[cam_idx]
                    point = optimized_points[point_idx]
                    
                    # Transform to camera frame
                    R = pose[:3, :3]
                    t = pose[:3, 3]
                    X_cam = R @ point + t
                    
                    # Project
                    if X_cam[2] > 0:
                        projected = project_point(X_cam, self.K)
                        if projected is not None:
                            error = np.sqrt((projected[0] - u_obs)**2 + 
                                          (projected[1] - v_obs)**2)
                            errors.append(error)
            
            ax3.hist(errors, bins=50, edgecolor='black')
            ax3.set_xlabel('Reprojection Error (pixels)')
            ax3.set_ylabel('Count')
            ax3.set_title('Reprojection Error Distribution')
            ax3.axvline(np.median(errors), color='r', linestyle='--', 
                       label=f'Median: {np.median(errors):.2f} px')
            ax3.legend()
        
        plt.tight_layout()
        plt.savefig('/home/claude/bundle_adjustment_results.png', dpi=150)
        print("\nVisualization saved to bundle_adjustment_results.png")
        plt.close()


# Main execution

if __name__ == "__main__":
    # Camera poses estimates in TP2
    camera_poses_provided = [
        np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]]),
        np.array([[ 9.99999086e-01,  5.54218703e-04, -1.23301244e-03,  2.87373955e-03],
                  [-5.53472899e-04,  9.99999664e-01,  6.05121998e-04,  1.35098025e-02],
                  [ 1.23334740e-03, -6.04439006e-04,  9.99999057e-01, -9.99904609e-01],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        np.array([[ 9.99997020e-01, -1.21127533e-03, -2.11946137e-03,  5.40120870e-03],
                  [ 1.21869157e-03,  9.99993128e-01,  3.50132893e-03,  2.53821160e-02],
                  [ 2.11520573e-03, -3.50390147e-03,  9.99991624e-01, -1.99983094e+00],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        np.array([[ 0.99997888, -0.00491749, -0.00425019,  0.01860248],
                  [ 0.00493906,  0.9999749 ,  0.00507939,  0.02999638],
                  [ 0.00422511, -0.00510028,  0.99997807, -2.99973315],
                  [ 0.        ,  0.        ,  0.        ,  1.        ]]),
        np.array([[ 0.99992938, -0.00949936, -0.00714063,  0.04044515],
                  [ 0.00954163,  0.99993702,  0.00590894,  0.0318811 ],
                  [ 0.00708405, -0.00597665,  0.99995705, -3.99949279],
                  [ 0.        ,  0.        ,  0.        ,  1.        ]])
    ]
    
    print("KITTI Bundle Adjustment Pipeline")
    print("="*70)
    
    # Check if data directory exists
    data_path = Path("data/Sequence")
    
    if data_path.exists():
        print(f"\nUsing data from: {data_path}")
        
        ba = KITTIBundleAdjustment(data_path, K)
        ba.load_images()
        
        if len(ba.images) >= len(camera_poses_provided):
            optimized_poses, optimized_points = ba.run_optimization(
                camera_poses_provided[:len(ba.images)],
                use_robust_loss=True
            )
            
            if optimized_poses is not None:
                ba.visualize_results(optimized_poses, optimized_points)
        else:
            print(f"Warning: Only {len(ba.images)} images found, " + 
                  f"but {len(camera_poses_provided)} poses provided")
    else:
        print(f"\nData directory not found: {data_path}")
        print("Running with synthetic example instead...")
        
        # Run the synthetic example from bundle_adjustment.py
        from bundle_adjustment import create_synthetic_example
        
        observations, poses, points = create_synthetic_example()
        optimized_poses, optimized_points, summary = run_bundle_adjustment(
            observations, poses, points, K,
            use_robust_loss=True, max_iterations=50
        )
