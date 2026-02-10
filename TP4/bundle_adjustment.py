import numpy as np
import PyCeres
import cv2
from scipy.spatial.transform import Rotation as R

# Camera intrinsic matrix K
K = np.array([[718.8, 0, 607.1],
              [0, 718.8, 185.2],
              [0, 0, 1]])

# Initial camera poses from TP2
camera_poses = [
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


# ============================================================================
# Step 1: Projection Model
# ============================================================================

def project_point(X_camera, K):
    """
    Project a 3D point in camera frame to image plane using pinhole model.
    
    Args:
        X_camera: 3D point in camera coordinates [Xc, Yc, Zc]
        K: Camera intrinsic matrix
    
    Returns:
        [u, v]: 2D image coordinates
    """
    fx, fy = K[0, 0], K[1, 1]
    cu, cv = K[0, 2], K[1, 2]
    
    Xc, Yc, Zc = X_camera
    
    # Avoid division by zero
    if Zc < 1e-6:
        return None
    
    u = fx * (Xc / Zc) + cu
    v = fy * (Yc / Zc) + cv
    
    return np.array([u, v])


# ============================================================================
# Step 2: Residual Definition (Cost Function)
# ============================================================================

class ReprojectionError(PyCeres.CostFunction):
    """
    Reprojection error cost function for Bundle Adjustment.
    
    The residual is: r_ij = p_observed_ij - π(R_j * X_w,i + t_j)
    
    Parameters to optimize:
    - camera_pose: [rx, ry, rz, tx, ty, tz] (angle-axis rotation + translation)
    - point_3d: [X, Y, Z] (3D point in world frame)
    """
    
    def __init__(self, observed_x, observed_y, K):
        """
        Args:
            observed_x, observed_y: Observed 2D point in image
            K: Camera intrinsic matrix
        """
        super().__init__()
        self.observed = np.array([observed_x, observed_y])
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cu, self.cv = K[0, 2], K[1, 2]
        
        # Set the sizes: 2 residuals, parameter blocks of size 6 and 3
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 3])
    
    def Evaluate(self, parameters, residuals, jacobians):
        """
        Compute the residual and optionally the Jacobians.
        
        Args:
            parameters: List of parameter blocks [camera_pose, point_3d]
            residuals: Output residual vector
            jacobians: Output Jacobian matrices (if not None)
        
        Returns:
            True if evaluation succeeded
        """
        camera_pose = parameters[0]  # [rx, ry, rz, tx, ty, tz]
        point_3d = parameters[1]      # [X, Y, Z]
        
        # Extract rotation (angle-axis) and translation
        rvec = camera_pose[:3]
        tvec = camera_pose[3:6]
        
        # Convert angle-axis to rotation matrix
        angle = np.linalg.norm(rvec)
        if angle < 1e-10:
            R_mat = np.eye(3)
        else:
            axis = rvec / angle
            R_mat = R.from_rotvec(rvec).as_matrix()
        
        # Transform 3D point to camera frame: X_c = R * X_w + t
        X_camera = R_mat @ point_3d + tvec
        
        Xc, Yc, Zc = X_camera
        
        # Check if point is behind camera
        if Zc < 1e-6:
            residuals[0] = 1e6
            residuals[1] = 1e6
            return True
        
        # Project to image plane
        u = self.fx * (Xc / Zc) + self.cu
        v = self.fy * (Yc / Zc) + self.cv
        
        # Compute residual
        residuals[0] = u - self.observed[0]
        residuals[1] = v - self.observed[1]
        
        # Compute Jacobians if requested (numerical approximation is fine for now)
        # Ceres can also use automatic differentiation
        
        return True


class ReprojectionErrorNumeric(PyCeres.NumericDiffCostFunction):
    """
    Reprojection error using numeric differentiation (easier to implement).
    """
    
    def __init__(self, observed_x, observed_y, K):
        super().__init__()
        self.observed = np.array([observed_x, observed_y])
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cu, self.cv = K[0, 2], K[1, 2]
        
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 3])
    
    def Evaluate(self, parameters, residuals):
        camera_pose = parameters[0]
        point_3d = parameters[1]
        
        rvec = camera_pose[:3]
        tvec = camera_pose[3:6]
        
        # Rotation
        angle = np.linalg.norm(rvec)
        if angle < 1e-10:
            R_mat = np.eye(3)
        else:
            R_mat = R.from_rotvec(rvec).as_matrix()
        
        # Transform to camera frame
        X_camera = R_mat @ point_3d + tvec
        Xc, Yc, Zc = X_camera
        
        if Zc < 1e-6:
            residuals[0] = 1e6
            residuals[1] = 1e6
            return True
        
        # Project
        u = self.fx * (Xc / Zc) + self.cu
        v = self.fy * (Yc / Zc) + self.cv
        
        residuals[0] = u - self.observed[0]
        residuals[1] = v - self.observed[1]
        
        return True


# ============================================================================
# Helper functions to convert poses
# ============================================================================

def pose_matrix_to_params(pose_matrix):
    """
    Convert 4x4 pose matrix to [rx, ry, rz, tx, ty, tz].
    """
    R_mat = pose_matrix[:3, :3]
    t_vec = pose_matrix[:3, 3]
    
    # Convert rotation matrix to angle-axis
    r = R.from_matrix(R_mat)
    rvec = r.as_rotvec()
    
    return np.concatenate([rvec, t_vec])


def params_to_pose_matrix(params):
    """
    Convert [rx, ry, rz, tx, ty, tz] to 4x4 pose matrix.
    """
    rvec = params[:3]
    tvec = params[3:6]
    
    R_mat = R.from_rotvec(rvec).as_matrix()
    
    pose = np.eye(4)
    pose[:3, :3] = R_mat
    pose[:3, 3] = tvec
    
    return pose


# ============================================================================
# Step 3 & 4: Problem Construction and Optimization
# ============================================================================

def run_bundle_adjustment(observations, initial_poses, initial_points, K, 
                         use_robust_loss=False, max_iterations=50):
    """
    Run Bundle Adjustment.
    
    Args:
        observations: List of (camera_idx, point_idx, u, v) tuples
        initial_poses: List of 4x4 pose matrices
        initial_points: Dictionary {point_idx: [X, Y, Z]}
        K: Camera intrinsic matrix
        use_robust_loss: Whether to use Huber loss
        max_iterations: Maximum number of iterations
    
    Returns:
        Optimized poses and points
    """
    # Convert poses to optimization parameters
    camera_params = {}
    for i, pose in enumerate(initial_poses):
        camera_params[i] = pose_matrix_to_params(pose).copy()
    
    # Copy point parameters
    point_params = {}
    for idx, point in initial_points.items():
        point_params[idx] = np.array(point, dtype=np.float64).copy()
    
    # Create Ceres problem
    problem = PyCeres.Problem()
    
    # Loss function (Huber or None)
    loss = PyCeres.HuberLoss(1.0) if use_robust_loss else None
    
    # Add residual blocks for each observation
    for cam_idx, point_idx, u, v in observations:
        cost_function = ReprojectionErrorNumeric(u, v, K)
        
        problem.AddResidualBlock(
            cost_function,
            loss,
            camera_params[cam_idx],
            point_params[point_idx]
        )
    
    # Gauge fixing: Fix the first camera pose
    problem.SetParameterBlockConstant(camera_params[0])
    
    # Configure solver
    options = PyCeres.SolverOptions()
    options.linear_solver_type = PyCeres.LinearSolverType.SPARSE_SCHUR
    options.max_num_iterations = max_iterations
    options.minimizer_progress_to_stdout = True
    
    # Solve
    summary = PyCeres.Summary()
    PyCeres.Solve(options, problem, summary)
    
    print(summary.BriefReport())
    
    # Convert back to pose matrices
    optimized_poses = {}
    for i, params in camera_params.items():
        optimized_poses[i] = params_to_pose_matrix(params)
    
    return optimized_poses, point_params, summary

