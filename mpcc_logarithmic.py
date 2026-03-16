"""
Logarithmic MPCC (Model Predictive Contouring Control) for Dual Quaternions

This module implements the logarithmic lag and contouring error components
from the Lie algebra se(3) representation, providing a geometrically consistent
extension of classical MPCC.

References:
    - Logarithmic map of dual quaternion error
    - Left Jacobian of SO(3)
    - Lag and contouring error decomposition in Lie algebra
"""

import casadi as ca
import numpy as np
from casadi import MX, sqrt, dot, cross, vertcat, horzcat


def skew_symmetric_3d(v):
    """
    Create skew-symmetric matrix from 3D vector (wedge operator).
    
    Parameters
    ----------
    v : ca.MX or np.ndarray
        3D vector [v1, v2, v3]
    
    Returns
    -------
    ca.MX (3x3)
        Skew-symmetric matrix v^∧ such that v^∧ @ w = v × w
    """
    v1, v2, v3 = v[0], v[1], v[2]
    return vertcat(
        horzcat(0, -v3, v2),
        horzcat(v3, 0, -v1),
        horzcat(-v2, v1, 0)
    )


def norm_3d(v):
    """Compute Euclidean norm of 3D vector."""
    return sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def left_jacobian_so3(phi, eps=1e-6):
    """
    Compute the left Jacobian of SO(3) for a 3D rotation vector.
    
    J_l(φ) = I₃ + (1-cos‖φ‖)/‖φ‖² φ^∧ + (‖φ‖-sin‖φ‖)/‖φ‖³ (φ^∧)²
    
    Parameters
    ----------
    phi : ca.MX (3,)
        Rotation vector (axis-angle representation)
    eps : float
        Regularization threshold for small angles
    
    Returns
    -------
    ca.MX (3x3)
        Left Jacobian matrix J_l(φ)
    """
    phi_norm = norm_3d(phi)
    phi_wedge = skew_symmetric_3d(phi)
    phi_wedge_sq = phi_wedge @ phi_wedge
    
    I3 = MX.eye(3)
    
    # Coefficients
    # a = (1 - cos(‖φ‖)) / ‖φ‖²
    a = (1 - ca.cos(phi_norm)) / (phi_norm**2 + eps)
    
    # b = (‖φ‖ - sin(‖φ‖)) / ‖φ‖³
    b = (phi_norm - ca.sin(phi_norm)) / (phi_norm**3 + eps)
    
    J_l = I3 + a * phi_wedge + b * phi_wedge_sq
    
    return J_l


def left_jacobian_so3_inv(phi, eps=1e-6):
    """
    Compute the inverse of the left Jacobian of SO(3).
    
    For small angles: J_l^{-1}(φ) ≈ I₃ - (1/2) φ^∧
    For general case: use numerical inversion
    
    Parameters
    ----------
    phi : ca.MX (3,)
        Rotation vector
    eps : float
        Regularization threshold
    
    Returns
    -------
    ca.MX (3x3)
        Inverse of left Jacobian J_l(φ)^{-1}
    """
    J_l = left_jacobian_so3(phi, eps)
    # Numerical inversion using CasADi
    J_l_inv = ca.inv(J_l)
    return J_l_inv


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion (w, x, y, z) to rotation matrix R ∈ SO(3).
    
    Parameters
    ----------
    q : ca.MX (4,)
        Quaternion [w, x, y, z]
    
    Returns
    -------
    ca.MX (3x3)
        Rotation matrix
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    
    R = vertcat(
        horzcat(1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)),
        horzcat(    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)),
        horzcat(    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2))
    )
    return R


def rotation_matrix_to_rotation_vector(R, eps=1e-6):
    """
    Convert rotation matrix to rotation vector (axis-angle representation).
    
    φ = θ * k, where θ ∈ [0, π] and k is the rotation axis
    
    Parameters
    ----------
    R : ca.MX (3x3)
        Rotation matrix
    eps : float
        Regularization threshold
    
    Returns
    -------
    ca.MX (3,)
        Rotation vector φ
    """
    # Trace of R
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    # Angle θ from trace
    cos_theta = (trace - 1) / 2
    # Clamp to [-1, 1] to avoid numerical issues
    cos_theta = ca.if_else(cos_theta > 1, 1, ca.if_else(cos_theta < -1, -1, cos_theta))
    theta = ca.acos(cos_theta)
    
    # Axis k from skew-symmetric part
    # 2 sin(θ) k = [R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]
    sin_theta = ca.sin(theta)
    
    # Avoid division by zero
    denom = 2 * sin_theta + eps
    
    k = vertcat(
        (R[2, 1] - R[1, 2]) / denom,
        (R[0, 2] - R[2, 0]) / denom,
        (R[1, 0] - R[0, 1]) / denom
    )
    
    phi = theta * k
    return phi


def logarithmic_translational_error(phi, R_d, p, p_d, eps=1e-6):
    """
    Compute the logarithmic translational error ρ from Lie algebra.
    
    ρ = J_l(φ)^{-1} R_d^⊤ (p - p_d)
    
    Parameters
    ----------
    phi : ca.MX (3,)
        Rotational logarithmic error (rotation vector)
    R_d : ca.MX (3x3)
        Desired rotation matrix
    p : ca.MX (3,)
        Current position
    p_d : ca.MX (3,)
        Desired position
    eps : float
        Regularization threshold
    
    Returns
    -------
    ca.MX (3,)
        Logarithmic translational error ρ
    """
    J_l_inv = left_jacobian_so3_inv(phi, eps)
    e_p_r = R_d.T @ (p - p_d)  # Position error in desired frame
    rho = J_l_inv @ e_p_r
    return rho


def logarithmic_lag_and_contouring_errors(phi, rho, t_r):
    """
    Decompose logarithmic translational error into lag and contouring components.
    
    ρ_l = t^r (t^{r⊤} ρ)          (logarithmic lag)
    ρ_c = (I₃ - t^r t^{r⊤}) ρ    (logarithmic contouring)
    
    Parameters
    ----------
    phi : ca.MX (3,)
        Rotational logarithmic error
    rho : ca.MX (3,)
        Logarithmic translational error
    t_r : ca.MX (3,)
        Path tangent vector in desired frame (unit length)
    
    Returns
    -------
    rho_l : ca.MX (3,)
        Logarithmic lag error
    rho_c : ca.MX (3,)
        Logarithmic contouring error
    """
    I3 = MX.eye(3)
    
    # Projection onto tangent
    t_r_t_r_T = t_r @ t_r.T  # 3x3 outer product
    rho_l = t_r_t_r_T @ rho
    
    # Projection onto normal
    P_normal = I3 - t_r_t_r_T
    rho_c = P_normal @ rho
    
    return rho_l, rho_c


def mpcc_logarithmic_cost(phi, rho, rho_l, rho_c, Q_l, Q_c, kappa=None):
    """
    Compute MPCC cost from logarithmic lag and contouring components.
    
    J_mpcc = ‖ρ_l‖²_Q_l + ‖ρ_c‖²_Q_c - κ ‖v_b‖
    
    where κ is a positive weight for progress penalization.
    
    Parameters
    ----------
    phi : ca.MX (3,)
        Rotational logarithmic error
    rho : ca.MX (3,)
        Logarithmic translational error
    rho_l : ca.MX (3,)
        Logarithmic lag error
    rho_c : ca.MX (3,)
        Logarithmic contouring error
    Q_l : np.ndarray or float
        Weight matrix/scalar for lag error
    Q_c : np.ndarray or float
        Weight matrix/scalar for contouring error
    kappa : float, optional
        Weight for progress along path
    
    Returns
    -------
    ca.MX
        Scalar cost value
    """
    if isinstance(Q_l, (int, float)):
        cost_lag = Q_l * dot(rho_l, rho_l)
    else:
        cost_lag = rho_l.T @ Q_l @ rho_l
    
    if isinstance(Q_c, (int, float)):
        cost_contouring = Q_c * dot(rho_c, rho_c)
    else:
        cost_contouring = rho_c.T @ Q_c @ rho_c
    
    return cost_lag + cost_contouring


# ============================================================================
# HELPER FUNCTION: Build complete logarithmic MPCC cost
# ============================================================================

def build_logarithmic_mpcc_terms(dual, dual_d, p_tangent_d, R_d_func, p_from_dq_func, eps=1e-6):
    """
    Build all terms needed for logarithmic MPCC cost.
    
    This is a convenience function to compute all logarithmic components
    from dual quaternion inputs.
    
    Parameters
    ----------
    dual : ca.MX (8,)
        Current dual quaternion [q_w, q_x, q_y, q_z, d_w, d_x, d_y, d_z]
    dual_d : ca.MX (8,)
        Desired dual quaternion
    p_tangent_d : ca.MX (3,)
        Path tangent vector (in world frame)
    R_d_func : callable
        Function to compute desired rotation matrix from desired dual quaternion
        Signature: R_d = R_d_func(dual_d)
    p_from_dq_func : callable
        Function to extract position from dual quaternion
        Signature: p = p_from_dq_func(dq)
    eps : float
        Regularization threshold
    
    Returns
    -------
    dict with keys:
        'phi': Rotational logarithmic error
        'rho': Logarithmic translational error
        'rho_l': Logarithmic lag error
        'rho_c': Logarithmic contouring error
        'R_d': Desired rotation matrix
        't_r': Tangent vector in desired frame
    """
    # Extract positions
    p = p_from_dq_func(dual)
    p_d = p_from_dq_func(dual_d)
    
    # Extract rotations
    q_d = dual_d[0:4]
    q = dual[0:4]
    
    # Compute rotation matrices
    R_d = R_d_func(q_d)
    R = R_d_func(q)
    
    # Compute relative rotation matrix: R_rel = R_d^⊤ R
    R_rel = R_d.T @ R
    
    # Extract rotation vector from relative rotation
    phi = rotation_matrix_to_rotation_vector(R_rel, eps)
    
    # Compute logarithmic translational error
    rho = logarithmic_translational_error(phi, R_d, p, p_d, eps)
    
    # Tangent vector in desired frame
    t_r = R_d.T @ p_tangent_d  # Transform tangent to desired frame
    t_r_norm = norm_3d(t_r)
    t_r_normalized = t_r / (t_r_norm + eps)  # Normalize
    
    # Decompose into lag and contouring
    rho_l, rho_c = logarithmic_lag_and_contouring_errors(phi, rho, t_r_normalized)
    
    return {
        'phi': phi,
        'rho': rho,
        'rho_l': rho_l,
        'rho_c': rho_c,
        'R_d': R_d,
        't_r': t_r_normalized,
    }
