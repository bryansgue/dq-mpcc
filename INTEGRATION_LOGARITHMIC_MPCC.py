"""
Integration guide: How to use logarithmic MPCC in nmpc_acados.py

This file shows the exact modifications needed to replace your Euclidean MPCC
with Logarithmic MPCC in your nmpc_acados.py file.
"""

# ============================================================================
# STEP 1: Import the new module at the top of nmpc_acados.py
# ============================================================================

from mpcc_logarithmic import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_rotation_vector,
    logarithmic_translational_error,
    logarithmic_lag_and_contouring_errors,
    build_logarithmic_mpcc_terms,
    left_jacobian_so3,
    left_jacobian_so3_inv,
    norm_3d,
)


# ============================================================================
# STEP 2: Helper function to compute rotation matrix from quaternion in CasADi
# ============================================================================

def quaternion_to_matrix_casadi(q):
    """
    Convert CasADi quaternion (w, x, y, z) to rotation matrix.
    This wraps quaternion_to_rotation_matrix for use in OCP.
    """
    return quaternion_to_rotation_matrix(q)


# ============================================================================
# STEP 3: Modify create_ocp_solver() to include logarithmic MPCC
# ============================================================================

def create_ocp_solver_with_logarithmic_mpcc(
    x0, N_horizon, t_horizon, 
    F_max, F_min, tau_1_max, tau_1_min, tau_2_max, tau_2_min, tau_3_max, tau_3_min,
    L, ts
) -> AcadosOcp:
    """
    Create OCP with logarithmic MPCC instead of Euclidean MPCC.
    
    New Parameters:
    ---------------
    None (uses existing configuration, but interprets it geometrically)
    
    Returns:
    --------
    AcadosOcp with logarithmic lag and contouring errors
    """
    
    ocp = AcadosOcp()
    model, get_trans, get_quat, constraint, error_lie_2, dual_error, ln, Ad, conjugate, rotation = quadrotorModel(L)
    ocp.model = model
    
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ocp.p = model.p
    ocp.dims.N = N_horizon
    
    # Control effort matrix (unchanged)
    R = MX.zeros(4, 4)
    R[0, 0] = 20/F_max
    R[1, 1] = 60/tau_1_max
    R[2, 2] = 60/tau_2_max
    R[3, 3] = 60/tau_3_max
    
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    
    # ========================================================================
    # Extract current and desired states
    # ========================================================================
    dual_d = ocp.p[0:8]
    dual = model.x[0:8]
    q = dual[0:4]
    q_d = dual_d[0:4]
    
    # Get rotational error using existing ln() function
    ln_error = ln(dual_error(dual_d, dual))  # This gives you Ln(q_e)
    
    Q_l = MX.zeros(6, 6)
    Q_l[0, 0] = 4
    Q_l[1, 1] = 4
    Q_l[2, 2] = 4
    Q_l[3, 3] = 0
    Q_l[4, 4] = 0
    Q_l[5, 5] = 0
    
    # ========================================================================
    # NEW: Logarithmic MPCC Terms
    # ========================================================================
    
    # Positions from dual quaternions
    p = dual_quaternion_to_position(dual)        # Current position
    p_d = dual_quaternion_to_position(dual_d)    # Desired position
    
    # Rotation matrices
    R_d = quaternion_to_rotation_matrix(q_d)     # Desired rotation
    R = quaternion_to_rotation_matrix(q)         # Current rotation
    
    # Relative rotation: R_rel = R_d^⊤ R
    R_rel = R_d.T @ R
    
    # Extract rotation vector (φ) from relative rotation
    phi = rotation_matrix_to_rotation_vector(R_rel, eps=1e-6)
    
    # Logarithmic translational error: ρ = J_l(φ)^{-1} R_d^⊤ (p - p_d)
    rho = logarithmic_translational_error(phi, R_d, p, p_d, eps=1e-6)
    
    # Path tangent vector (from parameters)
    tangent_world = ocp.p[11:14]  # This is sd_p in your code
    
    # Transform tangent to desired frame: t^r = R_d^⊤ t^w
    t_r_unnormalized = R_d.T @ tangent_world
    t_r_norm = norm_3d(t_r_unnormalized)
    t_r = t_r_unnormalized / (t_r_norm + 1e-6)
    
    # Decompose ρ into lag and contouring components
    rho_l, rho_c = logarithmic_lag_and_contouring_errors(phi, rho, t_r)
    
    # ========================================================================
    # Cost function with logarithmic MPCC
    # ========================================================================
    
    # Weights
    Q_rho_l = 1.0 * np.eye(3)  # Weight for logarithmic lag
    Q_rho_c = 5.0 * np.eye(3)  # Weight for logarithmic contouring (higher penalty)
    U_mat = 1.0 * np.diag([0.2, 10, 10, 10])
    
    # Cost terms
    control_cost = 0.5 * model.u[0:4].T @ U_mat @ model.u[0:4]
    
    # Logarithmic MPCC costs
    cost_rho_lag = rho_l.T @ Q_rho_l @ rho_l
    cost_rho_contouring = rho_c.T @ Q_rho_c @ rho_c
    
    # Progress along path (penalize when not moving forward)
    v_b = model.x[11:14]
    vel_progress = ca.dot(t_r, v_b)  # Using t_r (in desired frame)
    cost_progress = -0.5 * vel_progress  # Negative to reward forward progress
    
    # Total cost (stage)
    ocp.model.cost_expr_ext_cost = (
        10 * (ln_error.T @ Q_l @ ln_error) +      # Rotational error from Lie algebra
        control_cost +                              # Control effort
        cost_rho_lag +                              # Logarithmic lag error
        cost_rho_contouring +                       # Logarithmic contouring error
        cost_progress                               # Progress penalty
    )
    
    # Terminal cost (no progress term)
    ocp.model.cost_expr_ext_cost_e = (
        10 * (ln_error.T @ Q_l @ ln_error) +
        cost_rho_lag +
        cost_rho_contouring
    )
    
    # ========================================================================
    # Constraints and solver options (unchanged)
    # ========================================================================
    
    ocp.parameter_values = np.zeros(nx + nu)
    
    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = np.array([F_min, tau_1_min, tau_2_min, tau_3_min])
    ocp.constraints.ubu = np.array([F_max, tau_1_max, tau_2_max, tau_3_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = x0
    
    ocp.model.con_h_expr = constraint.expr
    nsbx = 0
    nh = constraint.expr.shape[0]
    nsh = nh
    ns = nsh + nsbx
    
    ocp.cost.zl = 100 * np.ones((ns,))
    ocp.cost.Zl = 100 * np.ones((ns,))
    ocp.cost.Zu = 100 * np.ones((ns,))
    ocp.cost.zu = 100 * np.ones((ns,))
    
    ocp.constraints.lh = np.array([constraint.min])
    ocp.constraints.uh = np.array([constraint.max])
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array(range(nsh))
    
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = N_horizon // 4
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.Tsim = ts
    ocp.solver_options.tf = t_horizon
    
    return ocp


# ============================================================================
# COMPARISON TABLE: Euclidean vs. Logarithmic MPCC
# ============================================================================
"""
┌─────────────────────────────────┬──────────────────────────┬──────────────────────────┐
│ Component                       │ Euclidean MPCC (yours)   │ Logarithmic MPCC (new)   │
├─────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Position Error                  │ e_t = p_d - p            │ ρ = J_l(φ)^{-1} R_d^⊤...│
│ Lag Component                   │ e_l = (t·e_t) t          │ ρ_l = t(t^⊤ρ)           │
│ Contouring Component            │ e_c = (I-tt^⊤) e_t       │ ρ_c = (I-tt^⊤) ρ        │
│ Geometric Interpretation        │ Simple distance          │ Local pose correction    │
│ Rotational Coupling             │ None (separate)          │ Through φ & J_l(φ)      │
│ Handling Large Rotations        │ Poor                     │ Consistent               │
│ Limiting Case (φ→0)            │ N/A                      │ Recovers Euclidean form  │
└─────────────────────────────────┴──────────────────────────┴──────────────────────────┘
"""


# ============================================================================
# KEY MATHEMATICAL CHANGES
# ============================================================================
"""
1. ROTATIONAL ERROR (unchanged):
   φ = rotation_vector(R_d^⊤ R)
   
2. TRANSLATIONAL ERROR (NEW):
   OLD: e_t = p_d - p
   NEW: ρ = J_l(φ)^{-1} R_d^⊤ (p - p_d)
        where J_l(φ) = I + (1-cos‖φ‖)/‖φ‖² φ^∧ + (‖φ‖-sin‖φ‖)/‖φ‖³ (φ^∧)²
   
3. LAG ERROR:
   OLD: e_l = (t·e_t) t
   NEW: ρ_l = t(t^⊤ ρ)
        [Same structure, but ρ is pose-coupled]
   
4. CONTOURING ERROR:
   OLD: e_c = (I - tt^⊤) e_t
   NEW: ρ_c = (I - tt^⊤) ρ
        [Same structure, but ρ is pose-coupled]
   
5. COST FUNCTION:
   OLD: J = 10‖Ln(q_e)‖²_Q + ‖e_l‖²_Q_l + ‖e_c‖²_Q_c - κ v·t
   NEW: J = 10‖Ln(q_e)‖²_Q + ‖ρ_l‖²_Q_l + ‖ρ_c‖²_Q_c - κ v·t
        [Only ρ replaces e in translational terms]
"""
