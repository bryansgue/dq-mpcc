"""
EJEMPLO PRÁCTICO: Cómo modificar nmpc_acados.py paso a paso
"""

# ============================================================================
# PASO 1: Agregar imports al principio de nmpc_acados.py
# ============================================================================

# Reemplazar la sección de imports con:

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from ode_acados import quadrotorModel
from casadi import Function, MX, vertcat, sin, cos, fabs, DM
import casadi as ca
import numpy as np
from casadi import sqrt, dot, fabs, sign

# ✓ NEW IMPORTS:
from mpcc_logarithmic import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_rotation_vector,
    logarithmic_translational_error,
    logarithmic_lag_and_contouring_errors,
    norm_3d,
    left_jacobian_so3_inv,
    skew_symmetric_3d,
)


# ============================================================================
# PASO 2: En create_ocp_solver(), busca la sección #mpcc (~línea 113)
# ============================================================================

# ─── ANTES (Euclidiano) ─────────────────────────────────────────────────
    #mpcc
    s = dual_quaternion_to_position(dual)
    sd = dual_quaternion_to_position(dual_d)
    e_t = (sd - s)
    
    #Vector tangente
    sd_p = ocp.p[11:14]
    tangent_normalized = sd_p
    el = dot(tangent_normalized, e_t) * tangent_normalized
    
    # ERROR DE CONTORNO
    I = MX.eye(3) 
    P_ec = I - tangent_normalized.T @ tangent_normalized
    ec = P_ec @ e_t

# ─── DESPUÉS (Logarítmico) ─────────────────────────────────────────────

    # ====================================================================
    # LOGARITHMIC MPCC SECTION
    # ====================================================================
    
    # Extract current and desired positions
    p = dual_quaternion_to_position(dual)      # Posición actual
    p_d = dual_quaternion_to_position(dual_d)  # Posición deseada
    
    # Extract quaternions
    q = dual[0:4]      # Cuaternión actual
    q_d = dual_d[0:4]  # Cuaternión deseado
    
    # Compute rotation matrices
    R = quaternion_to_rotation_matrix(q)       # Matriz actual
    R_d = quaternion_to_rotation_matrix(q_d)   # Matriz deseada
    
    # Step 1: Compute relative rotation R_rel = R_d^T @ R
    R_rel = R_d.T @ R
    
    # Step 2: Extract rotation vector φ from relative rotation
    phi = rotation_matrix_to_rotation_vector(R_rel, eps=1e-6)
    
    # Step 3: Compute Jacobian left inverse J_l(φ)^{-1}
    J_l_inv = left_jacobian_so3_inv(phi, eps=1e-6)
    
    # Step 4: Compute logarithmic translational error
    #         ρ = J_l(φ)^{-1} R_d^T (p - p_d)
    rho = logarithmic_translational_error(phi, R_d, p, p_d, eps=1e-6)
    
    # Step 5: Transform path tangent to desired frame
    tangent_world = ocp.p[11:14]  # Path tangent in world frame
    tangent_desired_unnormalized = R_d.T @ tangent_world
    tangent_norm = norm_3d(tangent_desired_unnormalized)
    t_r = tangent_desired_unnormalized / (tangent_norm + 1e-6)  # t^r (normalized)
    
    # Step 6: Decompose ρ into lag and contouring components
    #         ρ_l = t^r (t^{r,T} ρ)
    #         ρ_c = (I - t^r t^{r,T}) ρ
    rho_l, rho_c = logarithmic_lag_and_contouring_errors(phi, rho, t_r)


# ============================================================================
# PASO 3: Modificar la función de costo (~línea 145)
# ============================================================================

# ─── ANTES ──────────────────────────────────────────────────────────────
    Q_el = 1 * np.eye(3)  
    Q_ec = 5 * np.eye(3) 
    Q_vels = 0.5
    U_mat = 1 * np.diag([ 0.2, 10, 10, 10])

    control_cost = 0.5*model.u[0:4].T @ U_mat @ model.u[0:4]
    error_contorno = 1*ec.T @ Q_ec @ ec
    error_lag = 1*el.T @ Q_el @ el
    vel_progres = dot(tangent_normalized, v)
    vel_progres_cost = Q_vels*vel_progres  

    ocp.model.cost_expr_ext_cost = 10*(ln_error.T@Q_l@ln_error) +  control_cost + error_contorno + error_lag  - vel_progres_cost
    ocp.model.cost_expr_ext_cost_e =  10*(ln_error.T@Q_l@ln_error) + error_contorno + error_lag  - vel_progres_cost

# ─── DESPUÉS ────────────────────────────────────────────────────────────

    # Define weights for logarithmic MPCC
    Q_rho_l = 1.0 * np.eye(3)     # Weight for logarithmic lag error
    Q_rho_c = 5.0 * np.eye(3)     # Weight for logarithmic contouring error (stronger)
    U_mat = 1.0 * np.diag([0.2, 10, 10, 10])
    
    # Cost terms
    control_cost = 0.5 * model.u[0:4].T @ U_mat @ model.u[0:4]
    
    # Logarithmic MPCC costs (pose-coupled)
    cost_rho_lag = rho_l.T @ Q_rho_l @ rho_l
    cost_rho_contouring = rho_c.T @ Q_rho_c @ rho_c
    
    # Progress reward: velocity projected onto path tangent
    v_b = model.x[11:14]  # Linear velocity in body frame
    # Note: For consistency, could also use rotated velocity in inertial frame
    vel_progress = dot(t_r, v_b)
    cost_progress = -0.5 * vel_progress  # Negative to reward forward motion
    
    # Stage cost: Rotational (Lie) + Control + Translational (Lie-MPCC)
    ocp.model.cost_expr_ext_cost = (
        10 * (ln_error.T @ Q_l @ ln_error) +     # Rotational error
        control_cost +                             # Control effort
        cost_rho_lag +                             # Logarithmic lag
        cost_rho_contouring +                      # Logarithmic contouring
        cost_progress                              # Progress penalty
    )
    
    # Terminal cost: No progress term
    ocp.model.cost_expr_ext_cost_e = (
        10 * (ln_error.T @ Q_l @ ln_error) +
        cost_rho_lag +
        cost_rho_contouring
    )


# ============================================================================
# PASO 4: (OPCIONAL) Agregar diagnóstico
# ============================================================================

# Puedes agregar logging en el loop de simulación para diagnosticar:

def log_mpcc_diagnostics(t, phi, rho, rho_l, rho_c, v_b, t_r):
    """
    Log diagnostics of logarithmic MPCC for debugging.
    """
    phi_norm = np.linalg.norm(phi)
    rho_norm = np.linalg.norm(rho)
    rho_l_norm = np.linalg.norm(rho_l)
    rho_c_norm = np.linalg.norm(rho_c)
    vel_progress = np.dot(t_r, v_b)
    
    print(f"[t={t:.2f}s] φ={phi_norm:.4f} | ρ={rho_norm:.4f} | "
          f"ρ_l={rho_l_norm:.4f} | ρ_c={rho_c_norm:.4f} | v·t={vel_progress:.4f}")
    
    return {
        'phi_norm': phi_norm,
        'rho_norm': rho_norm,
        'rho_l_norm': rho_l_norm,
        'rho_c_norm': rho_c_norm,
        'vel_progress': vel_progress,
    }


# ============================================================================
# RESUMEN DE CAMBIOS
# ============================================================================

"""
CAMBIOS TOTALES EN nmpc_acados.py:

1. ✓ Imports (3 líneas nuevas)
   - Agregar funciones logarítmicas de mpcc_logarithmic.py

2. ✓ Bloque MPCC (reemplazar ~15 líneas, agregar ~20)
   - Cambiar de Euclidiano a Logarítmico
   - Calcular φ, J_l(φ)^{-1}, ρ
   - Transformar tangente a frame deseado
   - Descomponer ρ en ρ_l y ρ_c

3. ✓ Función de costo (reemplazar ~8 líneas)
   - El, ec → rho_l, rho_c
   - Mismo número de términos
   - Misma estructura, diferente significado geométrico

4. ✓ (Opcional) Diagnóstico
   - Agregar logging para verificar comportamiento

LÍNEAS AFECTADAS APROX.:
- create_ocp_solver() línea ~40-50: imports
- create_ocp_solver() línea ~113-135: sección MPCC
- create_ocp_solver() línea ~145-160: función costo

TIEMPO DE IMPLEMENTACIÓN: ~10 minutos
RIESGO: BAJO (cambio bien localizado, fácil de revertir)
"""


# ============================================================================
# TABLA DE VERIFICACIÓN PRE-INTEGRACIÓN
# ============================================================================

CHECKLIST = """
Antes de integrar, verificar:

□ mpcc_logarithmic.py compila sin errores
  → python3 -c "from mpcc_logarithmic import *; print('OK')"

□ ode_acados.py tiene función quaternion_to_rotation_matrix()
  → Buscar: def quaternion_to_matrix_casadi() o similar

□ Tu actual MPCC en nmpc_acados.py funciona (baseline)
  → Ejecutar OCP con configuración Euclidiana como referencia

□ Parámetros en ocp.p están en orden correcto:
  [0:8] → dual_d
  [8:11] → w_b_d
  [11:14] → v_i_d (o tangent)
  [14:18] → nominal_input

□ Función ln() en ode_acados.py está disponible
  → Buscar: ln = ... en quadrotorModel()

□ Manuales de test preparados:
  - Trayectoria recta (t_r constante)
  - Curva simple (círculo o parábola)
  - Comparar resultados Euclidiano vs Logarítmico
"""

print(CHECKLIST)
