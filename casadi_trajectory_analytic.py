#!/usr/bin/env python
"""
Ejemplo de cómo crear interpolación analítica CasADi 
sin necesidad de waypoints discretos para MPCC
"""
import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

def create_analytic_casadi_trajectory(spline_t, spline_x, spline_y, spline_z, 
                                      spline_yaw, value, xd_p, yd_p, zd_p):
    """
    Crea funciones CasADi analíticas para la trayectoria usando los splines ya existentes.
    
    VENTAJA: No necesita waypoints discretos, usa directamente los splines de scipy
    DESVENTAJA: Requiere convertir coeficientes del spline a CasADi
    
    Args:
        spline_t: CubicSpline que mapea arc_length -> tiempo
        spline_x, y, z: CubicSplines de posición
        spline_yaw: CubicSpline del ángulo yaw unwrapped
        value: parámetro global de la trayectoria
        xd_p, yd_p, zd_p: funciones derivadas de la trayectoria
    
    Returns:
        gamma_pos, gamma_vel, gamma_quat: Funciones CasADi(s) -> referencias
    """
    
    # Variable simbólica de longitud de arco
    s_sym = ca.SX.sym('s')
    
    # ========== OPCIÓN A: Usar piecewise polynomials (más eficiente) ==========
    # Los CubicSpline de scipy guardan los coeficientes en spline.c
    # Podemos reconstruir las funciones por tramos en CasADi
    
    def spline_to_casadi(spline_obj, s_sym):
        """
        Convierte un CubicSpline de scipy a expresión CasADi piecewise.
        
        CubicSpline almacena coeficientes como:
        y(x) = c[3]*(x-x[i])**3 + c[2]*(x-x[i])**2 + c[1]*(x-x[i]) + c[0]
        """
        # Obtener puntos de quiebre (knots)
        x_knots = spline_obj.x  # Puntos donde se definen los polinomios
        c_coeffs = spline_obj.c  # Coeficientes [4, n_intervals]
        
        # Crear expresión piecewise en CasADi
        # Inicializar con el primer segmento
        x0 = x_knots[0]
        dx = s_sym - x0
        expr = c_coeffs[3, 0]*dx**3 + c_coeffs[2, 0]*dx**2 + c_coeffs[1, 0]*dx + c_coeffs[0, 0]
        
        # Agregar condiciones para cada segmento
        for i in range(1, len(x_knots) - 1):
            xi = x_knots[i]
            dx_i = s_sym - xi
            poly_i = (c_coeffs[3, i]*dx_i**3 + c_coeffs[2, i]*dx_i**2 + 
                      c_coeffs[1, i]*dx_i + c_coeffs[0, i])
            
            # if s >= xi: usar poly_i, else: mantener expr anterior
            expr = ca.if_else(s_sym >= xi, poly_i, expr)
        
        return expr
    
    # Convertir splines a expresiones CasADi
    t_expr = spline_to_casadi(spline_t, s_sym)
    x_expr = spline_to_casadi(spline_x, s_sym)
    y_expr = spline_to_casadi(spline_y, s_sym)
    z_expr = spline_to_casadi(spline_z, s_sym)
    yaw_expr = spline_to_casadi(spline_yaw, s_sym)
    
    # ========== Funciones de referencia ==========
    # Posición
    pos_expr = ca.vertcat(x_expr, y_expr, z_expr)
    gamma_pos = ca.Function('gamma_pos', [s_sym], [pos_expr])
    
    # Quaternion desde yaw
    half_yaw = yaw_expr / 2
    quat_expr = ca.vertcat(
        ca.cos(half_yaw),  # qw
        0,                  # qx
        0,                  # qy
        ca.sin(half_yaw)   # qz
    )
    gamma_quat = ca.Function('gamma_quat', [s_sym], [quat_expr])
    
    # Velocidad: derivada numérica del spline
    # dx/ds = (dx/dt) / (dt/ds) = xd_p(t) / (ds/dt)
    # Aproximación: usar derivada del spline
    dx_dt = ca.jacobian(x_expr, s_sym)
    dy_dt = ca.jacobian(y_expr, s_sym)
    dz_dt = ca.jacobian(z_expr, s_sym)
    vel_expr = ca.vertcat(dx_dt, dy_dt, dz_dt)
    vel_norm = ca.norm_2(vel_expr) + 1e-8
    vel_normalized = vel_expr / vel_norm
    
    gamma_vel = ca.Function('gamma_vel', [s_sym], [vel_normalized])
    
    return gamma_pos, gamma_vel, gamma_quat


def create_casadi_trajectory_from_functions(xd, yd, zd, xd_p, yd_p, zd_p, 
                                            spline_t, spline_yaw, 
                                            s_min, s_max):
    """
    OPCIÓN B: Crear funciones CasADi evaluando las funciones paramétricas
    directamente sin usar waypoints.
    
    VENTAJA: Más simple conceptualmente
    DESVENTAJA: Requiere lookup de s->t en cada evaluación
    
    Args:
        xd, yd, zd: Funciones de posición f(t)
        xd_p, yd_p, zd_p: Funciones de velocidad df/dt
        spline_t: CubicSpline s -> t
        spline_yaw: CubicSpline s -> yaw
        s_min, s_max: Rango válido de longitud de arco
    
    Returns:
        gamma_pos, gamma_vel, gamma_quat: Funciones CasADi
    """
    
    s_sym = ca.SX.sym('s')
    
    # ========== Problema: xd, yd, zd son funciones Python, no CasADi ==========
    # Solución: Si las funciones son analíticas, reescribirlas en CasADi
    
    # Ejemplo: Para tu trayectoria específica
    # xd(t) = 7 * sin(value * 0.04 * t) + 3
    # Necesitamos expresarla en función de s usando t = spline_t(s)
    
    # Esto requiere que spline_t sea evaluable en CasADi
    # Usar la conversión piecewise de la OPCIÓN A
    
    # Por simplicidad, si la trayectoria es muy compleja, 
    # es mejor usar waypoints (método actual)
    
    pass  # Ver implementación en OPCIÓN A


# ========== RECOMENDACIÓN ==========
def should_you_use_waypoints():
    """
    Cuándo usar waypoints vs expresión analítica:
    
    USAR WAYPOINTS (método actual) SI:
    - La trayectoria viene de datos medidos/muestreados
    - No hay expresión analítica cerrada
    - Necesitas máxima flexibilidad
    - 30-50 waypoints son suficientes para tu resolución
    
    USAR EXPRESIÓN ANALÍTICA SI:
    - La trayectoria es paramétrica (seno, coseno, polinomios)
    - Quieres máxima precisión sin discretización
    - Puedes expresar todo en CasADi simbólicamente
    
    TU CASO: 
    - Trayectoria paramétrica (senos) → PODRÍAS usar analítica
    - Pero usas CubicSpline para s->t → waypoints son razonables
    - 30 waypoints para 80m ≈ 2.7m entre puntos es SUFICIENTE
    
    CONCLUSIÓN: Tu implementación actual es CORRECTA y eficiente.
    Solo considera aumentar a 50 waypoints si ves errores de interpolación.
    """
    pass
