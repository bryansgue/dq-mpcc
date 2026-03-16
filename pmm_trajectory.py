"""
PMM Trajectory Loader for MPCC

Este módulo carga la trayectoria generada por PPM.py (Point Mass Model)
y la convierte al formato requerido por MPC2.py:
- Parametrización por longitud de arco (s)
- Cuaterniones de orientación (yaw alineado con velocidad)
- Funciones de interpolación compatibles con el MPCC
- SLERP para interpolación suave de cuaterniones

Archivos de entrada (generados por PPM.py):
- xref_optimo_3D_PMM.npy: Estados [x, y, z, vx, vy, vz] (6 x N+1)
- uref_optimo_3D_PMM.npy: Controles [ax, ay, az] (3 x N)
- tref_optimo_3D_PMM.npy: Vector de tiempo (N+1)

Salida:
- position_by_arc_length(s): Retorna [x, y, z] para un valor de arco s
- velocity_by_arc_length(s): Retorna [vx, vy, vz] para un valor de arco s
- quaternion_by_arc_length(s): Retorna [w, x, y, z] para un valor de arco s (usando SLERP)
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
import os


def slerp_quaternion(q0, q1, alpha):
    """
    Spherical Linear Interpolation (SLERP) entre dos cuaterniones.
    
    Args:
        q0: Cuaternión inicial [w, x, y, z]
        q1: Cuaternión final [w, x, y, z]
        alpha: Factor de interpolación [0, 1]
    
    Returns:
        Cuaternión interpolado [w, x, y, z]
    """
    # Asegurar camino corto (dot product positivo)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    
    # Si muy cercanos, usar LERP
    if dot > 0.9995:
        result = (1 - alpha) * q0 + alpha * q1
        return result / (np.linalg.norm(result) + 1e-10)
    
    # SLERP
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)
    
    if sin_theta < 1e-10:
        result = (1 - alpha) * q0 + alpha * q1
        return result / (np.linalg.norm(result) + 1e-10)
    
    weight_0 = np.sin((1 - alpha) * theta) / sin_theta
    weight_1 = np.sin(alpha * theta) / sin_theta
    
    result = weight_0 * q0 + weight_1 * q1
    return result / (np.linalg.norm(result) + 1e-10)


def load_pmm_trajectory(data_dir=None, verbose=True, min_velocity=0.1):
    """
    Carga la trayectoria PMM y crea funciones de interpolación por longitud de arco.
    
    Args:
        data_dir: Directorio donde están los archivos .npy (default: directorio actual)
        verbose: Imprimir información de la trayectoria
        min_velocity: Velocidad mínima para calcular orientación (evita divisiones por cero)
    
    Returns:
        dict con:
        - position_by_arc_length: función(s) -> [x, y, z]
        - velocity_by_arc_length: función(s) -> [vx, vy, vz]
        - quaternion_by_arc_length: función(s) -> [w, x, y, z]
        - total_length: longitud total de la trayectoria
        - total_time: tiempo total de la trayectoria
        - arc_lengths: vector de longitudes de arco
    """
    
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cargar archivos
    xref_file = os.path.join(data_dir, 'xref_optimo_3D_PMM.npy')
    tref_file = os.path.join(data_dir, 'tref_optimo_3D_PMM.npy')
    
    if not os.path.exists(xref_file):
        raise FileNotFoundError(f"No se encontró {xref_file}. Ejecuta PPM.py primero.")
    
    X = np.load(xref_file)  # (6, N+1) o (N+1, 6)
    t = np.load(tref_file)  # (N+1,)
    
    # Asegurar formato correcto (6 x N+1)
    if X.shape[0] != 6:
        X = X.T
    
    n_points = X.shape[1]
    
    # Extraer posiciones y velocidades
    positions = X[0:3, :].T  # (N+1, 3)
    velocities = X[3:6, :].T  # (N+1, 3)
    
    # =========================================================================
    # CALCULAR LONGITUD DE ARCO
    # =========================================================================
    arc_lengths = np.zeros(n_points)
    for i in range(1, n_points):
        segment_length = np.linalg.norm(positions[i] - positions[i-1])
        arc_lengths[i] = arc_lengths[i-1] + segment_length
    
    total_length = arc_lengths[-1]
    total_time = t[-1]
    
    # =========================================================================
    # CALCULAR CUATERNIONES DE ORIENTACIÓN
    # =========================================================================
    # El yaw se calcula basado en la dirección de velocidad horizontal
    quaternions = np.zeros((n_points, 4))  # [w, x, y, z]
    
    yaw_angles = np.zeros(n_points)
    
    for i in range(n_points):
        vx, vy, vz = velocities[i]
        v_horiz = np.sqrt(vx**2 + vy**2)
        
        if v_horiz > min_velocity:
            # Yaw = atan2(vy, vx)
            yaw = np.arctan2(vy, vx)
        else:
            # Velocidad horizontal muy baja, usar dirección al siguiente punto
            if i < n_points - 1:
                dx = positions[i+1, 0] - positions[i, 0]
                dy = positions[i+1, 1] - positions[i, 1]
                if np.sqrt(dx**2 + dy**2) > 1e-6:
                    yaw = np.arctan2(dy, dx)
                else:
                    yaw = yaw_angles[i-1] if i > 0 else 0.0
            else:
                yaw = yaw_angles[i-1] if i > 0 else 0.0
        
        yaw_angles[i] = yaw
    
    # Unwrap yaw para evitar saltos de 2π
    yaw_angles = np.unwrap(yaw_angles)
    
    # Convertir yaw a cuaternión (solo rotación alrededor de z)
    # q = [cos(yaw/2), 0, 0, sin(yaw/2)]
    for i in range(n_points):
        yaw = yaw_angles[i]
        quaternions[i, 0] = np.cos(yaw / 2)  # w
        quaternions[i, 1] = 0.0               # x
        quaternions[i, 2] = 0.0               # y
        quaternions[i, 3] = np.sin(yaw / 2)  # z
    
    # =========================================================================
    # ASEGURAR CONTINUIDAD DE CUATERNIONES (mismo hemisferio)
    # =========================================================================
    # Evitar saltos de signo que causan problemas en interpolación
    for i in range(1, n_points):
        if np.dot(quaternions[i-1], quaternions[i]) < 0:
            quaternions[i] = -quaternions[i]
    
    # =========================================================================
    # ASEGURAR VELOCIDAD MÍNIMA
    # =========================================================================
    # En el punto inicial, si la velocidad es cero, usar la dirección al siguiente punto
    for i in range(n_points):
        v_mag = np.linalg.norm(velocities[i])
        if v_mag < min_velocity:
            if i < n_points - 1:
                direction = positions[i+1] - positions[i]
                dir_mag = np.linalg.norm(direction)
                if dir_mag > 1e-6:
                    velocities[i] = (direction / dir_mag) * min_velocity
    
    # =========================================================================
    # CREAR FUNCIONES DE INTERPOLACIÓN
    # =========================================================================
    # Interpolación cúbica para posición/velocidad (suavidad)
    pos_interp = interp1d(arc_lengths, positions, axis=0, kind='cubic', 
                          bounds_error=False, fill_value=(positions[0], positions[-1]))
    
    vel_interp = interp1d(arc_lengths, velocities, axis=0, kind='cubic',
                          bounds_error=False, fill_value=(velocities[0], velocities[-1]))
    
    # Para cuaterniones: usar scipy.spatial.transform.Slerp (robusto)
    # Convertir a formato scipy: [x, y, z, w] en lugar de [w, x, y, z]
    quats_scipy = np.zeros((n_points, 4))
    quats_scipy[:, 0] = quaternions[:, 1]  # x
    quats_scipy[:, 1] = quaternions[:, 2]  # y
    quats_scipy[:, 2] = quaternions[:, 3]  # z
    quats_scipy[:, 3] = quaternions[:, 0]  # w
    
    rotations = Rotation.from_quat(quats_scipy)
    quat_slerp = Slerp(arc_lengths, rotations)
    
    def position_by_arc_length(s):
        """Retorna posición [x, y, z] para longitud de arco s"""
        s_clipped = np.clip(s, 0, total_length)
        return pos_interp(s_clipped)
    
    def velocity_by_arc_length(s):
        """Retorna velocidad [vx, vy, vz] para longitud de arco s"""
        s_clipped = np.clip(s, 0, total_length)
        return vel_interp(s_clipped)
    
    def quaternion_by_arc_length(s):
        """Retorna cuaternión [w, x, y, z] usando SLERP para interpolación suave"""
        s_clipped = np.clip(s, arc_lengths[0] + 1e-6, arc_lengths[-1] - 1e-6)
        rot = quat_slerp(s_clipped)
        q_scipy = rot.as_quat()  # [x, y, z, w]
        # Convertir a [w, x, y, z]
        q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
        return q
    
    # =========================================================================
    # CALCULAR ESTADÍSTICAS
    # =========================================================================
    vel_magnitudes = np.linalg.norm(velocities, axis=1)
    
    if verbose:
        print("=" * 60)
        print("PMM Trajectory Loaded Successfully")
        print("=" * 60)
        print(f"  - Puntos: {n_points}")
        print(f"  - Longitud total: {total_length:.2f} m")
        print(f"  - Tiempo total: {total_time:.2f} s")
        print(f"  - Velocidad promedio: {total_length/total_time:.2f} m/s")
        print(f"  - Velocidad máxima: {vel_magnitudes.max():.2f} m/s")
        print(f"  - Posición inicial: [{positions[0, 0]:.2f}, {positions[0, 1]:.2f}, {positions[0, 2]:.2f}]")
        print(f"  - Posición final: [{positions[-1, 0]:.2f}, {positions[-1, 1]:.2f}, {positions[-1, 2]:.2f}]")
        print(f"  - Yaw inicial: {np.degrees(yaw_angles[0]):.1f}°")
        print("=" * 60)
    
    return {
        'position_by_arc_length': position_by_arc_length,
        'velocity_by_arc_length': velocity_by_arc_length,
        'quaternion_by_arc_length': quaternion_by_arc_length,
        'total_length': total_length,
        'total_time': total_time,
        'arc_lengths': arc_lengths,
        'positions': positions,
        'velocities': velocities,
        'quaternions': quaternions,
        'yaw_angles': yaw_angles,
        't': t
    }


def create_adaptive_waypoints(pmm_data, target_length, n_base_waypoints=30, 
                               angle_threshold_deg=15):
    """
    Crea waypoints adaptivos para PMM con más densidad en zonas de giro.
    
    Esta función genera waypoints que capturan mejor los giros bruscos
    de la trayectoria PMM, evitando discontinuidades en la interpolación
    CasADi.
    
    Args:
        pmm_data: Dict retornado por load_pmm_trajectory()
        target_length: Longitud objetivo de trayectoria (m)
        n_base_waypoints: Número base de waypoints
        angle_threshold_deg: Umbral de ángulo (°) para añadir puntos extra
    
    Returns:
        s_waypoints, pos_waypoints, vel_waypoints, quat_waypoints
    """
    pos_func = pmm_data['position_by_arc_length']
    vel_func = pmm_data['velocity_by_arc_length']
    quat_func = pmm_data['quaternion_by_arc_length']
    total_len = min(pmm_data['total_length'], target_length)
    
    # =========================================================================
    # PASO 1: Muestreo fino para detectar curvatura
    # =========================================================================
    n_samples = 1000
    s_fine = np.linspace(0.01, total_len - 0.01, n_samples)
    
    # Calcular "densidad requerida" basada en cambio angular
    # Donde hay más cambio de ángulo, necesitamos más puntos
    curvature_weight = np.ones(n_samples)
    
    for i in range(n_samples - 1):
        q1 = quat_func(s_fine[i])
        q2 = quat_func(s_fine[i+1])
        dot = abs(np.dot(q1, q2))
        angle_deg = 2 * np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi
        
        # Normalizar por el intervalo
        ds = s_fine[i+1] - s_fine[i]
        angle_rate = angle_deg / ds  # grados por metro
        
        # Peso: 1 en tramos rectos, más alto en curvas
        # angle_rate > 10°/m se considera curva
        if angle_rate > 5:
            curvature_weight[i] = 1 + (angle_rate / 10)
        else:
            curvature_weight[i] = 1.0
    
    curvature_weight[-1] = curvature_weight[-2]  # Copiar último valor
    
    # =========================================================================
    # PASO 2: Distribuir waypoints según curvatura
    # =========================================================================
    # Calcular integral de peso para distribución
    cumulative_weight = np.cumsum(curvature_weight)
    cumulative_weight = cumulative_weight / cumulative_weight[-1]  # Normalizar a [0, 1]
    
    # Generar n_base_waypoints distribuidos según la curvatura
    target_fractions = np.linspace(0, 1, n_base_waypoints)
    s_waypoints = np.interp(target_fractions, cumulative_weight, s_fine)
    
    # Asegurar extremos exactos
    s_waypoints[0] = 0.01
    s_waypoints[-1] = total_len - 0.01
    
    n_wp = len(s_waypoints)    # =========================================================================
    # PASO 3: Generar datos de waypoints
    # =========================================================================
    pos_waypoints = np.zeros((3, n_wp))
    vel_waypoints = np.zeros((3, n_wp))
    quat_waypoints = np.zeros((4, n_wp))
    
    for i, s in enumerate(s_waypoints):
        s_safe = np.clip(s, 0.01, total_len - 0.01)
        pos_waypoints[:, i] = pos_func(s_safe)
        vel_waypoints[:, i] = vel_func(s_safe)
        quat_waypoints[:, i] = quat_func(s_safe)
    
    # Asegurar continuidad de cuaterniones
    for i in range(1, n_wp):
        if np.dot(quat_waypoints[:, i-1], quat_waypoints[:, i]) < 0:
            quat_waypoints[:, i] = -quat_waypoints[:, i]
    
    # =========================================================================
    # PASO 4: Verificar calidad
    # =========================================================================
    max_angle = 0
    for i in range(n_wp - 1):
        q1 = quat_waypoints[:, i]
        q2 = quat_waypoints[:, i+1]
        dot = abs(np.dot(q1, q2))
        angle = 2 * np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi
        if angle > max_angle:
            max_angle = angle
    
    print(f"[PMM] Waypoints adaptivos: {n_wp} puntos")
    print(f"[PMM]   - Spacing min: {np.min(np.diff(s_waypoints)):.2f} m")
    print(f"[PMM]   - Spacing max: {np.max(np.diff(s_waypoints)):.2f} m")
    print(f"[PMM]   - Max angle change: {max_angle:.1f}°")
    
    return s_waypoints, pos_waypoints, vel_waypoints, quat_waypoints


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing PMM Trajectory Loader")
    print("=" * 60)
    
    try:
        pmm = load_pmm_trajectory(verbose=True)
        
        # Test interpolation
        print("\nTest de interpolación:")
        test_s_values = [0, 10, 20, 40, pmm['total_length']]
        for s in test_s_values:
            pos = pmm['position_by_arc_length'](s)
            vel = pmm['velocity_by_arc_length'](s)
            quat = pmm['quaternion_by_arc_length'](s)
            v_mag = np.linalg.norm(vel)
            print(f"  s={s:6.2f}m: pos=[{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}], "
                  f"|v|={v_mag:5.2f} m/s, q=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
        
        print("\n✓ Carga exitosa!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("  Ejecuta primero: python3 PPM.py")
