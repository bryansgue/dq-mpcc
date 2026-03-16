"""
Path Planning 3D - Point Mass Model (PMM)
Optimizado para:
- Tiempo mínimo
- Aprovechar la inercia (mantener dirección)
- Trayectorias suaves sin cambios abruptos
- No desacelerar innecesariamente

Estados: [x, y, z, vx, vy, vz] (6 estados)
Controles: [ax, ay, az] (3 controles - aceleración)

Dinámica PMM:
  ṗ = v
  v̇ = a
"""

import numpy as np
import casadi as ca
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# PARÁMETROS
# =============================================================================
g_val = 9.81   # gravedad [m/s²]

# =============================================================================
# CONFIGURACIÓN DEL PROBLEMA
# =============================================================================
n_states = 6     # [x, y, z, vx, vy, vz]
n_controls = 3   # [ax, ay, az]

# Gates (posiciones y orientaciones)
gate_positions = np.array([
    [0, 0, 2],       # Gate 0 (inicio)
    [-2.5, 6, 2],    # Gate 1
    [3.5, 0, 2],     # Gate 2
    [-5, -6, 2],     # Gate 3
    [-5, -6, 4],      # Gate 4
    [8, -4, 4],      # Gate 5
    [8, 6, 2],       # Gate 6
    [10, 0, 3],      # Gate 7
    [10, -6, 3],     # Gate 8 (final)
], dtype=float)

# Normales de los gates (dirección de cruce)
gate_normals = np.array([
    [-0.368, 0.882, 0.294],
    [0.659, 0.0, 0.753],
    [-0.203, -0.976, -0.081],
    [-0.647, -0.740, -0.185],
   [0.647, 0.740, 0.185],
    [0.623, 0.778, -0.078],
    [0.436, 0.873, -0.218],
    [0.164, -0.983, -0.082],
    [0.0, -0.894, -0.447],
], dtype=float)

# Normalizar
for i in range(len(gate_normals)):
    gate_normals[i] /= np.linalg.norm(gate_normals[i])

n_gates = len(gate_positions)
N_segments = n_gates - 1
gate_radius = 1.0
safety_margin = 0.15

print("=" * 60)
print("PATH PLANNING 3D - POINT MASS MODEL (PMM)")
print("Optimizado para inercia y suavidad")
print("=" * 60)
for i, (pos, n) in enumerate(zip(gate_positions, gate_normals)):
    print(f"Gate {i}: pos={pos}, n={np.round(n, 3)}")

# =============================================================================
# DISCRETIZACIÓN
# =============================================================================
N_per_segment = 20  # Pasos por segmento (reducido para mejor convergencia)
N = N_per_segment * N_segments

# Índices donde debe estar exactamente en cada gate
k_gate = [i * N_per_segment for i in range(n_gates)]
delta_k = N_per_segment // 2

# Estimación inicial del tiempo
total_dist = sum(np.linalg.norm(gate_positions[i+1] - gate_positions[i]) for i in range(N_segments))
v_avg_est = 6.0
T_est = total_dist / v_avg_est

print(f"\nN total = {N}")
print(f"Índices de gates: {k_gate}")
print(f"Distancia total: {total_dist:.1f}m")
print(f"Tiempo estimado: {T_est:.1f}s")

# =============================================================================
# VARIABLES DE OPTIMIZACIÓN
# =============================================================================
X = ca.MX.sym('X', n_states, N + 1)
U = ca.MX.sym('U', n_controls, N)
T = ca.MX.sym('T')  # Tiempo total

dt = T / N  # dt variable

# =============================================================================
# FUNCIÓN DE COSTO - OPTIMIZADA PARA INERCIA Y SUAVIDAD
# =============================================================================
# Pesos - BALANCE ÓPTIMO
w_T = 100.0         # Minimizar tiempo
w_a = 0.001         # Penalizar aceleración (bajo)
w_da = 0.5          # Suavidad en jerk
w_dda = 0.2         # Suavidad en snap
w_v_change = 0.02   # Penalizar cambios de dirección (muy bajo para inercia)
w_center = 1.0      # Pasar cerca del centro

cost = w_T * T

# Costo de controles y suavidad
for k in range(N):
    a_k = U[:, k]
    v_k = X[3:6, k]
    
    # Penalizar aceleración (bajo)
    cost += w_a * ca.sumsqr(a_k) * dt
    
    # Penalizar jerk (cambio de aceleración) - SUAVIDAD
    if k > 0:
        da = (U[:, k] - U[:, k-1]) / dt
        cost += w_da * ca.sumsqr(da) * dt
    
    # Penalizar cambios bruscos de dirección de velocidad (INERCIA)
    if k > 0:
        v_prev = X[3:6, k-1]
        v_curr = X[3:6, k]
        dv = v_curr - v_prev
        cost += w_v_change * ca.sumsqr(dv) * dt

# Costo de pasar cerca del centro (bajo para flexibilidad)
for i in range(1, n_gates):
    k_i = k_gate[i]
    p_gate = X[0:3, k_i]
    gpos = gate_positions[i]
    cost += w_center * ca.sumsqr(p_gate - gpos)

# =============================================================================
# RESTRICCIONES
# =============================================================================
g = []
lbg = []
ubg = []

# --- 1. Dinámica PMM (Multiple Shooting con RK4) ---
def pmm_dynamics(x, u):
    """Dinámica Point Mass Model: ṗ = v, v̇ = a"""
    p = x[0:3]
    v = x[3:6]
    a = u
    return ca.vertcat(v, a)

for k in range(N):
    xk = X[:, k]
    uk = U[:, k]
    
    # RK4
    k1 = pmm_dynamics(xk, uk)
    k2 = pmm_dynamics(xk + dt/2 * k1, uk)
    k3 = pmm_dynamics(xk + dt/2 * k2, uk)
    k4 = pmm_dynamics(xk + dt * k3, uk)
    x_next = xk + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    g.append(X[:, k+1] - x_next)
    lbg.extend([0] * n_states)
    ubg.extend([0] * n_states)

# --- 2. Condiciones iniciales ---
x0_pos = gate_positions[0]
# Velocidad inicial cero (hover antes de empezar)
x0_vel = np.array([0, 0, 0])

g.append(X[0:3, 0] - x0_pos)
lbg.extend([0, 0, 0])
ubg.extend([0, 0, 0])

g.append(X[3:6, 0] - x0_vel)
lbg.extend([0, 0, 0])
ubg.extend([0, 0, 0])

# --- 3. Condiciones finales ---
xf_pos = gate_positions[-1]

g.append(X[0:3, N] - xf_pos)
lbg.extend([0, 0, 0])
ubg.extend([0, 0, 0])

# Velocidad final: permitir algo de velocidad (no frenar completamente)
v_final_max = 2.0
g.append(X[3:6, N])
lbg.extend([-v_final_max, -v_final_max, -v_final_max])
ubg.extend([v_final_max, v_final_max, v_final_max])

# --- 4. Restricciones de cruce por gates ---
min_crossing_dist = 0.10  # Relajado
max_perp_dist = 0.05      # Tolerancia perpendicular relajada
max_r = gate_radius - safety_margin  # Radio máximo permitido (0.85m)

for i in range(1, n_gates):
    k_i = k_gate[i]
    gpos = gate_positions[i]
    gnorm = gate_normals[i]
    
    p_i = X[0:3, k_i]
    d = p_i - gpos
    
    # Distancia perpendicular al plano ≈ 0
    dist_perp = ca.dot(d, gnorm)
    g.append(dist_perp)
    lbg.append(-max_perp_dist)
    ubg.append(max_perp_dist)
    
    # Distancia radial dentro del gate (permite usar todo el espacio)
    d_plane = d - dist_perp * gnorm
    dist_radial_sq = ca.sumsqr(d_plane)
    g.append(dist_radial_sq)
    lbg.append(0)
    ubg.append(max_r**2)

# --- 5. Restricciones de dirección de cruce ---
for i in range(1, n_gates):
    k_i = k_gate[i]
    gpos = gate_positions[i]
    gnorm = gate_normals[i]
    
    # Antes del gate
    k_before = max(0, k_i - delta_k)
    p_before = X[0:3, k_before]
    sd_before = ca.dot(p_before - gpos, gnorm)
    g.append(sd_before)
    lbg.append(-1000)
    ubg.append(-min_crossing_dist)
    
    # Después del gate
    k_after = min(N, k_i + delta_k) if i < n_gates - 1 else N
    p_after = X[0:3, k_after]
    sd_after = ca.dot(p_after - gpos, gnorm)
    g.append(sd_after)
    lbg.append(min_crossing_dist)
    ubg.append(1000)

# --- 6. Restricciones de colisión cerca de gates ---
# Reducido para menos restricciones
collision_range = 2
for i in range(1, n_gates):
    k_i = k_gate[i]
    gpos = gate_positions[i]
    gnorm = gate_normals[i]
    
    for offset in range(-collision_range, collision_range + 1):
        k_check = k_i + offset
        if k_check < 0 or k_check > N:
            continue
        
        p = X[0:3, k_check]
        d = p - gpos
        dist_perp = ca.dot(d, gnorm)
        d_plane = d - dist_perp * gnorm
        dist_radial_sq = ca.sumsqr(d_plane)
        
        g.append(dist_radial_sq)
        lbg.append(0)
        ubg.append(max_r**2)

# --- 7. Monotonicidad (no volver atrás) - OPCIONAL, puede causar infeasibilidad ---
# Comentado para mejor convergencia - las restricciones de cruce ya garantizan dirección correcta
# for i in range(1, n_gates - 1):
#     k_i = k_gate[i]
#     k_next = k_gate[i + 1]
#     gpos = gate_positions[i]
#     gnorm = gate_normals[i]
#     
#     k_mid = (k_i + k_next) // 2
#     p_mid = X[0:3, k_mid]
#     sd_mid = ca.dot(p_mid - gpos, gnorm)
#     g.append(sd_mid)
#     lbg.append(0.05)
#     ubg.append(1000)

# =============================================================================
# FORMULAR NLP
# =============================================================================
opt_vars = ca.vertcat(
    ca.reshape(X, -1, 1),
    ca.reshape(U, -1, 1),
    T
)

n_vars = opt_vars.shape[0]
g_vec = ca.vertcat(*g)

nlp = {'f': cost, 'x': opt_vars, 'g': g_vec}

opts = {
    'ipopt.max_iter': 3000,
    'ipopt.tol': 1e-4,
    'ipopt.acceptable_tol': 1e-3,
    'ipopt.acceptable_iter': 15,
    'ipopt.print_level': 5,
    'print_time': 1,
    'ipopt.mu_strategy': 'adaptive',
    'ipopt.mu_init': 0.1,
    'ipopt.nlp_scaling_method': 'gradient-based',
}

solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# =============================================================================
# LÍMITES DE VARIABLES
# =============================================================================
lbx = np.full(n_vars, -np.inf)
ubx = np.full(n_vars, np.inf)

# Límites de estados
v_max = 15.0           # Velocidad máxima [m/s]
a_max = 20.0           # Aceleración máxima [m/s²] (~2g)
z_min, z_max = 0.0, 15.0

for k in range(N + 1):
    idx = k * n_states
    
    # Posición z
    lbx[idx + 2] = z_min
    ubx[idx + 2] = z_max
    
    # Velocidades
    for j in range(3, 6):
        lbx[idx + j] = -v_max
        ubx[idx + j] = v_max

# Límites de controles (aceleración)
ctrl_offset = n_states * (N + 1)
for k in range(N):
    idx = ctrl_offset + k * n_controls
    lbx[idx:idx+3] = -a_max
    ubx[idx:idx+3] = a_max

# Tiempo
lbx[-1] = 1.0
ubx[-1] = 60.0

# =============================================================================
# INICIALIZACIÓN INTELIGENTE
# =============================================================================
x0_init = np.zeros(n_vars)

# Interpolar posiciones y velocidades entre gates
for k in range(N + 1):
    seg = min(k // N_per_segment, N_segments - 1)
    t_local = (k - seg * N_per_segment) / N_per_segment
    
    pos_start = gate_positions[seg]
    pos_end = gate_positions[seg + 1]
    pos = (1 - t_local) * pos_start + t_local * pos_end
    
    # Velocidad inicial: dirección hacia el siguiente gate
    vel_dir = pos_end - pos_start
    vel_mag = np.linalg.norm(vel_dir)
    if vel_mag > 0:
        vel = (vel_dir / vel_mag) * v_avg_est
    else:
        vel = np.zeros(3)
    
    idx = k * n_states
    x0_init[idx:idx + 3] = pos
    x0_init[idx + 3:idx + 6] = vel

# Controles iniciales (aceleración cero)
for k in range(N):
    idx = ctrl_offset + k * n_controls
    x0_init[idx:idx + 3] = 0

x0_init[-1] = T_est

print(f"\nVariables: {n_vars}")
print(f"Restricciones: {len(lbg)}")

# =============================================================================
# RESOLVER
# =============================================================================
print("\n" + "=" * 60)
print("RESOLVIENDO OCP...")
print("=" * 60)

solution = solver(x0=x0_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

# =============================================================================
# EXTRAER RESULTADOS
# =============================================================================
sol = solution['x'].full().flatten()

X_opt = sol[:n_states * (N + 1)].reshape((N + 1, n_states)).T
U_opt = sol[n_states * (N + 1):n_states * (N + 1) + n_controls * N].reshape((N, n_controls)).T
T_opt = sol[-1]

dt_opt = T_opt / N
t_traj = np.linspace(0, T_opt, N + 1)

print("\n" + "=" * 60)
print("RESULTADO")
print("=" * 60)
print(f"Tiempo óptimo: {T_opt:.2f} s")

# =============================================================================
# VERIFICACIÓN DE CRUCE
# =============================================================================
print("\n" + "-" * 60)
print("VERIFICACIÓN DE CRUCE POR GATES")
print("-" * 60)

all_ok = True
for i in range(1, n_gates):
    k_i = k_gate[i]
    gpos = gate_positions[i]
    gnorm = gate_normals[i]
    
    p_i = X_opt[0:3, k_i]
    d = p_i - gpos
    dist_perp = np.dot(d, gnorm)
    d_plane = d - dist_perp * gnorm
    dist_radial = np.linalg.norm(d_plane)
    
    k_before = max(0, k_i - delta_k)
    k_after = min(N, k_i + delta_k) if i < n_gates - 1 else N
    
    sd_before = np.dot(X_opt[0:3, k_before] - gpos, gnorm)
    sd_after = np.dot(X_opt[0:3, k_after] - gpos, gnorm)
    
    ok = (sd_before < 0) and (sd_after > 0) and (dist_radial < gate_radius)
    
    print(f"\nGate {i}:")
    print(f"  En gate (t={t_traj[k_i]:.2f}s): r={dist_radial:.4f}m, perp={dist_perp:.4f}m")
    print(f"  Antes: signed_dist={sd_before:.4f}m {'(<0 ✓)' if sd_before < 0 else '(ERROR)'}")
    print(f"  Después: signed_dist={sd_after:.4f}m {'(>0 ✓)' if sd_after > 0 else '(ERROR)'}")
    print(f"  {'[✓ CRUCE CORRECTO]' if ok else '[✗ ERROR]'}")
    
    if not ok:
        all_ok = False

if all_ok:
    print("\n" + "=" * 60)
    print("RESULTADO FINAL: TODOS LOS GATES CRUZADOS ✓")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("ADVERTENCIA: ALGUNOS GATES NO SE CRUZARON CORRECTAMENTE")
    print("=" * 60)

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================
np.save('xref_optimo_3D_PMM.npy', X_opt)
np.save('uref_optimo_3D_PMM.npy', U_opt)
np.save('tref_optimo_3D_PMM.npy', t_traj)

print(f"\nArchivos guardados:")
print(f"  xref_optimo_3D_PMM.npy: {X_opt.shape}")
print(f"  uref_optimo_3D_PMM.npy: {U_opt.shape}")

# =============================================================================
# MÉTRICAS DE SUAVIDAD
# =============================================================================
vel_magnitudes = np.linalg.norm(X_opt[3:6, :], axis=0)
acc_magnitudes = np.linalg.norm(U_opt, axis=0)

# Calcular jerk (derivada de aceleración)
jerk = np.diff(U_opt, axis=1) / dt_opt
jerk_magnitudes = np.linalg.norm(jerk, axis=0)

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"Gates: {n_gates}")
print(f"Tiempo óptimo: {T_opt:.2f}s")
print(f"Distancia total: {total_dist:.1f}m")
print(f"Vel promedio: {total_dist/T_opt:.2f} m/s")
print(f"Vel máxima: {vel_magnitudes.max():.2f} m/s")
print(f"Vel mínima: {vel_magnitudes.min():.2f} m/s")
print(f"Acc máxima: {acc_magnitudes.max():.2f} m/s²")
print(f"Jerk máximo: {jerk_magnitudes.max():.2f} m/s³")
print(f"Jerk promedio: {jerk_magnitudes.mean():.2f} m/s³")
print("=" * 60)

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# Trayectoria 3D
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(X_opt[0, :], X_opt[1, :], X_opt[2, :], 'b-', linewidth=2, label='Trayectoria')

# Dibujar gates
for i, (pos, n) in enumerate(zip(gate_positions, gate_normals)):
    theta = np.linspace(0, 2*np.pi, 50)
    
    if abs(n[2]) < 0.9:
        perp1 = np.cross(n, [0, 0, 1])
    else:
        perp1 = np.cross(n, [1, 0, 0])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(n, perp1)
    
    circle = np.array([pos + gate_radius * (np.cos(t) * perp1 + np.sin(t) * perp2) for t in theta])
    ax1.plot(circle[:, 0], circle[:, 1], circle[:, 2], 'g-', linewidth=2)
    ax1.scatter(*pos, c='r', s=50, marker='o')
    ax1.quiver(pos[0], pos[1], pos[2], n[0]*0.5, n[1]*0.5, n[2]*0.5, color='r', arrow_length_ratio=0.3)

# Ajustar límites EXACTOS al espacio usado (min/max de la trayectoria)
x_min, x_max = X_opt[0, :].min() - gate_radius, X_opt[0, :].max() + gate_radius
y_min, y_max = X_opt[1, :].min() - gate_radius, X_opt[1, :].max() + gate_radius
z_min_plot, z_max = X_opt[2, :].min() - gate_radius, X_opt[2, :].max() + gate_radius

ax1.set_xlim([x_min, x_max])
ax1.set_ylim([y_min, y_max])
ax1.set_zlim([z_min_plot, z_max])

# Misma escala en los 3 ejes, pero cada uno con su rango real
ax1.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min_plot])

ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_zlabel('Z [m]')
ax1.set_title('Trayectoria 3D - PMM con Inercia')
ax1.legend()

# Velocidad
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_traj, X_opt[3, :], 'r-', label='vx')
ax2.plot(t_traj, X_opt[4, :], 'g-', label='vy')
ax2.plot(t_traj, X_opt[5, :], 'b-', label='vz')
ax2.plot(t_traj, vel_magnitudes, 'k--', linewidth=2, label='|v|')
ax2.set_xlabel('Tiempo [s]')
ax2.set_ylabel('Velocidad [m/s]')
ax2.set_title('Perfil de Velocidad')
ax2.legend()
ax2.grid(True)

# Aceleración
ax3 = fig.add_subplot(2, 2, 3)
t_ctrl = np.linspace(0, T_opt, N)
ax3.plot(t_ctrl, U_opt[0, :], 'r-', label='ax')
ax3.plot(t_ctrl, U_opt[1, :], 'g-', label='ay')
ax3.plot(t_ctrl, U_opt[2, :], 'b-', label='az')
ax3.plot(t_ctrl, acc_magnitudes, 'k--', linewidth=2, label='|a|')
ax3.set_xlabel('Tiempo [s]')
ax3.set_ylabel('Aceleración [m/s²]')
ax3.set_title('Perfil de Aceleración (Control)')
ax3.legend()
ax3.grid(True)

# Jerk (suavidad)
ax4 = fig.add_subplot(2, 2, 4)
t_jerk = np.linspace(0, T_opt, N-1)
ax4.plot(t_jerk, jerk_magnitudes, 'm-', linewidth=1.5)
ax4.set_xlabel('Tiempo [s]')
ax4.set_ylabel('Jerk [m/s³]')
ax4.set_title('Perfil de Jerk (Suavidad)')
ax4.grid(True)

plt.tight_layout()
plt.savefig('path_3D_PMM.png', dpi=150)
plt.show()
