# Guía: Interpolación con Waypoints para MPCC

## ¿Por qué son necesarios los waypoints?

### Problema fundamental
CasADi necesita **expresiones simbólicas** que pueda derivar automáticamente para el optimizador MPCC. No puede llamar a funciones Python arbitrarias durante la optimización.

### Solución: Waypoints discretos
Creamos puntos de referencia discretos que CasADi puede interpolar usando splines/polinomios simbólicos.

---

## Función `create_mpcc_waypoints()`

### Propósito
Convierte funciones Python continuas en waypoints discretos optimizados para CasADi.

### Firma
```python
s_waypoints, pos_waypoints, vel_waypoints, quat_waypoints = create_mpcc_waypoints(
    position_by_arc_length,      # Función: s -> [x, y, z]
    velocity_by_arc_length,      # Función: s -> [vx, vy, vz]
    quaternion_by_arc_length,    # Función: s -> [qw, qx, qy, qz]
    target_path_length,          # Longitud total [m]
    n_waypoints=30               # Número de waypoints
)
```

### Retorna
- **s_waypoints**: Longitudes de arco uniformemente espaciadas `[0, L]`
- **pos_waypoints**: Posiciones en waypoints `[3, N]`
- **vel_waypoints**: Velocidades **normalizadas** (vectores tangentes) `[3, N]`
- **quat_waypoints**: Quaterniones con **corrección de hemisferio** `[4, N]`

---

## Características clave

### 1. Corrección de hemisferio para quaterniones ⚠️
**CRÍTICO**: Los quaterniones `q` y `-q` representan la **misma rotación**.

```python
# Sin corrección: interpolación puede "dar la vuelta larga"
q1 = [0.9, 0.1, 0.1, 0.1]   # θ ≈ 20°
q2 = [-0.9, -0.1, -0.1, -0.1]  # θ ≈ 20° (misma rotación!)
# Interpolación lineal incorrecta: pasará por θ = 180° ❌

# Con corrección: ambos en mismo hemisferio
q1 = [0.9, 0.1, 0.1, 0.1]
q2 = [0.9, 0.1, 0.1, 0.1]  # Invertido si dot(q1,q2) < 0
# Interpolación lineal correcta: transición suave ✓
```

**Implementación**:
```python
for i in range(1, n_waypoints):
    if np.dot(quat_waypoints[:, i-1], quat_waypoints[:, i]) < 0:
        quat_waypoints[:, i] = -quat_waypoints[:, i]
```

### 2. Normalización de velocidades
Las velocidades se usan como **vectores tangentes** (dirección, no magnitud).

```python
# El MPCC controla la velocidad de avance (u_s) separadamente
# Solo necesitamos la DIRECCIÓN de la trayectoria
vel_waypoints[:, i] /= np.linalg.norm(vel_waypoints[:, i])
```

---

## ¿Cuántos waypoints usar?

### Regla general
```
spacing = path_length / (n_waypoints - 1)
```

| Trayectoria | N_waypoints recomendado | Spacing aprox. |
|-------------|------------------------|----------------|
| 80m suave   | 30                     | 2.7 m          |
| 80m con curvas cerradas | 50        | 1.6 m          |
| 200m autopista | 50-80               | 2.5-4 m        |
| Circuito racing | 100+                | < 1 m          |

### Trade-offs

#### Pocos waypoints (10-20)
- ✅ Construcción rápida del problema
- ✅ Optimización más rápida
- ❌ Errores de interpolación en curvas cerradas
- ❌ Puede perder detalles de la trayectoria

#### Muchos waypoints (100+)
- ✅ Alta precisión
- ✅ Captura todos los detalles
- ❌ Construcción lenta del problema
- ❌ Optimización más lenta (más variables internas)

### Recomendación práctica
**Empezar con 30**, luego ajustar según `verify_waypoint_interpolation_quality()`:

```python
if pos_error_max > 0.01:  # Error > 1cm
    N_WAYPOINTS = 50  # Aumentar resolución
```

---

## Función de verificación

### `verify_waypoint_interpolation_quality()`

Compara waypoints con funciones originales para detectar errores de discretización.

```python
pos_error_max, quat_error_max, quality_ok = verify_waypoint_interpolation_quality(
    s_waypoints, pos_waypoints, quat_waypoints,
    position_by_arc_length, quaternion_by_arc_length
)
```

### Criterios de calidad
- **Excelente**: `pos_error_max < 0.005 m` (5mm)
- **Bueno**: `pos_error_max < 0.01 m` (1cm) ✓
- **Marginal**: `pos_error_max < 0.05 m` (5cm)
- **Pobre**: `pos_error_max > 0.05 m` ❌ Aumentar waypoints

---

## Ejemplo de uso en main()

### Antes (código largo y difícil de mantener)
```python
# 60+ líneas de código inline
N_WAYPOINTS = 30
s_waypoints = np.linspace(...)
pos_waypoints = np.zeros(...)
# ... bucles, correcciones, verificaciones ...
```

### Después (limpio y modular)
```python
# 3 líneas principales + verificación
s_waypoints, pos_waypoints, vel_waypoints, quat_waypoints = create_mpcc_waypoints(
    position_by_arc_length, velocity_by_arc_length, quaternion_by_arc_length,
    target_path_length, n_waypoints=30
)

pos_error, quat_error, ok = verify_waypoint_interpolation_quality(
    s_waypoints, pos_waypoints, quat_waypoints,
    position_by_arc_length, quaternion_by_arc_length
)

if not ok:
    rospy.logwarn("Consider increasing N_WAYPOINTS")
```

---

## Alternativas (NO recomendadas en tu caso)

### Opción 1: Expresión analítica pura
```python
# Si tu trayectoria fuera completamente analítica:
s_sym = ca.SX.sym('s')
x_expr = ca.sin(s_sym / 10.0) * 7 + 3
y_expr = ca.sin(s_sym / 5.0) * 7
# ...
```
**Problema**: Tu mapeo `s → t` usa CubicSpline (numérico), no analítico.

### Opción 2: Tabla lookup densa
```python
# Almacenar 1000+ puntos y hacer lookup durante optimización
```
**Problema**: Muy lento, degrada performance del solver.

---

## Debugging común

### Error: "Interpolación oscila"
**Causa**: Pocos waypoints en zona de alta curvatura  
**Solución**: Aumentar `N_WAYPOINTS` a 50-100

### Error: "Quaternion gira 180° súbitamente"
**Causa**: Falta corrección de hemisferio  
**Solución**: Ya implementada en `create_mpcc_waypoints()` ✓

### Error: "Solver muy lento"
**Causa**: Demasiados waypoints (200+)  
**Solución**: Reducir a 30-80, verificar calidad con `verify_...`

### Error: "Velocidad no sigue curvas"
**Causa**: Velocidades no normalizadas  
**Solución**: Ya implementada normalización ✓

---

## Performance esperado

Con 30 waypoints:
- Construcción del problema: **< 5 segundos**
- Tiempo de optimización por iteración: **< 10 ms** (@100Hz)
- Error de interpolación: **< 1 cm**

Con 100 waypoints:
- Construcción del problema: **10-15 segundos**
- Tiempo de optimización: **15-20 ms** (puede no alcanzar 100Hz)
- Error de interpolación: **< 1 mm**

---

## Conclusión

✅ **Tu implementación actual es correcta y eficiente**  
✅ **30 waypoints es un buen punto de partida**  
✅ **La refactorización mejora legibilidad sin cambiar funcionalidad**  
✅ **Usa verificación de calidad para ajustar según necesidad**

**No es necesario cambiar el enfoque de waypoints**, solo mantenerlo modular y bien documentado.
