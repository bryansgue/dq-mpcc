#!/usr/bin/env python3
"""
EJEMPLO COMPLETO: Uso de todas las funciones de plotting_utils.py

Este script demuestra cómo usar las 18 funciones de visualización
disponibles en plotting_utils.py para diferentes análisis.
"""

import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import (
    # Configuración de figuras
    fancy_plots_4, fancy_plots_3, fancy_plots_1,
    # Estados
    plot_states_quaternion, plot_states_position,
    # Control
    plot_control_actions,
    # Velocidades
    plot_angular_velocities, plot_linear_velocities,
    # Costos
    plot_cost_total, plot_cost_orientation, plot_cost_translation, plot_cost_control,
    # Errores
    plot_norm_quat, plot_norm_real, plot_norm_dual,
    # Estabilidad
    plot_lyapunov, plot_lyapunov_dot,
    # Tiempo
    plot_time
)

# ============================================================================
# GENERACIÓN DE DATOS DE PRUEBA
# ============================================================================

def generate_test_data(N=300):
    """
    Genera datos sintéticos para demostración.
    
    Args:
        N: número de puntos de tiempo
        
    Returns:
        Diccionario con todos los datos de prueba
    """
    
    # Tiempo
    t = np.linspace(0, 30, N)
    
    # Quaterniones (con ligera perturbación)
    q_actual = np.array([
        0.95 + 0.05 * np.sin(0.1 * t),          # qw
        0.1 + 0.05 * np.cos(0.15 * t),          # qx
        0.05 + 0.05 * np.sin(0.12 * t),         # qy
        0.05 + 0.05 * np.cos(0.13 * t)          # qz
    ])
    q_desired = np.array([
        0.95 * np.ones_like(t),                 # qw
        0.1 * np.ones_like(t),                  # qx
        0.05 * np.ones_like(t),                 # qy
        0.05 * np.ones_like(t)                  # qz
    ])
    
    # Posiciones (trayectoria sinusoidal)
    p_actual = np.array([
        3 + 2 * np.sin(0.1 * t) + 0.1 * np.random.randn(N),  # x
        2 + 1.5 * np.cos(0.1 * t) + 0.1 * np.random.randn(N),  # y
        5 + 0.5 * np.sin(0.15 * t) + 0.1 * np.random.randn(N)  # z
    ])
    p_desired = np.array([
        3 + 2 * np.sin(0.1 * t),                 # x_d
        2 + 1.5 * np.cos(0.1 * t),               # y_d
        5 + 0.5 * np.sin(0.15 * t)               # z_d
    ])
    
    # Control (thrust + torques)
    F = 9.81 + 2 * np.sin(0.05 * t)  # Thrust
    M = np.array([
        0.05 * np.sin(0.1 * t),        # τx
        0.05 * np.cos(0.12 * t),       # τy
        0.02 * np.sin(0.15 * t)        # τz
    ])
    
    # Velocidades
    v = np.array([
        0.2 * np.cos(0.1 * t),                   # vx
        -0.15 * np.sin(0.1 * t),                 # vy
        0.075 * np.cos(0.15 * t)                 # vz
    ])
    w = np.array([
        0.01 * np.sin(0.1 * t),                  # ωx
        0.01 * np.cos(0.12 * t),                 # ωy
        0.005 * np.sin(0.15 * t)                 # ωz
    ])
    
    # Costos
    error_q = np.linalg.norm(q_actual - q_desired, axis=0) ** 2
    error_p = np.linalg.norm(p_actual - p_desired, axis=0) ** 2
    error_u = np.linalg.norm(np.vstack([F, M]), axis=0) ** 2
    cost_total = error_q + error_p
    
    # Norms de errores
    norm_q = np.linalg.norm(q_actual, axis=0)
    norm_real = np.linalg.norm(error_q)
    norm_dual = np.linalg.norm(error_p)
    
    # Lyapunov (función energética)
    V = np.cumsum(error_q + error_p) / 1000  # Integral de error
    V_dot = -np.diff(V, prepend=V[0])  # Derivada
    
    # Tiempos
    sample_time = 0.01 * np.ones_like(t)
    real_time = 0.01 + 0.002 * np.sin(0.1 * t)
    
    return {
        't': t,
        'q_actual': q_actual,
        'q_desired': q_desired,
        'p_actual': p_actual,
        'p_desired': p_desired,
        'F': F,
        'M': M,
        'v': v,
        'w': w,
        'cost_q': error_q,
        'cost_p': error_p,
        'cost_u': error_u,
        'cost_total': cost_total,
        'norm_q': norm_q,
        'norm_real': norm_real,
        'norm_dual': norm_dual,
        'V': V,
        'V_dot': V_dot,
        'sample_time': sample_time,
        'real_time': real_time
    }


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def example_1_quaternions(data, output_path="/tmp"):
    """Ejemplo 1: Graficar Quaterniones"""
    print("\n📊 Ejemplo 1: Quaterniones")
    print("-" * 50)
    
    fig, ax1, ax2, ax3, ax4 = fancy_plots_4()
    plot_states_quaternion(fig, ax1, ax2, ax3, ax4,
                          data['q_actual'][np.newaxis, :],
                          data['q_desired'][np.newaxis, :],
                          data['t'],
                          "Example_Quaternions",
                          output_path)
    print(f"✅ Guardado en {output_path}/Example_Quaternions.pdf")
    plt.close('all')


def example_2_position(data, output_path="/tmp"):
    """Ejemplo 2: Graficar Posiciones"""
    print("\n📊 Ejemplo 2: Posiciones")
    print("-" * 50)
    
    fig, ax1, ax2, ax3 = fancy_plots_3()
    plot_states_position(fig, ax1, ax2, ax3,
                        data['p_actual'],
                        data['p_desired'],
                        data['t'],
                        "Example_Position",
                        output_path)
    print(f"✅ Guardado en {output_path}/Example_Position.pdf")
    plt.close('all')


def example_3_velocities(data, output_path="/tmp"):
    """Ejemplo 3: Graficar Velocidades"""
    print("\n📊 Ejemplo 3: Velocidades")
    print("-" * 50)
    
    # Velocidades angulares
    fig, ax1, ax2, ax3 = fancy_plots_3()
    plot_angular_velocities(fig, ax1, ax2, ax3,
                           data['w'],
                           data['t'],
                           "Example_Angular_Velocities",
                           output_path)
    print(f"✅ Guardado: Angular Velocities")
    plt.close('all')
    
    # Velocidades lineales
    fig, ax1, ax2, ax3 = fancy_plots_3()
    plot_linear_velocities(fig, ax1, ax2, ax3,
                          data['v'],
                          data['t'],
                          "Example_Linear_Velocities",
                          output_path)
    print(f"✅ Guardado: Linear Velocities")
    plt.close('all')


def example_4_control(data, output_path="/tmp"):
    """Ejemplo 4: Graficar Acciones de Control"""
    print("\n📊 Ejemplo 4: Control")
    print("-" * 50)
    
    fig, ax1, ax2, ax3, ax4 = fancy_plots_4()
    plot_control_actions(fig, ax1, ax2, ax3, ax4,
                        data['F'].reshape(1, -1),
                        data['M'],
                        data['t'],
                        "Example_Control_Actions",
                        output_path)
    print(f"✅ Guardado: Control Actions")
    plt.close('all')


def example_5_costs(data, output_path="/tmp"):
    """Ejemplo 5: Graficar Costos Desglosados"""
    print("\n📊 Ejemplo 5: Costos Desglosados")
    print("-" * 50)
    
    # Costo total
    fig, ax = fancy_plots_1()
    plot_cost_total(fig, ax,
                   data['cost_total'].reshape(1, -1),
                   data['t'],
                   "Example_Cost_Total",
                   output_path)
    print(f"✅ Guardado: Cost Total")
    plt.close('all')
    
    # Costo de orientación
    fig, ax = fancy_plots_1()
    plot_cost_orientation(fig, ax,
                         data['cost_q'].reshape(1, -1),
                         data['t'],
                         "Example_Cost_Orientation",
                         output_path)
    print(f"✅ Guardado: Cost Orientation")
    plt.close('all')
    
    # Costo de traslación
    fig, ax = fancy_plots_1()
    plot_cost_translation(fig, ax,
                         data['cost_p'].reshape(1, -1),
                         data['t'],
                         "Example_Cost_Translation",
                         output_path)
    print(f"✅ Guardado: Cost Translation")
    plt.close('all')
    
    # Costo de control
    fig, ax = fancy_plots_1()
    plot_cost_control(fig, ax,
                     data['cost_u'].reshape(1, -1),
                     data['t'],
                     "Example_Cost_Control",
                     output_path)
    print(f"✅ Guardado: Cost Control")
    plt.close('all')


def example_6_error_analysis(data, output_path="/tmp"):
    """Ejemplo 6: Análisis de Errores"""
    print("\n📊 Ejemplo 6: Análisis de Errores")
    print("-" * 50)
    
    # Norma quaternión
    fig, ax = fancy_plots_1()
    plot_norm_quat(fig, ax,
                  np.linalg.norm(data['q_actual'], axis=0).reshape(1, -1),
                  data['t'],
                  "Example_Norm_Quat")
    print(f"✅ Guardado: Norm Quaternion")
    plt.close('all')
    
    # Norma parte real
    fig, ax = fancy_plots_1()
    plot_norm_real(fig, ax,
                  data['cost_q'].reshape(1, -1),
                  data['t'],
                  "Example_Norm_Real")
    print(f"✅ Guardado: Norm Real")
    plt.close('all')
    
    # Norma parte dual
    fig, ax = fancy_plots_1()
    plot_norm_dual(fig, ax,
                  data['cost_p'].reshape(1, -1),
                  data['t'],
                  "Example_Norm_Dual")
    print(f"✅ Guardado: Norm Dual")
    plt.close('all')


def example_7_stability(data, output_path="/tmp"):
    """Ejemplo 7: Análisis de Estabilidad"""
    print("\n📊 Ejemplo 7: Estabilidad (Lyapunov)")
    print("-" * 50)
    
    # Función de Lyapunov
    fig, ax = fancy_plots_1()
    plot_lyapunov(fig, ax,
                 data['V'].reshape(1, -1),
                 data['t'],
                 "Example_Lyapunov")
    print(f"✅ Guardado: Lyapunov Function")
    plt.close('all')
    
    # Derivada de Lyapunov
    fig, ax = fancy_plots_1()
    plot_lyapunov_dot(fig, ax,
                     data['V_dot'].reshape(1, -1),
                     data['t'],
                     "Example_Lyapunov_Dot")
    print(f"✅ Guardado: Lyapunov Derivative")
    plt.close('all')


def example_8_timing(data, output_path="/tmp"):
    """Ejemplo 8: Análisis de Tiempos"""
    print("\n📊 Ejemplo 8: Análisis de Tiempos")
    print("-" * 50)
    
    fig, ax = fancy_plots_1()
    plot_time(fig, ax,
             data['sample_time'].reshape(1, -1),
             data['real_time'].reshape(1, -1),
             data['t'],
             "Example_Time_Analysis",
             output_path)
    print(f"✅ Guardado: Time Analysis")
    plt.close('all')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import os
    
    print("\n" + "="*70)
    print("DEMOSTRACIÓN COMPLETA: plotting_utils.py")
    print("="*70)
    
    # Crear directorio de salida
    output_dir = "/tmp/plotting_examples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar datos de prueba
    print("\n📊 Generando datos de prueba...")
    data = generate_test_data(N=300)
    print(f"✅ Datos generados: {len(data['t'])} puntos de tiempo")
    
    # Ejecutar ejemplos
    print("\n" + "="*70)
    print("EJECUTANDO EJEMPLOS")
    print("="*70)
    
    try:
        example_1_quaternions(data, output_dir)
        example_2_position(data, output_dir)
        example_3_velocities(data, output_dir)
        example_4_control(data, output_dir)
        example_5_costs(data, output_dir)
        example_6_error_analysis(data, output_dir)
        example_7_stability(data, output_dir)
        example_8_timing(data, output_dir)
        
        print("\n" + "="*70)
        print("✅ DEMOSTRACIÓN COMPLETADA CON ÉXITO")
        print("="*70)
        print(f"\n📁 Archivos guardados en: {output_dir}")
        print(f"\n🎨 Total de gráficas generadas:")
        files = os.listdir(output_dir)
        pdfs = len([f for f in files if f.endswith('.pdf')])
        print(f"   - PDF files: {pdfs}")
        print(f"   - PNG files: {len([f for f in files if f.endswith('.png')])}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

