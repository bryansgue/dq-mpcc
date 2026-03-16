#!/usr/bin/env python
"""
Diagnostic script to analyze velocity behavior in identification data
Checks for constant velocities, plateaus, and movement variability
"""

import numpy as np
import scipy.io as sio
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("DIAGNOSTIC: VELOCITY ANALYSIS")
print("="*70)

# Load data
mat_file = os.path.join(script_dir, "Dual_cost_identification.mat")
try:
    data = sio.loadmat(mat_file)
    X = data['X']
    t = data['t']
    print(f"\n✓ Loaded: {mat_file}")
    print(f"  X shape: {X.shape}")
    print(f"  t shape: {t.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Extract first experiment
exp_idx = 0
X_exp = X[exp_idx, :, :].T  # [N_steps x 14]
t_exp = t[exp_idx, :].flatten()

# State indices (0-indexed)
# 8:11 = angular velocities (wx, wy, wz)
# 11:14 = linear velocities (vx, vy, vz)

N_steps = X_exp.shape[0]
print(f"\nExperiment 0: {N_steps} time steps")

# ============================================================
# 1. ANALYZE LINEAR VELOCITIES
# ============================================================
print("\n" + "="*70)
print("LINEAR VELOCITIES (v_x, v_y, v_z)")
print("="*70)

v_x = X_exp[:, 11]
v_y = X_exp[:, 12]
v_z = X_exp[:, 13]

for i, v_name in enumerate(['v_x', 'v_y', 'v_z']):
    v = X_exp[:, 11+i]
    
    # Count how many steps are identical
    v_diff = np.diff(v)
    n_identical = np.sum(np.abs(v_diff) < 1e-10)
    pct_identical = 100 * n_identical / (N_steps - 1)
    
    # Standard deviation (variability)
    std_v = np.std(v)
    mean_v = np.mean(v)
    
    # Max and min
    max_v = np.max(v)
    min_v = np.min(v)
    
    # Consecutive identical values
    consecutive_identical = []
    count = 0
    for diff in v_diff:
        if np.abs(diff) < 1e-10:
            count += 1
        else:
            if count > 0:
                consecutive_identical.append(count)
            count = 0
    
    if consecutive_identical:
        max_consecutive = max(consecutive_identical)
    else:
        max_consecutive = 0
    
    print(f"\n{v_name}:")
    print(f"  Mean: {mean_v:.6f}")
    print(f"  Std Dev: {std_v:.6e}")
    print(f"  Range: [{min_v:.6f}, {max_v:.6f}]")
    print(f"  Steps with IDENTICAL velocity: {n_identical}/{N_steps-1} ({pct_identical:.2f}%)")
    print(f"  Longest consecutive identical: {max_consecutive} steps")
    print(f"  Max velocity change per step: {np.max(np.abs(v_diff)):.6e}")
    print(f"  Mean velocity change per step: {np.mean(np.abs(v_diff)):.6e}")

# ============================================================
# 2. ANALYZE ANGULAR VELOCITIES
# ============================================================
print("\n" + "="*70)
print("ANGULAR VELOCITIES (ω_x, ω_y, ω_z)")
print("="*70)

for i, w_name in enumerate(['ω_x', 'ω_y', 'ω_z']):
    w = X_exp[:, 8+i]
    
    # Count identical steps
    w_diff = np.diff(w)
    n_identical = np.sum(np.abs(w_diff) < 1e-10)
    pct_identical = 100 * n_identical / (N_steps - 1)
    
    std_w = np.std(w)
    mean_w = np.mean(w)
    max_w = np.max(w)
    min_w = np.min(w)
    
    consecutive_identical = []
    count = 0
    for diff in w_diff:
        if np.abs(diff) < 1e-10:
            count += 1
        else:
            if count > 0:
                consecutive_identical.append(count)
            count = 0
    
    if consecutive_identical:
        max_consecutive = max(consecutive_identical)
    else:
        max_consecutive = 0
    
    print(f"\n{w_name}:")
    print(f"  Mean: {mean_w:.6f}")
    print(f"  Std Dev: {std_w:.6e}")
    print(f"  Range: [{min_w:.6f}, {max_w:.6f}]")
    print(f"  Steps with IDENTICAL velocity: {n_identical}/{N_steps-1} ({pct_identical:.2f}%)")
    print(f"  Longest consecutive identical: {max_consecutive} steps")
    print(f"  Max velocity change per step: {np.max(np.abs(w_diff)):.6e}")
    print(f"  Mean velocity change per step: {np.mean(np.abs(w_diff)):.6e}")

# ============================================================
# 3. OVERALL MOVEMENT ACTIVITY
# ============================================================
print("\n" + "="*70)
print("OVERALL MOVEMENT ACTIVITY")
print("="*70)

# Total velocity magnitude
v_mag = np.sqrt(v_x**2 + v_y**2 + v_z**2)
w_mag = np.sqrt(X_exp[:, 8]**2 + X_exp[:, 9]**2 + X_exp[:, 10]**2)

print(f"\nLinear velocity magnitude:")
print(f"  Mean: {np.mean(v_mag):.6f} m/s")
print(f"  Max: {np.max(v_mag):.6f} m/s")
print(f"  Min: {np.min(v_mag):.6f} m/s")
print(f"  Time moving (|v| > 0.001): {100*np.sum(v_mag > 0.001)/N_steps:.1f}%")
print(f"  Time stationary (|v| < 0.001): {100*np.sum(v_mag < 0.001)/N_steps:.1f}%")

print(f"\nAngular velocity magnitude:")
print(f"  Mean: {np.mean(w_mag):.6f} rad/s")
print(f"  Max: {np.max(w_mag):.6f} rad/s")
print(f"  Min: {np.min(w_mag):.6f} rad/s")
print(f"  Time rotating (|ω| > 0.001): {100*np.sum(w_mag > 0.001)/N_steps:.1f}%")
print(f"  Time stationary (|ω| < 0.001): {100*np.sum(w_mag < 0.001)/N_steps:.1f}%")

# ============================================================
# 4. RECOMMENDATION
# ============================================================
print("\n" + "="*70)
print("ANALYSIS & RECOMMENDATION")
print("="*70)

n_const_v = np.sum(np.abs(np.diff(v_x)) < 1e-10) + np.sum(np.abs(np.diff(v_y)) < 1e-10) + np.sum(np.abs(np.diff(v_z)) < 1e-10)
n_const_w = np.sum(np.abs(np.diff(X_exp[:, 8])) < 1e-10) + np.sum(np.abs(np.diff(X_exp[:, 9])) < 1e-10) + np.sum(np.abs(np.diff(X_exp[:, 10])) < 1e-10)

total_const = n_const_v + n_const_w

if total_const > (N_steps - 1) * 0.5:
    print(f"\n⚠️  WARNING: MORE THAN 50% OF VELOCITY STEPS ARE CONSTANT!")
    print(f"   This means velocities barely change between time steps.")
    print(f"   → Numerical derivatives will be very small or zero")
    print(f"   → Consider: Use raw plant velocities instead of derived ones")
elif total_const > (N_steps - 1) * 0.2:
    print(f"\n⚠️  CAUTION: {100*total_const/(N_steps-1):.1f}% of velocity steps are constant")
    print(f"   Some periods have little motion. This affects derivatives.")
    print(f"   → Consider filtering or smoothing the data")
else:
    print(f"\n✓ OK: Motion is reasonably variable ({100*total_const/(N_steps-1):.1f}% constant)")
    print(f"   Numerical derivatives should work reasonably well")

print("\n" + "="*70 + "\n")
