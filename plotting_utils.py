"""
PLOTTING UTILITIES FOR MPC2
============================
Extracted plotting functions from dual_quaternion module
Provides local implementations to avoid external dependencies
"""

import matplotlib.pyplot as plt
import os
import numpy as np


def fancy_plots_4():
    """
    Creates a figure with 4 subplots arranged vertically.
    Optimized for LaTeX document proportions.
    
    Returns:
        fig: matplotlib figure object
        ax1, ax2, ax3, ax4: matplotlib axes objects
    """
    # Define parameters fancy plot
    pts_per_inch = 72.27
    text_width_in_pts = 300.0
    text_width_in_inches = text_width_in_pts / pts_per_inch
    
    inverse_latex_scale = 2
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    fig_size = (1.0 * csize, 0.7 * csize)
    
    text_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            'ps.usedistiller': 'xpdf',
            'text.usetex': False,
            'figure.figsize': fig_size,
            'text.latex.preamble': [r'\usepackage{amsmath}'],
            }
    plt.rc(params)
    plt.clf()
    
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    return fig, ax1, ax2, ax3, ax4


def fancy_plots_3():
    """
    Creates a figure with 3 subplots arranged vertically.
    Optimized for LaTeX document proportions.
    
    Returns:
        fig: matplotlib figure object
        ax1, ax2, ax3: matplotlib axes objects
    """
    pts_per_inch = 72.27
    text_width_in_pts = 300.0
    text_width_in_inches = text_width_in_pts / pts_per_inch
    
    inverse_latex_scale = 2
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    fig_size = (1.0 * csize, 0.7 * csize)
    
    text_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            'ps.usedistiller': 'xpdf',
            'text.usetex': False,
            'figure.figsize': fig_size,
            'text.latex.preamble': [r'\usepackage{amsmath}'],
            }
    plt.rc(params)
    plt.clf()
    
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    return fig, ax1, ax2, ax3


def fancy_plots_1():
    """
    Creates a figure with 1 subplot.
    Optimized for LaTeX document proportions.
    
    Returns:
        fig: matplotlib figure object
        ax1: matplotlib axes object
    """
    pts_per_inch = 72.27
    text_width_in_pts = 300.0
    text_width_in_inches = text_width_in_pts / pts_per_inch
    
    inverse_latex_scale = 2
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    fig_size = (1.0 * csize, 0.7 * csize)
    
    text_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            'ps.usedistiller': 'xpdf',
            'text.usetex': False,
            'figure.figsize': fig_size,
            'text.latex.preamble': [r'\usepackage{amsmath}'],
            }
    plt.rc(params)
    plt.clf()
    
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(111)
    
    return fig, ax1


def plot_states_quaternion(fig11, ax11, ax21, ax31, ax41, x, xd, t, name, path):
    """
    Plot quaternion states (actual vs desired) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11, ax21, ax31, ax41: matplotlib axes objects
        x: array of actual quaternion states (4 × N)
        xd: array of desired quaternion states (4 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x.shape[1]]
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax31.set_xlim((t[0], t[-1]))
    ax41.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    ax21.set_xticklabels([])
    ax31.set_xticklabels([])

    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                color='#C43C29', lw=1.0, ls="-")

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], xd[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="--")

    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")

    state_2_e_d, = ax21.plot(t[0:t.shape[0]], xd[1, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="--")

    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    state_3_e_d, = ax31.plot(t[0:t.shape[0]], xd[2, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="--")

    state_4_e, = ax41.plot(t[0:t.shape[0]], x[3, 0:t.shape[0]],
                    color='#36323E', lw=1.0, ls="-")

    state_4_e_d, = ax41.plot(t[0:t.shape[0]], xd[3, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="--")

    ax11.set_ylabel(r"$[]$", rotation='vertical')
    ax11.legend([state_1_e, state_1_e_d],
            [ r'$q_w$', r'$q_{wd}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ax21.set_ylabel(r"$[]$", rotation='vertical')
    ax21.legend([state_2_e, state_2_e_d],
            [r'$q_1$', r'$q_{1d}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])

    ax31.set_ylabel(r"$[]$", rotation='vertical')
    ax31.legend([state_3_e, state_3_e_d],
            [r'$q_2$', r'$q_{2d}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.set_xticklabels([])

    ax41.set_ylabel(r"$[]$", rotation='vertical')
    ax41.legend([state_4_e, state_4_e_d],
            [r'$q_3$', r'$q_{3d}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax41.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax41.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_states_position(fig11, ax11, ax21, ax31, x, xd, t, name, path):
    """
    Plot position states (actual vs desired) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11, ax21, ax31: matplotlib axes objects
        x: array of actual position states (3 × N) [x, y, z]
        xd: array of desired position states (3 × N) [x, y, z]
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x.shape[1]]
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax31.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    ax21.set_xticklabels([])
    
    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                color='#C43C29', lw=1.0, ls="-")

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], xd[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="--")

    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")

    state_2_e_d, = ax21.plot(t[0:t.shape[0]], xd[1, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="--")

    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    state_3_e_d, = ax31.plot(t[0:t.shape[0]], xd[2, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="--")

    ax11.set_ylabel(r"$[m]$", rotation='vertical')
    ax11.legend([state_1_e, state_1_e_d],
            [ r'$x$', r'$x_d$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ax21.set_ylabel(r"$[m]$", rotation='vertical')
    ax21.legend([state_2_e, state_2_e_d],
            [r'$y$', r'$y_d$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])

    ax31.set_ylabel(r"$[m]$", rotation='vertical')
    ax31.legend([state_3_e, state_3_e_d],
            [r'$z$', r'$z_d$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_control_actions(fig11, ax11, ax21, ax31, ax41, F, M, t, name, path):
    """
    Plot control actions (thrust and torques) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11, ax21, ax31, ax41: matplotlib axes objects
        F: array of thrust values (1 × N)
        M: array of torques (3 × N) [τx, τy, τz]
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:M.shape[1]]
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax31.set_xlim((t[0], t[-1]))
    ax41.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    ax21.set_xticklabels([])
    ax31.set_xticklabels([])

    state_1_e, = ax11.plot(t[0:t.shape[0]], F[0, 0:t.shape[0]],
                color='#C43C29', lw=1.0, ls="-")

    state_2_e, = ax21.plot(t[0:t.shape[0]], M[0, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")

    state_3_e, = ax31.plot(t[0:t.shape[0]], M[1, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    state_4_e, = ax41.plot(t[0:t.shape[0]], M[2, 0:t.shape[0]],
                    color='#36323E', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[N]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$f_z$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ax21.set_ylabel(r"$[N.m]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'$\tau_x$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])

    ax31.set_ylabel(r"$[N.m]$", rotation='vertical')
    ax31.legend([state_3_e],
            [r'$\tau_y$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.set_xticklabels([])

    ax41.set_ylabel(r"$[N.m]$", rotation='vertical')
    ax41.legend([state_4_e],
            [r'$\tau_z$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax41.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax41.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_cost_total(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot total cost function over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of cost values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Cost$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\mathbf{t}^{i}_{d, k} - \mathrm{trans}(\mathbf{x}_k)||^{2} + ||\mathbf{q}_{d, k} - \mathrm{quat}(\mathbf{x}_k)||^{2}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_angular_velocities(fig11, ax11, ax21, ax31, x, t, name, path):
    """
    Plot angular velocities (ωx, ωy, ωz) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11, ax21, ax31: matplotlib axes objects
        x: array of angular velocities (3 × N) [ωx, ωy, ωz]
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x.shape[1]]
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax31.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    ax21.set_xticklabels([])
    
    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                color='#C43C29', lw=1.0, ls="-")

    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")

    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$w_x$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ax21.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'$w_y$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])

    ax31.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax31.legend([state_3_e],
            [r'$w_z$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_linear_velocities(fig11, ax11, ax21, ax31, x, t, name, path):
    """
    Plot linear velocities (vx, vy, vz) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11, ax21, ax31: matplotlib axes objects
        x: array of linear velocities (3 × N) [vx, vy, vz]
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x.shape[1]]
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax31.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    ax21.set_xticklabels([])
    
    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                color='#C43C29', lw=1.0, ls="-")

    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")

    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$v_x$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ax21.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'$v_y$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])

    ax31.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax31.legend([state_3_e],
            [r'$v_z$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_cost_orientation(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot orientation cost (quaternion error) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of cost values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Cost$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\mathbf{q}_{d, k} - \mathrm{quat}(\mathbf{x}_k)||^{2}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_cost_translation(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot translation cost (position error) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of cost values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Cost$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\mathbf{t}^{i}_{d, k} - \mathrm{trans}(\mathbf{x}_k)||^{2}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_cost_control(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot control effort cost over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of cost values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Cost$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\mathbf{u}_k||^{2}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_norm_quat(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot quaternion norm (error magnitude) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of norm values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure (optional)
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[rad]$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||q_e||$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    if path:
        pdf_file_path = os.path.join(path, name + ".pdf")
        png_file_path = os.path.join(path, name + ".png")
    else:
        pdf_file_path = name + ".pdf"
        png_file_path = name + ".png"
    
    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_norm_real(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot real part error (rotational error) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of error values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure (optional)
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Error~(real~part)$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\ln(\mathbf{q}^{*}_d \circ \mathbf{q})||$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    if path:
        pdf_file_path = os.path.join(path, name + ".pdf")
        png_file_path = os.path.join(path, name + ".png")
    else:
        pdf_file_path = name + ".pdf"
        png_file_path = name + ".png"
    
    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_norm_dual(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot dual part error (translational error) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of error values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure (optional)
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Error~(dual~part)$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\ln(\mathbf{p}^b - \mathrm{Ad}_{\mathbf{q}_e} \mathbf{p}^{b}_d)||$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    if path:
        pdf_file_path = os.path.join(path, name + ".pdf")
        png_file_path = os.path.join(path, name + ".png")
    else:
        pdf_file_path = name + ".pdf"
        png_file_path = name + ".png"
    
    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_lyapunov_dot(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot time derivative of Lyapunov function.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of Lyapunov derivative values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure (optional)
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$\dot{V}$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$\dot{V}(t)$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    if path:
        pdf_file_path = os.path.join(path, name + ".pdf")
        png_file_path = os.path.join(path, name + ".png")
    else:
        pdf_file_path = name + ".pdf"
        png_file_path = name + ".png"
    
    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_lyapunov(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot Lyapunov function over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of Lyapunov function values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure (optional)
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$V$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$V(t)$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    if path:
        pdf_file_path = os.path.join(path, name + ".pdf")
        png_file_path = os.path.join(path, name + ".png")
    else:
        pdf_file_path = name + ".pdf"
        png_file_path = name + ".png"
    
    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_time(fig11, ax11, x_sample, x_sample_real, t, name, path):
    """
    Plot computational time (actual vs sample time) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample: array of actual computation times (1 × N)
        x_sample_real: array of sample times (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x_sample.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    state_1_e, = ax11.plot(t[0:t.shape[0]], x_sample[0, 0:t.shape[0]],
                color='#C43C29', lw=1.0, ls="--")

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Time~[s]$", rotation='vertical')
    ax11.legend([state_1_e, state_1_e_d],
            [ r'$dt_{actual}$', r'$dt_{sample}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_angular_velocities(fig11, ax11, ax21, ax31, x, t, name, path):
    """
    Plot angular velocities (ωx, ωy, ωz) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11, ax21, ax31: matplotlib axes objects
        x: array of angular velocity values (3 × N) [ωx, ωy, ωz]
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x.shape[1]]
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax31.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    ax21.set_xticklabels([])
    
    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                color='#C43C29', lw=1.0, ls="-")

    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")

    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$w_x$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ax21.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'$w_y$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])

    ax31.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax31.legend([state_3_e],
            [r'$w_z$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_linear_velocities(fig11, ax11, ax21, ax31, x, t, name, path):
    """
    Plot linear velocities (vx, vy, vz) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11, ax21, ax31: matplotlib axes objects
        x: array of linear velocity values (3 × N) [vx, vy, vz]
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x.shape[1]]
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax31.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    ax21.set_xticklabels([])
    
    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                color='#C43C29', lw=1.0, ls="-")

    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")

    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$v_x$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ax21.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'$v_y$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])

    ax31.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax31.legend([state_3_e],
            [r'$v_z$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_cost_orientation(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot orientation cost (quaternion error) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of cost values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Cost$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\mathbf{q}_{d, k} - \mathrm{quat}(\mathbf{x}_k)||^{2}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_cost_translation(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot translation cost (position error) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of cost values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Cost$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\mathbf{t}^{i}_{d, k} - \mathrm{trans}(\mathbf{x}_k)||^{2}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_cost_control(fig11, ax11, x_sample_real, t, name, path):
    """
    Plot control effort cost over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of cost values (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Cost$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\mathbf{u}_k||^{2}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_norm_quat(fig11, ax11, x_sample_real, t, name):
    """
    Plot quaternion norm (constraint validation) over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of norm values (1 × N)
        t: time vector
        name: filename for saving (without extension)
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[rad]$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||q_e||$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    return None


def plot_norm_real(fig11, ax11, x_sample_real, t, name):
    """
    Plot norm of real part of logarithmic error over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of error values (1 × N)
        t: time vector
        name: filename for saving (without extension)
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Error~value~(real~part)$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\ln(\mathbf{q}^{*}_d \circ \mathbf{q})||$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    return None


def plot_norm_dual(fig11, ax11, x_sample_real, t, name):
    """
    Plot norm of dual part of logarithmic error over time.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of error values (1 × N)
        t: time vector
        name: filename for saving (without extension)
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Error~value~(dual~part)$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$||\ln(\mathbf{p}^b - \mathrm{Ad}_{\mathbf{q}_e} \mathbf{p}^{b}_d)||$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    return None


def plot_lyapunov_dot(fig11, ax11, x_sample_real, t, name):
    """
    Plot time derivative of Lyapunov function over time.
    Validates stability of control (should be negative).
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of Lyapunov derivative values (1 × N)
        t: time vector
        name: filename for saving (without extension)
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Time~derivative~V$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$\dot{V}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    return None


def plot_lyapunov(fig11, ax11, x_sample_real, t, name):
    """
    Plot Lyapunov function value over time.
    Represents total energy/stability measure of the system.
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample_real: array of Lyapunov function values (1 × N)
        t: time vector
        name: filename for saving (without extension)
    """
    t = t[0:x_sample_real.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$Lyapunov~Function$", rotation='vertical')
    ax11.legend([state_1_e_d],
            [r'$V$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    return None


def plot_time(fig11, ax11, x_sample, x_sample_real, t, name, path):
    """
    Plot time vector comparison (actual vs target sample time).
    
    Args:
        fig11: matplotlib figure object
        ax11: matplotlib axes object
        x_sample: array of target sample times (1 × N)
        x_sample_real: array of actual times (1 × N)
        t: time vector
        name: title/name for the plot
        path: directory path to save the figure
    """
    t = t[0:x_sample.shape[1]]
    ax11.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    
    state_1_e, = ax11.plot(t[0:t.shape[0]], x_sample[0, 0:t.shape[0]],
                color='#C43C29', lw=1.0, ls="--")

    state_1_e_d, = ax11.plot(t[0:t.shape[0]], x_sample_real[0, 0:t.shape[0]],
                color='#1D2121', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[s]$", rotation='vertical')
    ax11.legend([state_1_e, state_1_e_d],
            [ r'$dt$', r'$sample~time$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$Time [s]$", labelpad=5)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")

    fig11.savefig(pdf_file_path)
    fig11.savefig(png_file_path)
    return None


def plot_curvature_vs_velocity(t_active, s_history, u_s_history, v_tang_history,
                                position_by_arc_length, name, path):
    """
    Plot curvature radius vs virtual point speed and real tangential speed.
    Shows how the MPCC slows down in high-curvature (sharp curve) regions.

    Args:
        t_active:               time vector of active experiment (N,)
        s_history:              arc-length state at each step (N,)
        u_s_history:            virtual point speed u_s* (N,)
        v_tang_history:         real tangential speed of the drone (N,)
        position_by_arc_length: callable(s) -> [x, y, z]
        name:                   filename (without extension)
        path:                   directory to save
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    N = len(t_active)

    # ── Compute curvature radius at unique arc-length samples (avoid repetition) ──
    # κ = |r' × r''| / |r'|³  →  R = 1/κ
    ds = 0.1   # step for numerical derivatives [m]
    R_curve = np.zeros(N)

    for i in range(N):
        s = s_history[i]
        s_lo  = max(s - ds, 0.0)
        s_lo2 = max(s - 2*ds, 0.0)
        p_fwd  = np.array(position_by_arc_length(s + ds))
        p_bwd  = np.array(position_by_arc_length(s_lo))
        p_fwd2 = np.array(position_by_arc_length(s + 2*ds))
        p_bwd2 = np.array(position_by_arc_length(s_lo2))
        p_mid  = np.array(position_by_arc_length(s))

        r_prime  = (p_fwd - p_bwd)   / (2 * ds)
        r_pprime = (p_fwd2 - 2*p_mid + p_bwd2) / (4 * ds**2)

        cross = np.cross(r_prime, r_pprime)
        kappa = np.linalg.norm(cross) / (np.linalg.norm(r_prime)**3 + 1e-9)
        R_curve[i] = 1.0 / (kappa + 1e-6)

    # ── Clamp outliers: cap at 3× the median (inflection points → R→∞) ──────
    R_median = np.median(R_curve)
    R_cap    = 3.0 * R_median
    R_curve  = np.clip(R_curve, 0.0, R_cap)

    # ── Smooth with a wider moving average ───────────────────────────────────
    window = 51
    kernel = np.ones(window) / window
    R_smooth = np.convolve(R_curve, kernel, mode='same')

    # ── Figure with two vertically stacked axes (shared x) ───────────────────
    fig = plt.figure(figsize=(13, 7))
    gs  = gridspec.GridSpec(2, 1, hspace=0.08)

    # Top axis: curvature radius
    ax_top = fig.add_subplot(gs[0])
    ax_top.fill_between(t_active, R_smooth, alpha=0.18, color='#7B4FA6')
    curve_line, = ax_top.plot(t_active, R_smooth, color='#7B4FA6', lw=1.5, ls='-')
    ax_top.set_ylabel(r'$R_{curv}$ [m]', rotation='vertical', labelpad=8)
    ax_top.set_xlim((t_active[0], t_active[-1]))
    ax_top.set_xticklabels([])
    ax_top.legend([curve_line], [r'$R_{curv}(s)$ (radio de curvatura)'],
                  loc='upper right', frameon=True, fancybox=True, shadow=False,
                  borderpad=0.5, labelspacing=0.5, handlelength=3)
    ax_top.grid(color='#949494', linestyle='-.', linewidth=0.5)

    # Bottom axis: velocities
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    us_line, = ax_bot.plot(t_active, u_s_history,    color='#1D2121', lw=1.5, ls='-')
    vt_line, = ax_bot.plot(t_active, v_tang_history,  color='#C43C29', lw=1.5, ls='--')
    ax_bot.set_ylabel(r'Velocity [m/s]', rotation='vertical', labelpad=8)
    ax_bot.set_xlabel(r'$Time~[s]$', labelpad=5)
    ax_bot.set_xlim((t_active[0], t_active[-1]))
    ax_bot.legend([us_line, vt_line],
                  [r'$u_s^*$ (punto virtual)', r'$v_{tang}$ (componente tangente)'],
                  loc='upper right', frameon=True, fancybox=True, shadow=False,
                  borderpad=0.5, labelspacing=0.5, handlelength=3)
    ax_bot.grid(color='#949494', linestyle='-.', linewidth=0.5)

    fig.suptitle('Radio de Curvatura vs Velocidad MPCC', fontsize=12)

    pdf_file_path = os.path.join(path, name + ".pdf")
    png_file_path = os.path.join(path, name + ".png")
    fig.savefig(pdf_file_path, bbox_inches='tight')
    fig.savefig(png_file_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return None
