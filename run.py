"""
Comparative simulation of the Van der Pol system
with the Prescribed-Time + BLF-artanh controller.
Final version with 3-row tracking figure and zoomed control/disturbance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# --------------------------------------------------------
# Auxiliary functions
# --------------------------------------------------------
def reference(t, dr):
    w = 2 * np.pi * dr
    x1d = np.cos(w * t)
    dx1d = -w * np.sin(w * t)
    ddx1d = -w**2 * np.cos(w * t)
    return (x1d, dx1d, ddx1d)


def disturbance(t, df):
    w = 2 * np.pi * df
    return (
        3 * np.sin(w * t - 1.2)
        - 1.5 * np.cos(2.0 * w * t + 0.98)
        + 3.5 * np.cos(0.75 * w * t + 0.05)
        * np.exp(np.cos(5.75 * w * t + 0.43))
    )


def deltaZ(t, T, delta_z_max, delta_z_min, tc):
    if tc is None or t < tc:
        return delta_z_max - (delta_z_max - delta_z_min) * (t / T)
    else:
        return delta_z_min


def checkSwitchCondition(t, T, z, delta_z_min, tc):
    if t < T and abs(z) > delta_z_min:
        return tc
    else:
        if tc is None:
            tc = t
        return tc


# --------------------------------------------------------
# System and control functions
# --------------------------------------------------------
def f1(x): return x[1]
def f2(x): return 3.0 * (1 - x[0]**2) * x[1] - 2.0 * x[0]
def g(x): return 1.0
df1_dx1, df1_dx2, g1 = 0.0, 1.0, 1.0


def nu(t, T, tc):
    return T / (T - t) if tc is None else T / (T - tc)


def gainLz(t, z0, T, k_delta, delta_z_max, delta_z_min, tc):
    delta_val = deltaZ(t, T, delta_z_max, delta_z_min, tc)
    return (1.0 / T) * np.log(abs(z0) / delta_val) / np.log(1.0 / (1.0 - k_delta))


def gammaZ(t, z, z0, T, k_delta, delta_z_max, delta_z_min, tc):
    Lz = gainLz(t, z0, T, k_delta, delta_z_max, delta_z_min, tc)
    return Lz * nu(t, T, tc) * z


def gammaE(e, eps_e):
    return (1.0 + eps_e) * e / 2.0


def gammaS(s, e, delta_z, g1, a, eps_s, eps=1e-8):
    denom = (delta_z**2 - e**2) + eps
    gain_val = (1.0 / (a * denom)) + 0.5 * (g1**2) + eps_s
    return gain_val * s


def alfaW(alpha_w_gain, g1, eps=1e-8):
    return alpha_w_gain * (1.0 / (g1 + eps))


# --------------------------------------------------------
# Main simulation
# --------------------------------------------------------
def main():
    sampling_time = 1e-4
    df, dr = 7.0, 4.0
    dot_D_max = 2.0 * np.pi * df
    initial_x = np.array([5.0, -2.0], dtype=float)
    delta_z_min = 0.05
    t_delta_z = 0.5
    c, a = 2.0, 2.0

    configs = [
        {"name": "Adaptive δz, R=2.0", "delta_z_max": 2 * delta_z_min, "R": 2.0, "color": "tab:blue"},
        {"name": "Fixed δz, R=2.0", "delta_z_max": delta_z_min, "R": 2.0, "color": "tab:green"},
        {"name": "Adaptive δz, R=0.5", "delta_z_max": 2 * delta_z_min, "R": 0.5, "color": "tab:red"},
    ]

    x1d0, _, _ = reference(0.0, dr)
    z0 = initial_x[0] - x1d0
    T = t_delta_z * (1 + (delta_z_min / (abs(z0) - delta_z_min)))
    k_delta = t_delta_z / T
    total_time = 1.5 * T
    time_vec = np.arange(0, total_time, sampling_time)

    states = [initial_x.copy() for _ in configs]
    x1c_vals = [s[0] for s in states]
    phi_vals = [None for _ in configs]
    tc_vals = [None for _ in configs]
    data = [{"x1": [], "x2": [], "e": [], "s": [], "u": [], "d_hat": [], "delta_z": []} for _ in configs]
    d_series = [disturbance(t, df) for t in time_vec]

    for t_idx, t in enumerate(time_vec):
        x1d, dx1d, ddx1d = reference(t, dr)
        d_val = d_series[t_idx]
        for i, cfg in enumerate(configs):
            delta_z_max, R = cfg["delta_z_max"], cfg["R"]
            alfa = c / T
            b = (2.0 * alfa * R**2) / (dot_D_max**2)
            eps_e, eps_s, eps_w = alfa, alfa / a, alfa / b
            alpha_w_gain = ((a + b) / (2.0 * b)) + eps_w

            x, x1c, phi, tc = states[i], x1c_vals[i], phi_vals[i], tc_vals[i]
            z = x1c - x1d
            tc = checkSwitchCondition(t, T, z, delta_z_min, tc)
            delta_val = deltaZ(t, T, delta_z_max, delta_z_min, tc)
            gamma_z = gammaZ(t, z, z0, T, k_delta, delta_z_max, delta_z_min, tc)
            dot_x1c = dx1d - gamma_z
            x1c += dot_x1c * sampling_time
            e = x[0] - x1c
            dot_e = f1(x) - dot_x1c
            gamma_e = gammaE(e, eps_e)
            s = dot_e + gamma_e
            alpha_w = alfaW(alpha_w_gain, df1_dx2)
            if phi is None:
                phi = -alpha_w * s
            gamma_s = gammaS(s, e, delta_val, g1, a, eps_s)
            d_hat = alpha_w * s + phi
            phi += alpha_w * gamma_s * sampling_time
            v = (df1_dx1 * f1(x) - ddx1d + gamma_s) / (df1_dx2 + 1e-8)
            u = (-f2(x) - d_hat - v) / g(x)
            dx1 = f1(x)
            dx2 = f2(x) + g(x) * u + d_val
            x += sampling_time * np.array([dx1, dx2])

            data[i]["x1"].append(x[0])
            data[i]["x2"].append(x[1])
            data[i]["e"].append(e)
            data[i]["s"].append(s)
            data[i]["u"].append(u)
            data[i]["d_hat"].append(d_hat)
            data[i]["delta_z"].append(delta_val)

            states[i], x1c_vals[i], phi_vals[i], tc_vals[i] = x, x1c, phi, tc

    # --------------------------------------------------------
    # FIGURE 1 — Tracking (x1, x2, e, s)
    # --------------------------------------------------------
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 10))
    axs1[0].set_title("Tracking trajectories $x_1(t)$, $x_2(t)$ vs $x_{1d}(t)$")
    axs1[1].set_title("Tracking error $e(t)$")
    axs1[2].set_title("Sliding variable $s(t)$")

    x1d_series = [reference(t, dr)[0] for t in time_vec]
    axs1[0].plot(time_vec, x1d_series, 'k--', label="$x_{1d}(t)$")

    for cfg, d in zip(configs, data):
        color = cfg["color"]
        x1, x2 = np.array(d["x1"]), np.array(d["x2"])
        e, s, delta_z_t = np.array(d["e"]), np.array(d["s"]), np.array(d["delta_z"])
        R = cfg["R"]
        e_bound = delta_z_min * R / np.sqrt(1 + R**2)

        axs1[0].plot(time_vec, x1, color=color, label=f"$x_1(t)$ ({cfg['name']})")
        #axs1[0].plot(time_vec, x2, linestyle=":", color=color, alpha=0.8)

        axs1[1].plot(time_vec, e, color=color, label=cfg["name"])
        #axs1[1].plot(time_vec, delta_z_t, '--', color=color, alpha=0.3)
        #axs1[1].plot(time_vec, -delta_z_t, '--', color=color, alpha=0.3)
        #axs1[1].axhline(e_bound, linestyle='--', color=color, alpha=0.7)
        #axs1[1].axhline(-e_bound, linestyle='--', color=color, alpha=0.7)

        axs1[2].plot(time_vec, s, color=color, label=cfg["name"])
        #axs1[2].axhline(R, linestyle='--', color=color, alpha=0.5)
        #axs1[2].axhline(-R, linestyle='--', color=color, alpha=0.5)

    for ax in axs1:
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        ax.legend()

    fig1.tight_layout()
    fig1.savefig("tracking_dynamics.pdf", dpi=600, bbox_inches="tight")

    # --------------------------------------------------------
    # FIGURE 2 — Control & disturbance (with zoom)
    # --------------------------------------------------------
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8))
    axs2[0].set_title("Control input $u(t)$ (zoomed region [0.45–0.6] s)")
    axs2[1].set_title("Disturbance and estimation (zoomed region [0.45–0.6] s)")

    d_array = np.array(d_series)
    zoom_region = (0.45, 0.6)

    for cfg, d in zip(configs, data):
        color = cfg["color"]
        u = np.array(d["u"])
        d_hat = np.array(d["d_hat"])
        R = cfg["R"]

        axs2[0].plot(time_vec, u, color=color, label=cfg["name"])
        axs2[1].plot(time_vec, d_array, 'k', linewidth=1.0, label="$d(t)$" if cfg == configs[0] else "")
        axs2[1].plot(time_vec, d_hat, '-', color=color, label=f"$\\hat d(t)$ ({cfg['name']})")

    # Zooms
    for idx, ax in enumerate(axs2):
        axins = inset_axes(ax, width="30%", height="40%", loc='lower right')
        for cfg, d in zip(configs, data):
            color = cfg["color"]
            if idx == 0:
                axins.plot(time_vec, np.array(d["u"]), color=color)
            else:
                axins.plot(time_vec, np.array(d["d_hat"]), '-', color=color)
                axins.plot(time_vec, d_array, 'k', linewidth=0.8)
        axins.set_xlim(*zoom_region)
        axins.grid(True, alpha=0.3)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    for ax in axs2:
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        ax.legend()

    fig2.tight_layout()
    fig2.savefig("control_and_disturbance_zoomed.pdf", dpi=600, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------
# Run
# --------------------------------------------------------
if __name__ == "__main__":
    main()
