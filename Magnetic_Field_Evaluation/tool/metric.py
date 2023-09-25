import os
import pickle

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import torch
from dateutil.parser import parse
from matplotlib.colors import LogNorm
from setproctitle import setproctitle
from tool.evaluate import curl, divergence, laplacian
from tool.nf2.evaluation.unpack import load_cube
from tool.nf2.potential.potential_field import get_potential_field

setproctitle("nf2")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def find_max_within(array_3d, h):
    if h == 0:
        array_h = array_3d
    elif h > 0:
        array_h = array_3d[h:-h, h:-h, h:-h]

    max_idx = np.unravel_index(np.argmax(array_h), array_h.shape)
    # print(max_idx, array_h[max_idx], np.max(array_h))
    max_idx = tuple(np.array(max_idx) + h)
    return max_idx, array_3d[max_idx]
    # print(max_idx, array_3d[max_idx], np.max(array_h))


def calculate_derivative(Bx, By, Bz, dx, dy, dz):
    B = np.stack([Bx, By, Bz])  # G
    norm_B = np.linalg.norm(B, axis=0)

    Jx, Jy, Jz = curl(Bx, By, Bz, dx, dy, dz)
    J = np.stack([Jx, Jy, Jz])  # G/Mm
    norm_J = np.linalg.norm(J, axis=0)

    JxB = np.cross(J, B, axisa=0, axisb=0, axisc=0)  # G^2/Mm
    norm_JxB = np.linalg.norm(JxB, axis=0)

    div_B = divergence(Bx, By, Bz, dx, dy, dz)  # G/Mm
    norm_div_B = np.abs(div_B)

    laplacian_Bx = laplacian(Bx, dx, dy, dz)
    laplacian_By = laplacian(By, dx, dy, dz)
    laplacian_Bz = laplacian(Bz, dx, dy, dz)
    laplacian_B = np.stack([laplacian_Bx, laplacian_By, laplacian_Bz])  # G/Mm^2
    norm_laplacian_B = np.linalg.norm(laplacian_B, axis=0)

    energy_density_B = (norm_B**2) / (8 * np.pi)  # erg/cm^3

    return (
        B,
        norm_B,
        J,
        norm_J,
        JxB,
        norm_JxB,
        div_B,
        norm_div_B,
        laplacian_B,
        norm_laplacian_B,
        energy_density_B,
    )


def calculate_metric(
    dx,
    dy,
    dz,
    Lx,
    Ly,
    Lz,
    B,
    norm_B,
    J,
    norm_J,
    JxB,
    norm_JxB,
    div_B,
    norm_div_B,
    laplacian_B,
    norm_laplacian_B,
    energy_density_B,
):
    dV = dx * dy * dz  # Mm^3
    V = Lx * Ly * Lz  # Mm^3
    dV_cm = (dx * 1e8) * (dy * 1e8) * (dz * 1e8)  # cm^3

    total_energy = energy_density_B.sum() * dV_cm  # erg

    eps = 1e-7
    loss_force_free_integrand = (norm_JxB**2) / (norm_B**2 + eps)
    loss_force_free = (loss_force_free_integrand.sum() * dV) / V
    loss_force_free_mean = loss_force_free_integrand.mean()

    loss_div_free_integrand = norm_div_B**2
    loss_div_free = (loss_div_free_integrand.sum() * dV) / V
    loss_div_free_mean = loss_div_free_integrand.mean()

    # current-weighted average of the sine of the angle
    # between the magnetic field and the electrical current density
    sigma_i = norm_JxB / (norm_J * norm_B + eps)
    theta_i = np.arcsin(sigma_i)
    theta_i_mean = np.nanmean(theta_i)
    theta_i_mean = np.rad2deg(theta_i_mean)  # deg

    sigma_J = (norm_J * sigma_i).sum() / norm_J.sum()
    theta_J = np.arcsin(sigma_J)
    theta_J = np.rad2deg(theta_J)  # deg

    # laplacian
    norm_laplacian_B_mean = norm_laplacian_B.mean()
    max_idx_0, norm_laplacian_B_max_0 = find_max_within(norm_laplacian_B, 0)
    max_idx_1, norm_laplacian_B_max_1 = find_max_within(norm_laplacian_B, 1)
    max_idx_2, norm_laplacian_B_max_2 = find_max_within(norm_laplacian_B, 2)
    max_idx_3, norm_laplacian_B_max_3 = find_max_within(norm_laplacian_B, 3)
    max_idx_4, norm_laplacian_B_max_4 = find_max_within(norm_laplacian_B, 4)
    max_idx_5, norm_laplacian_B_max_5 = find_max_within(norm_laplacian_B, 5)

    return (
        total_energy,
        loss_force_free,
        loss_force_free_mean,
        loss_div_free,
        loss_div_free_mean,
        sigma_J,
        theta_J,
        theta_i_mean,
        norm_laplacian_B_mean,
        max_idx_0,
        norm_laplacian_B_max_0,
        max_idx_1,
        norm_laplacian_B_max_1,
        max_idx_2,
        norm_laplacian_B_max_2,
        max_idx_3,
        norm_laplacian_B_max_3,
        max_idx_4,
        norm_laplacian_B_max_4,
        max_idx_5,
        norm_laplacian_B_max_5,
    )


def draw_projection(
    array3D,
    title,
    save_dir,
    dx,
    dy,
    dz,
    Lx,
    Ly,
    Lz,
    z_Mm,
    cm=False,
    log=True,
    cmap="viridis",
):
    z_pixels = int(np.ceil(z_Mm / dz))

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    ax = fig.subplot_mosaic(
        """
        .A
        BC
        """,
        height_ratios=[1, 2],
        width_ratios=[1, 4],
    )

    if cm is True:
        dx = dx * 1e8
        dy = dy * 1e8
        dz = dz * 1e8
        unit = "cm"
    else:
        dx = dx
        dy = dy
        dz = dz
        unit = "Mm"

    array3D_xy = array3D.sum(-1) * dz
    array3D_yz = array3D.sum(0) * dx
    array3D_xz = array3D.sum(1) * dy

    vmin = np.min(array3D_xy)
    vmax = np.max(array3D_xy)
    if int(vmin) == 0:
        vmin = vmax * 1e-6

    if log is False:
        vmin = 0
        im = ax["C"].imshow(
            array3D_xy.T,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect="auto",
            extent=(0, Lx, 0, Ly),
        )
        ax["C"].set_xlabel("X [Mm]", fontsize=15)

        ax["A"].imshow(
            array3D_xz[:, :z_pixels].T,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect="auto",
            extent=(0, Lx, 0, z_Mm),
        )
        ax["A"].set_ylabel("Z [Mm]", fontsize=15)

        ax["B"].imshow(
            np.rot90(array3D_yz[:, :z_pixels]).T,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect="auto",
            extent=(0, z_Mm, 0, Ly),
        )
        ax["B"].set_ylabel("Y [Mm]", fontsize=15)
    else:
        im = ax["C"].imshow(
            array3D_xy.T,
            origin="lower",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap,
            aspect="auto",
            extent=(0, Lx, 0, Ly),
        )
        ax["C"].set_xlabel("X [Mm]", fontsize=15)

        ax["A"].imshow(
            array3D_xz[:, :z_pixels].T,
            origin="lower",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap,
            aspect="auto",
            extent=(0, Lx, 0, z_Mm),
        )
        ax["A"].set_ylabel("Z [Mm]", fontsize=15)

        ax["B"].imshow(
            np.rot90(array3D_yz[:, :z_pixels]).T,
            origin="lower",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap,
            aspect="auto",
            extent=(0, z_Mm, 0, Ly),
        )
        ax["B"].set_ylabel("Y [Mm]", fontsize=15)

    fig.colorbar(
        im,
        ax=ax["C"],
        label=f"Quantity \n integrated along each axis [(unit of quantity) * {unit}]",
    )
    fig.suptitle(title, fontsize=30)

    plt.savefig(save_dir, dpi=160)
    plt.close()


def load_nf2_file(nf2_file):
    B = load_cube(nf2_file, progress=True)
    B_pot = get_potential_field(B[:, :, 0, 2], B.shape[2], batch_size=int(1e3))

    Bx = B[..., 0]  # G
    By = B[..., 1]  # G
    Bz = B[..., 2]  # G

    Bx_pot = B_pot[..., 0]  # G
    By_pot = B_pot[..., 1]  # G
    Bz_pot = B_pot[..., 2]  # G

    state = torch.load(nf2_file)
    Mm_per_pixel = state["Mm_per_pixel"]
    Nx, Ny, Nz = state["cube_shape"]
    Lx = (Nx - 1) * Mm_per_pixel  # Mm
    Ly = (Ny - 1) * Mm_per_pixel  # Mm
    Lz = (Nz - 1) * Mm_per_pixel  # Mm

    x = np.linspace(0, Lx, Nx)  # Mm
    y = np.linspace(0, Ly, Ny)  # Mm
    z = np.linspace(0, Lz, Nz)  # Mm

    return x, y, z, Bx, By, Bz, Bx_pot, By_pot, Bz_pot


def load_nc_file(nc_file):
    nc = netCDF4.Dataset(nc_file, "r")

    x = np.array(nc.variables["x"])  # Mm
    y = np.array(nc.variables["y"])  # Mm
    z = np.array(nc.variables["z"])  # Mm

    Bx = np.array(nc.variables["Bx"]).transpose(2, 1, 0)  # G
    By = np.array(nc.variables["By"]).transpose(2, 1, 0)  # G
    Bz = np.array(nc.variables["Bz"]).transpose(2, 1, 0)  # G

    Bx_pot = np.array(nc.variables["Bx_pot"]).transpose(2, 1, 0)  # G
    By_pot = np.array(nc.variables["By_pot"]).transpose(2, 1, 0)  # G
    Bz_pot = np.array(nc.variables["Bz_pot"]).transpose(2, 1, 0)  # G

    return x, y, z, Bx, By, Bz, Bx_pot, By_pot, Bz_pot


def parse_nc_file(nc_file):
    obs_date = os.path.basename(nc_file).split(".")[0][6:].replace("_", "T")

    return obs_date  # YYYYMMDD_HHMMSS (string)


def parse_nf2_file(nf2_file):
    obs_date = os.path.basename(nf2_file).split(".")[0][:-4].replace("_", "T")

    return obs_date  # YYYYMMDD_HHMMSS (string)


def evaluate_single(file, *args):
    # ------------------------------------------------------------
    ext = os.path.basename(file).split(".")[1]

    if ext == "nc":
        obs_date = parse_nc_file(file)

    elif ext == "nf2":
        obs_date = parse_nf2_file(file)

    result_dir = os.path.join(
        args[0].result_dir, obs_date
    )  # result_dir / YYYYMMDD_HHMMSS
    os.makedirs(result_dir, exist_ok=True)

    obs_date = parse(obs_date)  # YYYYMMDD_HHMMSS (datetime)

    result_pickle = os.path.join(result_dir, "result.pickle")

    if os.path.exists(result_pickle):
        with open(result_pickle, "rb") as f:
            result = pickle.load(f)
        return result

    # ------------------------------------------------------------
    if ext == "nc":
        x, y, z, Bx, By, Bz, Bx_pot, By_pot, Bz_pot = load_nc_file(file)

    elif ext == "nf2":
        x, y, z, Bx, By, Bz, Bx_pot, By_pot, Bz_pot = load_nf2_file(file)

    # ------------------------------------------------------------
    dx = x[1] - x[0]  # Mm
    dy = y[1] - y[0]  # Mm
    dz = z[1] - z[0]  # Mm

    Lx = x[-1] - x[0]  # Mm
    Ly = y[-1] - y[0]  # Mm
    Lz = z[-1] - z[0]  # Mm

    # ------------------------------------------------------------

    (
        B,
        norm_B,
        J,
        norm_J,
        JxB,
        norm_JxB,
        div_B,
        norm_div_B,
        laplacian_B,
        norm_laplacian_B,
        energy_density_B,
    ) = calculate_derivative(Bx, By, Bz, dx, dy, dz)

    (
        B_pot,
        norm_B_pot,
        J_pot,
        norm_J_pot,
        JxB_pot,
        norm_JxB_pot,
        div_B_pot,
        norm_div_B_pot,
        laplacian_B_pot,
        norm_laplacian_B_pot,
        energy_density_B_pot,
    ) = calculate_derivative(Bx_pot, By_pot, Bz_pot, dx, dy, dz)

    # ------------------------------------------------------------
    dV_cm = (dx * 1e8) * (dy * 1e8) * (dz * 1e8)  # cm^3

    free_energy_density = energy_density_B - energy_density_B_pot
    total_free_energy = free_energy_density.sum() * dV_cm  # erg

    (
        total_energy,
        loss_force_free,
        loss_force_free_mean,
        loss_div_free,
        loss_div_free_mean,
        sigma_J,
        theta_J,
        theta_i_mean,
        norm_laplacian_B_mean,
        max_idx_0,
        norm_laplacian_B_max_0,
        max_idx_1,
        norm_laplacian_B_max_1,
        max_idx_2,
        norm_laplacian_B_max_2,
        max_idx_3,
        norm_laplacian_B_max_3,
        max_idx_4,
        norm_laplacian_B_max_4,
        max_idx_5,
        norm_laplacian_B_max_5,
    ) = calculate_metric(
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        B,
        norm_B,
        J,
        norm_J,
        JxB,
        norm_JxB,
        div_B,
        norm_div_B,
        laplacian_B,
        norm_laplacian_B,
        energy_density_B,
    )

    (
        total_energy_pot,
        loss_force_free_pot,
        loss_force_free_mean_pot,
        loss_div_free_pot,
        loss_div_free_mean_pot,
        sigma_J_pot,
        theta_J_pot,
        theta_i_mean_pot,
        norm_laplacian_B_mean_pot,
        max_idx_0_pot,
        norm_laplacian_B_max_0_pot,
        max_idx_1_pot,
        norm_laplacian_B_max_1_pot,
        max_idx_2_pot,
        norm_laplacian_B_max_2_pot,
        max_idx_3_pot,
        norm_laplacian_B_max_3_pot,
        max_idx_4_pot,
        norm_laplacian_B_max_4_pot,
        max_idx_5_pot,
        norm_laplacian_B_max_5_pot,
    ) = calculate_metric(
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        B_pot,
        norm_B_pot,
        J_pot,
        norm_J_pot,
        JxB_pot,
        norm_JxB_pot,
        div_B_pot,
        norm_div_B_pot,
        laplacian_B_pot,
        norm_laplacian_B_pot,
        energy_density_B_pot,
    )

    result = {
        "date": obs_date,
        "total_free_energy": total_free_energy,
        # B
        "total_energy": total_energy,
        "loss_force_free": loss_force_free,
        "loss_force_free_mean": loss_force_free_mean,
        "loss_div_free": loss_div_free,
        "loss_div_free_mean": loss_div_free_mean,
        "sigma_J": sigma_J,
        "theta_J": theta_J,
        "theta_i_mean": theta_i_mean,
        "norm_laplacian_B_mean": norm_laplacian_B_mean,
        "max_idx_0": max_idx_0,
        "norm_laplacian_B_max_0": norm_laplacian_B_max_0,
        "max_idx_1": max_idx_1,
        "norm_laplacian_B_max_1": norm_laplacian_B_max_1,
        "max_idx_2": max_idx_2,
        "norm_laplacian_B_max_2": norm_laplacian_B_max_2,
        "max_idx_3": max_idx_3,
        "norm_laplacian_B_max_3": norm_laplacian_B_max_3,
        "max_idx_4": max_idx_4,
        "norm_laplacian_B_max_4": norm_laplacian_B_max_4,
        "max_idx_5": max_idx_5,
        "norm_laplacian_B_max_5": norm_laplacian_B_max_5,
        # B_pot
        "total_energy_pot": total_energy_pot,
        "loss_force_free_pot": loss_force_free_pot,
        "loss_force_free_mean_pot": loss_force_free_mean_pot,
        "loss_div_free_pot": loss_div_free_pot,
        "loss_div_free_mean_pot": loss_div_free_mean_pot,
        "sigma_J_pot": sigma_J_pot,
        "theta_J_pot": theta_J_pot,
        "theta_i_mean_pot": theta_i_mean_pot,
        "norm_laplacian_B_mean_pot": norm_laplacian_B_mean_pot,
        "max_idx_0_pot": max_idx_0_pot,
        "norm_laplacian_B_max_0_pot": norm_laplacian_B_max_0_pot,
        "max_idx_1_pot": max_idx_1_pot,
        "norm_laplacian_B_max_1_pot": norm_laplacian_B_max_1_pot,
        "max_idx_2_pot": max_idx_2_pot,
        "norm_laplacian_B_max_2_pot": norm_laplacian_B_max_2_pot,
        "max_idx_3_pot": max_idx_3_pot,
        "norm_laplacian_B_max_3_pot": norm_laplacian_B_max_3_pot,
        "max_idx_4_pot": max_idx_4_pot,
        "norm_laplacian_B_max_4_pot": norm_laplacian_B_max_4_pot,
        "max_idx_5_pot": max_idx_5_pot,
        "norm_laplacian_B_max_5_pot": norm_laplacian_B_max_5_pot,
    }

    # ------------------------------------------------------------
    z_Mm = args[0].z_Mm

    draw_projection(
        energy_density_B,
        "Energy density (erg/cm^3)",
        os.path.join(result_dir, "energy.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
        cm=True,
        log=False,
        cmap="jet",
    )
    draw_projection(
        energy_density_B_pot,
        "Energy density pot (erg/cm^3)",
        os.path.join(result_dir, "energy_pot.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
        cm=True,
        log=False,
        cmap="jet",
    )
    draw_projection(
        free_energy_density,
        "Free energy density (erg/cm^3)",
        os.path.join(result_dir, "energy_free.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
        cm=True,
        log=False,
        cmap="jet",
    )

    draw_projection(
        norm_J,
        r"$|\nabla \times \mathbf{B}|$" + " (G/Mm)",
        os.path.join(result_dir, "J.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
    )
    draw_projection(
        norm_laplacian_B,
        r"$|\nabla^2 \mathbf{B}|$" + " (G/Mm^2)",
        os.path.join(result_dir, "laplacian_B.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
    )
    draw_projection(
        norm_JxB,
        r"$|\mathbf{J} \times \mathbf{B}|$" + " (G^2/Mm)",
        os.path.join(result_dir, "JxB.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
    )
    draw_projection(
        norm_div_B,
        r"$|\nabla \cdot \mathbf{B}|$" + " (G/Mm)",
        os.path.join(result_dir, "div_B.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
    )

    draw_projection(
        norm_J_pot,
        r"$|\nabla \times \mathbf{B}|$" + " pot (G/Mm)",
        os.path.join(result_dir, "J_pot.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
    )
    draw_projection(
        norm_laplacian_B_pot,
        r"$|\nabla^2 \mathbf{B}|$" + " pot (G/Mm^2)",
        os.path.join(result_dir, "laplacian_B_pot.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
    )
    draw_projection(
        norm_JxB_pot,
        r"$|\mathbf{J} \times \mathbf{B}|$" + " pot (G^2/Mm)",
        os.path.join(result_dir, "JxB_pot.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
    )
    draw_projection(
        norm_div_B_pot,
        r"$|\nabla \cdot \mathbf{B}|$" + " pot (G/Mm)",
        os.path.join(result_dir, "div_B_pot.png"),
        dx,
        dy,
        dz,
        Lx,
        Ly,
        Lz,
        z_Mm,
    )

    with open(result_pickle, "wb") as f:
        pickle.dump(result, f)

    return result
