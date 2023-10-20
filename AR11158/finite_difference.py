import numpy as np

# first order, second order differential operator

def Dx(f, h):
    Dx_f = np.zeros_like(f)
    Dx_f[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * h)
    Dx_f[0, :, :] = (-3 * f[0, :, :] + 4 * f[1, :, :] - f[2, :, :]) / (2 * h)
    Dx_f[-1, :, :] = (3 * f[-1, :, :] - 4 * f[-2, :, :] + f[-3, :, :]) / (2 * h)
    return Dx_f


def Dy(f, h):
    Dy_f = np.zeros_like(f)
    Dy_f[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * h)
    Dy_f[:, 0, :] = (-3 * f[:, 0, :] + 4 * f[:, 1, :] - f[:, 2, :]) / (2 * h)
    Dy_f[:, -1, :] = (3 * f[:, -1, :] - 4 * f[:, -2, :] + f[:, -3, :]) / (2 * h)
    return Dy_f


def Dz(f, h):
    Dz_f = np.zeros_like(f)
    Dz_f[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * h)
    Dz_f[:, :, 0] = (-3 * f[:, :, 0] + 4 * f[:, :, 1] - f[:, :, 2]) / (2 * h)
    Dz_f[:, :, -1] = (3 * f[:, :, -1] - 4 * f[:, :, -2] + f[:, :, -3]) / (2 * h)
    return Dz_f


def DDx(f, h):
    DDx_f = np.zeros_like(f)
    DDx_f[1:-1, :, :] = (f[2:, :, :] - 2 * f[1:-1, :, :] + f[:-2, :, :]) / (h**2)
    DDx_f[0, :, :] = (2 * f[0, :, :] - 5 * f[1, :, :] + 4 * f[2, :, :] - f[3, :, :]) / (
        h**2
    )
    DDx_f[-1, :, :] = (
        2 * f[-1, :, :] - 5 * f[-2, :, :] + 4 * f[-3, :, :] - f[-4, :, :]
    ) / (h**2)
    return DDx_f


def DDy(f, h):
    DDy_f = np.zeros_like(f)
    DDy_f[:, 1:-1, :] = (f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]) / (h**2)
    DDy_f[:, 0, :] = (2 * f[:, 0, :] - 5 * f[:, 1, :] + 4 * f[:, 2, :] - f[:, 3, :]) / (
        h**2
    )
    DDy_f[:, -1, :] = (
        2 * f[:, -1, :] - 5 * f[:, -2, :] + 4 * f[:, -3, :] - f[:, -4, :]
    ) / (h**2)
    return DDy_f


def DDz(f, h):
    DDz_f = np.zeros_like(f)
    DDz_f[:, :, 1:-1] = (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / (h**2)
    DDz_f[:, :, 0] = (2 * f[:, :, 0] - 5 * f[:, :, 1] + 4 * f[:, :, 2] - f[:, :, 3]) / (
        h**2
    )
    DDz_f[:, :, -1] = (
        2 * f[:, :, -1] - 5 * f[:, :, -2] + 4 * f[:, :, -3] - f[:, :, -4]
    ) / (h**2)
    return DDz_f

# differential operator

def gradient(f, dx, dy, dz):
    gradient_xcomp = Dx(f, dx)
    gradient_ycomp = Dy(f, dy)
    gradient_zcomp = Dz(f, dz)
    return gradient_xcomp, gradient_ycomp, gradient_zcomp


def curl(Fx, Fy, Fz, dx, dy, dz):
    curl_xcomp = Dy(Fz, dy) - Dz(Fy, dz)
    curl_ycomp = Dz(Fx, dz) - Dx(Fz, dx)
    curl_zcomp = Dx(Fy, dx) - Dy(Fx, dy)

    return curl_xcomp, curl_ycomp, curl_zcomp


def divergence(Fx, Fy, Fz, dx, dy, dz):
    return Dx(Fx, dx) + Dy(Fy, dy) + Dz(Fz, dz)


def laplacian(f, dx, dy, dz):
    return DDx(f, dx) + DDy(f, dy) + DDz(f, dz)