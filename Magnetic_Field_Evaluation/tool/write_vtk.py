import netCDF4
import numpy as np
import pyvista as pv


def create_mesh(nc_file):
    nc = netCDF4.Dataset(nc_file, "r")

    x = np.array(nc.variables["x"])
    y = np.array(nc.variables["y"])
    z = np.array(nc.variables["z"])
    x, y, z = np.meshgrid(x, y, z, indexing="ij")
    mesh = pv.StructuredGrid(x, y, z)
    mesh["Bx"] = np.array(nc.variables["Bx"]).reshape(-1, 1)
    mesh["By"] = np.array(nc.variables["By"]).reshape(-1, 1)
    mesh["Bz"] = np.array(nc.variables["Bz"]).reshape(-1, 1)
    mesh["Bx_pot"] = np.array(nc.variables["Bx_pot"]).reshape(-1, 1)
    mesh["By_pot"] = np.array(nc.variables["By_pot"]).reshape(-1, 1)
    mesh["Bz_pot"] = np.array(nc.variables["Bz_pot"]).reshape(-1, 1)

    return mesh
