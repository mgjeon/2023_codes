import os 
import glob
import argparse

import netCDF4
import numpy as np
import matplotlib.pyplot as plt

from tool.evaluate import *

parser = argparse.ArgumentParser()

parser.add_argument('--fname', type=str, help='path to nc file')

args = parser.parse_args()

if os.path.isdir(args.fname):
    files = sorted(glob.glob(os.path.join(args.fname, '*.nc')))
elif os.path.isfile(args.fname):
    files = [args.fname]

for file in files:
    nc = netCDF4.Dataset(file, 'r')

    x = np.array(nc.variables['x'])
    y = np.array(nc.variables['y'])
    z = np.array(nc.variables['z'])

    Bx = np.array(nc.variables['Bx']).transpose(2, 1, 0)
    By = np.array(nc.variables['By']).transpose(2, 1, 0)
    Bz = np.array(nc.variables['Bz']).transpose(2, 1, 0)

    Bx_pot = np.array(nc.variables['Bx_pot']).transpose(2, 1, 0)
    By_pot = np.array(nc.variables['By_pot']).transpose(2, 1, 0)
    Bz_pot = np.array(nc.variables['Bz_pot']).transpose(2, 1, 0)

    # dx, dy, dz
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    # divergence
    divergence_B = Dx(Bx, dx) + Dy(By, dy) + Dz(Bz, dz)

    # curl
    curl_B_xcomp = Dy(Bz, dy) - Dz(By, dz)
    curl_B_ycomp = Dz(Bx, dz) - Dx(Bz, dx)
    curl_B_zcomp = Dx(By, dx) - Dy(Bx, dy)

    curl_B = np.stack([curl_B_xcomp, curl_B_ycomp, curl_B_zcomp])
    curl_B_magnitude = np.sqrt((curl_B**2).sum(0))
    
    # plt.imshow(curl_B_magnitude.sum(-1).T, origin='lower')
    # plt.savefig('fig.png', dpi=160)

    # laplacian
    laplacian_B_xcomp = DDx(Bx, dx) + DDy(Bx, dy) + DDz(Bx, dz)
    laplacian_B_ycomp = DDx(By, dx) + DDy(By, dy) + DDz(By, dz)
    laplacian_B_zcomp = DDx(Bz, dx) + DDy(Bz, dy) + DDz(Bz, dz) 

    laplacian_B = np.stack([laplacian_B_xcomp, laplacian_B_ycomp, laplacian_B_zcomp])
    laplacian_B_magnitude = np.sqrt((laplacian_B**2).sum(0))
    
    plt.imshow(laplacian_B_magnitude.sum(-1).T, origin='lower')
    plt.savefig('fig.png', dpi=160)