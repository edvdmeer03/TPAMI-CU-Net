# Step2-mat2npy_h5.py
# Read MATLAB v7.3 .mat files (HDF5) and save as .npy

import numpy as np
import h5py
from pathlib import Path

HERE = Path(__file__).resolve().parent

def load_mat_v73(path, varname):
    with h5py.File(path, 'r') as f:
        dset = f[varname][:]
        # MATLAB (Fortran order) -> NumPy (C order): reverse axes to keep (H,W,C,N)
        dset = np.array(dset).transpose(tuple(range(dset.ndim))[::-1])
        return dset.astype(np.float32)

def main():
    # run this script from the same folder where the .mat files are
    x = load_mat_v73(HERE / 'data_x.mat', 'data_x')   # (H,W,C,N)
    y = load_mat_v73(HERE / 'data_y.mat', 'data_y')
    z = load_mat_v73(HERE / 'label.mat',  'label')

    # Save as .npy for the next step
    np.save(HERE / 'data_x.npy', x)
    np.save(HERE / 'data_y.npy', y)
    np.save(HERE / 'label.npy',  z)

    print('Saved:')
    print(HERE / 'data_x.npy')
    print(HERE / 'data_y.npy')
    print(HERE / 'label.npy')
    print('Shapes:', x.shape, y.shape, z.shape)

if __name__ == "__main__":
    main()
