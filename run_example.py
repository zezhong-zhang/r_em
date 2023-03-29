"""
Perform inference on a trained model using TensorFlow.

Author: Ivan Lobato
Email: Ivanlh20@gmail.com
"""

import os
import pathlib
from typing import Tuple
import numpy as np
import h5py
import matplotlib

# Check if running on remote SSH and use appropriate backend for matplotlib
remote_ssh = "SSH_CONNECTION" in os.environ
matplotlib.use('Agg' if remote_ssh else 'TkAgg')
import matplotlib.pyplot as plt

def fcn_set_gpu_id(gpu_visible_devices="0"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_visible_devices

fcn_set_gpu_id("0")

import tensorflow as tf

#########################################################################################
def load_test_data(path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    if path is None:
        path = pathlib.Path(__file__).resolve().parent / 'test_data.h5'
    else:
        path = pathlib.Path(path).resolve()

    with h5py.File(path, 'r') as h5file:
        x = np.asarray(h5file['x'][:], dtype=np.float32).transpose(0, 3, 2, 1)
        y = np.asarray(h5file['y'][:], dtype=np.float32).transpose(0, 3, 2, 1)
    
    return x, y

def fcn_inference():
    root = os.getcwd()

    # select network
    net = 'hrstem'

    # load its corresponding data
    fn_data = os.path.join(root, 'test_data', f'data_{net}.h5')
    x, y = load_test_data(fn_data)

    # load its corresponding model
    fn_model = os.path.join(root, 'models', f'r_{net}_model')
    net_r_cbed = tf.keras.models.load_model(fn_model)
    net_r_cbed.summary()

    n_data = x.shape[0]
    batch_size = 16

    # run inference
    y_p = net_r_cbed.predict(x, batch_size)

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    cb = [None, None, None]

    for ik in range(n_data):
        x_ik = x[ik, :, :, 0].squeeze()
        y_ik = y[ik, :, :, 0].squeeze()
        y_p_ik = y_p[ik, :, :, 0].squeeze()

        vmin = min(y_ik.min(), y_p_ik.min())
        vmax = max(y_ik.max(), y_p_ik.max())
    
        axs[0].imshow(x_ik, cmap='viridis')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].grid(False)
        axs[0].set_title(f"Detected {net} image", fontsize=14)
        if cb[0] is not None:
            cb[0].remove()
        cb[0] = fig.colorbar(axs[0].images[0], ax=axs[0], orientation='vertical', shrink=0.6)

        axs[1].imshow(y_p_ik, vmin=vmin, vmax=vmax, cmap='viridis')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].grid(False)
        axs[1].set_title(f"Restored {net} image", fontsize=14)
        if cb[1] is not None:
            cb[1].remove()
        cb[1] = fig.colorbar(axs[1].images[0], ax=axs[1], orientation='vertical', shrink=0.6)

        axs[2].imshow(y_ik, vmin=vmin, vmax=vmax, cmap='viridis')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].grid(False)
        axs[2].set_title(f"Ground truth {net} image", fontsize=14)
        if cb[2] is not None:
            cb[2].remove()
        cb[2] = fig.colorbar(axs[2].images[0], ax=axs[2], orientation='vertical', shrink=0.6)

        if remote_ssh:
            plt.savefig(f"restored_{net}.png", format='png')
        else:
            fig.show()

        print(ik)

if __name__ == '__main__':
    fcn_inference()