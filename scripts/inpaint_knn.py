# Code from https://github.com/aGIToz/PyInpaint
# based on harmonic extension on a non-local graph

import numpy as np
from PIL import Image
from scipy import spatial
import skimage


def create_patches(image_np, patch_size):
    h, w, d = image_np.shape
    pad_width = [(int((patch_size - 0.5) / 2.), int((patch_size + 0.5) / 2.)),
                 (int((patch_size - 0.5) / 2.), int((patch_size + 0.5) / 2.)),
                 (0, 0)]
    padded_image = np.pad(image_np, pad_width=pad_width, mode='symmetric')
    patches = skimage.util.view_as_windows(padded_image, window_shape=(patch_size, patch_size, d))
    patches = patches.reshape((h * w, patch_size * patch_size * d))
    return patches


def inpaint(image, mask, patch_size, k_boundary=4, k_search=1000, k_patch=5):
    w, h = image.size
    image_np = np.array(image)
    if image_np.ndim == 2:
        image_np = image_np[..., None]
    mask_np = np.array(mask)
    mask_np[mask_np > 0] = 1
    mask_np = mask_np[..., None]
    image_np = image_np.astype(float) / 255.
    image_np = image_np * mask_np

    signals = image_np.reshape((h * w, -1))
    signals_mask = np.tile(mask_np, 3).reshape((h * w, -1))

    # positions of each pixel, and kdtree for positions
    xs = np.arange(0, w, 1)
    ys = np.arange(h, 0, -1)
    meshx, meshy = np.meshgrid(xs, ys)
    positions = np.stack((meshx.reshape(h * w), meshy.reshape(h * w)), axis=1)    # shape (n, 2)

    kdt_pos = spatial.cKDTree(positions)

    known_idxes = np.nonzero(mask_np.flatten())[0]
    unknown_idxes = np.nonzero(1 - mask_np.flatten())[0]
    # create patches for similarity nearest neighbor finding
    patches = create_patches(image_np, patch_size)
    mask_patches = create_patches(signals_mask.reshape((h, w, -1)), patch_size)

    while len(unknown_idxes) > 0:
        known_thisstep = np.array([]).astype(int)
        for i in unknown_idxes:
            _, nn = kdt_pos.query(positions[i], k_boundary)
            if not np.all(np.isin(nn, unknown_idxes)):
                known_thisstep = np.append(known_thisstep, i)
                # select a bunch of nearest neighbors based on position
                _, nn = kdt_pos.query(positions[i], k_search)
                nn = nn[~np.isin(nn, unknown_idxes)]
                nn_patches = patches[nn]
                nn_patches = nn_patches * mask_patches[i]
                kdt_patch = spatial.cKDTree(nn_patches)
                _, idx_in_nn = kdt_patch.query(patches[i], k_patch)
                nn2 = nn[idx_in_nn]
                signals[i] = signals[nn2].mean(axis=0)
        # update known and unknown area
        known_idxes = np.concatenate((known_idxes, known_thisstep), axis=0)
        unknown_idxes = unknown_idxes[~np.isin(unknown_idxes, known_thisstep)]
        # update patches and mask patches
        patches = create_patches(signals.reshape((h, w, -1)), patch_size)
        signals_mask[known_thisstep] = 1
        mask_patches = create_patches(signals_mask.reshape((h, w, -1)), patch_size)
    
    inpainted = signals.reshape((h, w, -1))
    inpainted = (inpainted * 255).astype(np.uint8)
    inpainted = Image.fromarray(inpainted)
    return inpainted
