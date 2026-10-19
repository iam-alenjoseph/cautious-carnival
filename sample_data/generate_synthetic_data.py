"""
generate_synthetic_data.py

Generates realistic synthetic multiplexed tissue tiles (Nuclear + Membrane channels)
and companion QuPath tile metadata for pipeline testing and verification.
"""

import os
import json
import numpy as np
import tifffile
from scipy import ndimage as ndi

def generate_synthetic_tile(
    height: int = 512,
    width: int = 512,
    num_cells: int = 65,
    num_channels: int = 4,
    random_seed: int = 42
) -> np.ndarray:
    """
    Creates a synthetic 4-channel image:
      - Channel 0: Nuclear (DAPI) - high intensity in nuclei with internal chromatin variation
      - Channel 1: Membrane / Cytoplasm (PanCK) - surrounding rings around nuclei
      - Channel 2: Immune Marker (CD45) - expression in subset of cells
      - Channel 3: Proliferation Marker (Ki67) - nuclear signal in cycling subset
    """
    np.random.seed(random_seed)

    nuclear_img = np.zeros((height, width), dtype=np.float32)
    membrane_img = np.zeros((height, width), dtype=np.float32)
    cd45_img = np.zeros((height, width), dtype=np.float32)
    ki67_img = np.zeros((height, width), dtype=np.float32)

    # Place random non-overlapping cell centers
    centers = []
    margin = 30
    attempts = 0
    max_attempts = num_cells * 100

    while len(centers) < num_cells and attempts < max_attempts:
        attempts += 1
        cx = np.random.randint(margin, width - margin)
        cy = np.random.randint(margin, height - margin)

        # Ensure minimal distance between nuclei
        if any(np.hypot(cx - x, cy - y) < 22 for x, y in centers):
            continue
        centers.append((cx, cy))

    # Grid coordinates
    y_coords, x_coords = np.ogrid[:height, :width]

    for idx, (cx, cy) in enumerate(centers):
        # Elliptical nuclear radius
        rx = np.random.uniform(6.0, 11.0)
        ry = np.random.uniform(6.0, 11.0)
        theta = np.random.uniform(0, np.pi)

        # Coordinate rotation for ellipse
        x_rot = (x_coords - cx) * np.cos(theta) + (y_coords - cy) * np.sin(theta)
        y_rot = -(x_coords - cx) * np.sin(theta) + (y_coords - cy) * np.cos(theta)
        dist_sq = (x_rot / rx)**2 + (y_rot / ry)**2

        # 1. Nuclear mask & gradient
        nuc_mask = dist_sq <= 1.0
        nuc_intensity = np.random.uniform(1500, 3500)
        nuclear_img[nuc_mask] += nuc_intensity * (1.0 - 0.3 * np.sqrt(dist_sq[nuc_mask]))

        # 2. Membrane / Cytoplasm ring
        mem_rx = rx + np.random.uniform(4.0, 7.0)
        mem_ry = ry + np.random.uniform(4.0, 7.0)
        mem_dist_sq = (x_rot / mem_rx)**2 + (y_rot / mem_ry)**2
        mem_ring = (mem_dist_sq <= 1.0) & (dist_sq > 0.8)
        mem_intensity = np.random.uniform(1200, 4000)
        membrane_img[mem_ring] += mem_intensity

        # 3. CD45 expression (in ~25% of cells)
        if idx % 4 == 0:
            cd45_img[mem_ring] += np.random.uniform(1800, 4500)

        # 4. Ki67 expression (in ~20% of cells)
        if idx % 5 == 0:
            ki67_img[nuc_mask] += np.random.uniform(2000, 5000)

    # Add realistic background autofluorescence and sensor Poisson/Gaussian noise
    def add_noise(img, bg=200, noise_std=30):
        noisy = img + bg + np.random.normal(0, noise_std, img.shape)
        return np.clip(noisy, 0, 65535).astype(np.uint16)

    ch0 = add_noise(nuclear_img, bg=250, noise_std=35)
    ch1 = add_noise(membrane_img, bg=180, noise_std=30)
    ch2 = add_noise(cd45_img, bg=150, noise_std=25)
    ch3 = add_noise(ki67_img, bg=150, noise_std=25)

    # Stack into (H, W, 4)
    multichannel = np.stack([ch0, ch1, ch2, ch3], axis=-1)
    return multichannel


def main():
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "tiles_export"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating synthetic multiplexed tiles in: {output_dir}")

    tiles_info = []
    tile_coords = [
        ("Sample_01_tile_00000_x0_y0", 0, 0, 42),
        ("Sample_01_tile_00001_x448_y0", 448, 0, 43),
        ("Sample_01_tile_00002_x0_y448", 0, 448, 44),
        ("Sample_01_tile_00003_x448_y448", 448, 448, 45),
    ]

    for tile_id, x, y, seed in tile_coords:
        tile_arr = generate_synthetic_tile(height=512, width=512, num_cells=50, random_seed=seed)
        tile_path = os.path.join(output_dir, f"{tile_id}.tif")
        tifffile.imwrite(tile_path, tile_arr, compression="zlib")

        tiles_info.append({
            "tile_id": tile_id,
            "file_name": f"{tile_id}.tif",
            "file_path": tile_path,
            "x": x,
            "y": y,
            "width": 512,
            "height": 512,
            "downsample": 1.0,
            "nuclear_channel": 0,
            "membrane_channel": 1
        })
        print(f"  -> Generated {tile_path} (Shape: {tile_arr.shape})")

    # Generate metadata JSON matching QuPath Groovy export
    metadata = {
        "image_name": "Sample_01",
        "pixel_size_um": 0.5,
        "target_pixel_size_um": 0.5,
        "downsample": 1.0,
        "total_tiles": len(tiles_info),
        "tiles": tiles_info
    }
    meta_path = os.path.join(output_dir, "tiles_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Synthetic test dataset generated successfully at: {output_dir}")
    print(f"[INFO] Tile metadata file: {meta_path}")


if __name__ == "__main__":
    main()

