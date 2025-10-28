from cellpose import models
from tifffile import imread, imwrite as imsave
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt
import os
from datetime import datetime

# --------------- CONFIG ----------------
root_dir = "Train"
output_dir = "./results_cellpose_full_features"
os.makedirs(output_dir, exist_ok=True)

subfolders = ["01", "02"]
gt_folder_map = {"01": "01_GT/SEG", "02": "02_GT/SEG"}

# ---- Load pretrained Cellpose model with GPU enabled ----
model = models.CellposeModel(gpu=True)
print(f"Cellpose model loaded with GPU: {model.gpu}")

def normalize_image(img):
    """Normalize uint16/float images to 0–1."""
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def extract_all_features(mask, intensity_image):
    """Extract detailed 3D morphological and intensity features."""
    # ← FIXED: Only safe 3D-compatible properties (no convex hull dependencies)
    props = regionprops_table(
        mask,
        intensity_image=intensity_image,
        properties=[
            'label',
            'area',                     # Volume in 3D (number of voxels)
            'bbox_area',                # Bounding box area
            'centroid',                 # 3D centroid coordinates (centroid-0, centroid-1, centroid-2)
            'equivalent_diameter_area', # Diameter of sphere with same volume
            'extent',                   # Ratio of pixels in bounding box
            'mean_intensity',
            'max_intensity',
            'min_intensity',
        ]
    )
    df = pd.DataFrame(props)
    
    # Add derived features
    if len(df) > 0:
        df["volume"] = df["area"]  # In 3D, area = volume (voxel count)
        df["compactness"] = df["bbox_area"] / (df["area"] + 1e-6)  # How much cell fills its bounding box
    
    return df

all_features = []

for folder in subfolders:
    data_dir = os.path.join(root_dir, folder)
    gt_dir = os.path.join(root_dir, gt_folder_map[folder])
    tif_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif')])
    
    print(f"\nProcessing folder {folder} with {len(tif_files)} timepoints")
    print(f"Folder started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for idx, tif_file in enumerate(tif_files):
        print(f"\nProcessing {folder} timepoint {idx+1}/{len(tif_files)}: {tif_file}")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        
        img_path = os.path.join(data_dir, tif_file)

        # ---- Load and normalize ----
        print("Loading image...")
        img = imread(img_path)
        img = normalize_image(img)
        print(f"Image shape: {img.shape}")

        # ---- Run Cellpose 3D segmentation ----
        print("Running Cellpose segmentation...")
        masks, flows, styles = model.eval(
            img,
            diameter=None,
            channels=[0, 0],
            z_axis=0,
            do_3D=True
        )
        print(f"Segmentation complete - found {len(np.unique(masks))-1} cells")

        # ---- Extract features ----
        print("Extracting features...")
        df_features = extract_all_features(masks, img)
        df_features["timepoint"] = idx
        df_features["subfolder"] = folder
        all_features.append(df_features)

        # ---- Save mask ----
        print("Saving mask...")
        mask_save_path = os.path.join(output_dir, f"{folder}_{tif_file}_mask.tif")
        imsave(mask_save_path, masks.astype(np.uint16))

        # ---- Ground truth mask ----
        print("Processing ground truth...")
        gt_mask_path = os.path.join(gt_dir, tif_file)
        gt_exists = os.path.exists(gt_mask_path)
        gt_mask = imread(gt_mask_path) if gt_exists else None

        # ---- Visualization ----
        print("Creating visualization...")
        mid_z = img.shape[0] // 2
        if gt_exists:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        else:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(img[mid_z], cmap='gray')
        axs[0].set_title("Original Image (mid-Z)")
        axs[1].imshow(masks[mid_z], cmap='nipy_spectral')
        axs[1].set_title("Predicted Mask (mid-Z)")
        if gt_exists:
            axs[2].imshow(gt_mask[mid_z], cmap='nipy_spectral')
            axs[2].set_title("GT Mask (mid-Z)")
            axs[3].imshow(img[mid_z], cmap='gray')
            axs[3].imshow(masks[mid_z], cmap='nipy_spectral', alpha=0.5)
            axs[3].imshow(gt_mask[mid_z], cmap='cool', alpha=0.3)
            axs[3].set_title("Pred + GT Overlay")
        else:
            axs[2].imshow(img[mid_z], cmap='gray')
            axs[2].imshow(masks[mid_z], cmap='nipy_spectral', alpha=0.4)
            axs[2].set_title("Overlay")
        for a in axs: a.axis('off')
        plt.tight_layout()
        overlay_path = os.path.join(output_dir, f"{folder}_{tif_file}_overlay_GT.png")
        plt.savefig(overlay_path)
        plt.close()
        
        print(f"Timepoint {idx+1} completed at: {datetime.now().strftime('%H:%M:%S')}")
    
    print(f"Folder {folder} completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---- Merge and save all features ----
print(f"\n💾 Saving training features...")
final_df = pd.concat(all_features, ignore_index=True)
csv_path = os.path.join(output_dir, "full_segmentation_features.csv")
final_df.to_csv(csv_path, index=False)

print(f"\nSaved full feature table to: {csv_path}")
print(f"Extracted {len(final_df)} cells across all timepoints/folders.")
print("\nExample columns:")
print(final_df.columns.tolist())

# ===================== PROCESS TEST FOLDERS =====================
print(f"\nStarting TEST data processing at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
test_root = "Test"
test_folders = ["01", "02"]
test_all_features = []

for folder in test_folders:
    test_data_dir = os.path.join(test_root, folder)
    if not os.path.isdir(test_data_dir):
        print(f"Test subfolder missing: {test_data_dir}")
        continue
    
    test_tif_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith('.tif')])
    print(f"\nProcessing TEST folder {folder} with {len(test_tif_files)} timepoints")
    print(f"TEST folder started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for idx, tif_file in enumerate(test_tif_files):
        print(f"\n[TEST] Processing {folder} timepoint {idx+1}/{len(test_tif_files)}: {tif_file}")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        
        img_path = os.path.join(test_data_dir, tif_file)

        # ---- Load and normalize ----
        print("Loading TEST image...")
        img = imread(img_path)
        img = normalize_image(img)
        print(f"Image shape: {img.shape}")

        # ---- Run Cellpose 3D segmentation ----
        print("   🔬 Running Cellpose segmentation...")
        masks, flows, styles = model.eval(
            img,
            diameter=None,
            channels=[0, 0],
            z_axis=0,
            do_3D=True
        )
        print(f"Segmentation complete - found {len(np.unique(masks))-1} cells")

        # ---- Extract features ----
        print("Extracting features...")
        df_features = extract_all_features(masks, img)
        df_features["timepoint"] = idx
        df_features["subfolder"] = folder
        test_all_features.append(df_features)

        # ---- Save mask ----
        print("   💾 Saving TEST mask...")
        mask_save_path = os.path.join(output_dir, f"test_{folder}_{tif_file}_mask.tif")
        imsave(mask_save_path, masks.astype(np.uint16))

        # ---- Simple visualization (no GT) ----
        print("Creating TEST visualization...")
        mid_z = img.shape[0] // 2
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img[mid_z], cmap='gray')
        axs[0].set_title("Original Image (mid-Z)")
        axs[1].imshow(masks[mid_z], cmap='nipy_spectral')
        axs[1].set_title("Predicted Mask (mid-Z)")
        axs[2].imshow(img[mid_z], cmap='gray')
        axs[2].imshow(masks[mid_z], cmap='nipy_spectral', alpha=0.4)
        axs[2].set_title("Overlay")
        for a in axs: a.axis('off')
        plt.tight_layout()
        overlay_path = os.path.join(output_dir, f"test_{folder}_{tif_file}_overlay.png")
        plt.savefig(overlay_path)
        plt.close()
        
        print(f"TEST timepoint {idx+1} completed at: {datetime.now().strftime('%H:%M:%S')}")
    
    print(f"TEST folder {folder} completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---- Merge and save all test features ----
if len(test_all_features) > 0:
    print(f"\n💾 Saving TEST features...")
    test_final_df = pd.concat(test_all_features, ignore_index=True)
    test_csv_path = os.path.join(output_dir, "test_segmentation_features.csv")
    test_final_df.to_csv(test_csv_path, index=False)
    print(f"\nSaved TEST feature table to: {test_csv_path}")
    print(f"Extracted {len(test_final_df)} cells across all TEST timepoints/folders.")
    print("\nExample (TEST) columns:")
    print(test_final_df.columns.tolist())
else:
    print("\nNo test features extracted (check test folder structure and .tif presence).")

print(f"\nALL PROCESSING COMPLETED at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
