"""
preprocess_images.py

Offline preprocessing pipeline for diabetic retinopathy fundus images:
  1. Crop retinal disk (remove black background borders)
  2. Resize to 512x512 (LANCZOS)
  3. Save to data_preprocessed/ with same split/class structure

Run once before training. Subsequent training loads from data_preprocessed/.
"""

import os
import sys
import time
from PIL import Image, ImageFilter
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Retinal disk crop
# ---------------------------------------------------------------------------


def crop_retinal_disk(
    img: Image.Image, threshold: int = 15, margin: float = 0.03
) -> Image.Image:
    """
    Crop a fundus image to a tight square around the retinal disk, removing
    the large black background borders typical in fundus photography.

    Algorithm:
      1. Apply a mild Gaussian blur to suppress JPEG artefacts at the border.
      2. Build a binary mask: pixels whose max-channel value > threshold are
         considered part of the retina.
      3. Find the axis-aligned bounding box of the mask.
      4. Add a small fractional margin, then pad to a square.

    Args:
        img:        PIL RGB image.
        threshold:  Max-channel value below which a pixel is background (0-255).
        margin:     Extra fractional border to keep around the detected disk.

    Returns:
        Cropped (square) PIL image, or the original if no foreground found.
    """
    arr_blur = np.array(img.filter(ImageFilter.GaussianBlur(radius=5)))
    mask = arr_blur.max(axis=2) > threshold

    h, w = arr_blur.shape[:2]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img  # entirely black – return unchanged

    rmin = int(np.where(rows)[0][0])
    rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0])
    cmax = int(np.where(cols)[0][-1])

    # Fractional margin
    rh = rmax - rmin
    cw = cmax - cmin
    pad = int(max(rh, cw) * margin)
    rmin = max(0, rmin - pad)
    rmax = min(h - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w - 1, cmax + pad)

    # Extend shorter axis to make it square (symmetric padding)
    rh = rmax - rmin
    cw = cmax - cmin
    if rh > cw:
        diff = rh - cw
        cmin = max(0, cmin - diff // 2)
        cmax = min(w - 1, cmax + diff - diff // 2)
    elif cw > rh:
        diff = cw - rh
        rmin = max(0, rmin - diff // 2)
        rmax = min(h - 1, rmax + diff - diff // 2)

    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


# ---------------------------------------------------------------------------
# Per-image processing worker
# ---------------------------------------------------------------------------


def process_image(
    src_path: str, dst_path: str, output_size: int = 512
) -> tuple[str, bool, str]:
    """
    Load, crop, resize, and save one image.

    Returns (dst_path, success, error_msg).
    """
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with Image.open(src_path) as img:
            img_rgb = img.convert("RGB")
        cropped = crop_retinal_disk(img_rgb)
        resized = cropped.resize((output_size, output_size), Image.LANCZOS)
        resized.save(dst_path, "JPEG", quality=95)
        return dst_path, True, ""
    except Exception as e:
        return dst_path, False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    src_root = "data_split"
    dst_root = "data_preprocessed"
    output_size = 512
    max_workers = 4

    splits = ["train", "val", "test"]

    # Collect all (src, dst) pairs
    tasks = []
    for split in splits:
        split_dir = os.path.join(src_root, split)
        if not os.path.isdir(split_dir):
            print(f"WARNING: {split_dir} not found, skipping.")
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                src = os.path.join(cls_dir, fname)
                dst = os.path.join(dst_root, split, cls, fname)
                tasks.append((src, dst))

    total = len(tasks)
    print(f"Preprocessing {total} images  ({output_size}x{output_size} output)")
    print(f"Source:      {src_root}/")
    print(f"Destination: {dst_root}/")
    print(f"Workers:     {max_workers}")
    print()

    errors = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_image, src, dst, output_size): (src, dst)
            for src, dst in tasks
        }
        for future in as_completed(futures):
            dst_path, success, err = future.result()
            done += 1
            if not success:
                errors.append((dst_path, err))
            if done % 200 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  {done}/{total}  ({rate:.1f} img/s)  ETA: {eta / 60:.1f} min")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min.")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for p, e in errors[:10]:
            print(f"  {p}: {e}")
    else:
        print("No errors.")

    # Print summary stats
    print("\nOutput directory contents:")
    for split in splits:
        split_dir = os.path.join(dst_root, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                n = len(os.listdir(cls_dir))
                print(f"  {split}/{cls}: {n}")


if __name__ == "__main__":
    main()
