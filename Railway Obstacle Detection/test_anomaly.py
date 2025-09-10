"""
Minimal anomaly-cluster extractor
Steps:
1) Read background, crop region, and input frames
2) Preprocess (downsample -> crop -> remove far points)
3) Background comparison (difference by radius)
4) Post-process (drop tiny clusters -> merge close clusters)
5) Output anomaly clusters as .pcd files
"""

import os
import numpy as np
import open3d as o3d
from tqdm import tqdm


# ------------------------------ Basic utils ------------------------------ #

def np_to_pcloud(x):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x)) if x is not None and x.size else o3d.geometry.PointCloud()

def pcloud_to_np(pcd):
    return np.asarray(pcd.points) if pcd and len(pcd.points) else np.empty((0, 3), float)

def downsample(pcd, voxel):
    return pcd.voxel_down_sample(voxel) if pcd and len(pcd.points) and voxel > 0 else pcd

def crop_with_obbox(crop_src, pcd):
    obb = crop_src.get_oriented_bounding_box()
    return pcd.crop(obb)

def remove_far_points(pcd, center, max_dist):
    if len(pcd.points) == 0:
        return pcd
    pts = pcloud_to_np(pcd)
    d2 = np.sum((pts - center) ** 2, axis=1)
    keep = d2 <= (max_dist ** 2)
    return np_to_pcloud(pts[keep])

def preprocess_frame(frame, crop_region, voxel, max_far_dist):
    frame = downsample(frame, voxel)
    frame = crop_with_obbox(downsample(crop_region, voxel), frame)
    center = crop_region.get_oriented_bounding_box().center
    frame = remove_far_points(frame, center, max_far_dist)
    return frame

def preprocess_background(background, crop_region, voxel, max_far_dist):
    background = downsample(background, voxel)
    background = crop_with_obbox(downsample(crop_region, voxel), background)
    center = crop_region.get_oriented_bounding_box().center
    background = remove_far_points(background, center, max_far_dist)
    return background


# ----------------------- Difference & clustering -------------------------- #

def difference_points(bg, frame, radius):
    """Keep points in frame that have no neighbor within `radius` in bg."""
    pts = pcloud_to_np(frame)
    if pts.size == 0:
        return np.empty((0, 3), float)
    tree = o3d.geometry.KDTreeFlann(bg)
    out = [p for p in frame.points if tree.search_hybrid_vector_3d(p, radius, 1)[0] == 0]
    return np.asarray(out) if out else np.empty((0, 3), float)

def dbscan_clusters(pts, eps, min_points):
    if pts.size == 0:
        return [], []
    labels = np.array(np_to_pcloud(pts).cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    clusters = []
    for lab in np.unique(labels[labels >= 0]):
        clusters.append(np_to_pcloud(pts[labels == lab]))
    return clusters, labels

def remove_small_clusters(clusters, min_size):
    return [c for c in clusters if len(c.points) >= min_size]

def cluster_centroid(pcd):
    arr = pcloud_to_np(pcd)
    return np.mean(arr, axis=0) if arr.size else np.zeros(3)

def merge_close_clusters(clusters, merge_dist):
    """Greedy merge by centroid distance."""
    if not clusters:
        return []
    centroids = [cluster_centroid(c) for c in clusters]
    used = [False] * len(clusters)
    merged = []

    for i in range(len(clusters)):
        if used[i]:
            continue
        base = clusters[i]
        base_centroid = centroids[i]
        group = [base]
        used[i] = True
        for j in range(i + 1, len(clusters)):
            if used[j]:
                continue
            if np.linalg.norm(base_centroid - centroids[j]) <= merge_dist:
                group.append(clusters[j])
                used[j] = True
        # concatenate points of grouped clusters
        merged_pts = np.vstack([pcloud_to_np(g) for g in group if len(g.points)])
        merged.append(np_to_pcloud(merged_pts))
    return merged


# --------------------------------- I/O ------------------------------------ #

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_clusters(clusters, out_dir, stem):
    ensure_dir(out_dir)
    for idx, c in enumerate(clusters):
        if len(c.points) == 0:
            continue
        out_path = os.path.join(out_dir, f"{stem}_cluster_{idx}.pcd")
        o3d.io.write_point_cloud(out_path, c)


# --------------------------------- Main ----------------------------------- #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Minimal anomaly cluster extraction")
    parser.add_argument("--background", required=True, help="Background point cloud (.pcd/.ply)")
    parser.add_argument("--crop", required=True, help="Crop region point cloud (.pcd/.ply)")
    parser.add_argument("--input_root", required=True, help="Input root: expects <scene>/pcd/*.pcd")
    parser.add_argument("--output_root", default="./output_clusters", help="Output directory")
    # preprocess
    parser.add_argument("--voxel", type=float, default=0.1, help="Voxel size for downsampling")
    parser.add_argument("--max_far_dist", type=float, default=200.0, help="Remove points farther than this (meters)")
    # difference & clustering
    parser.add_argument("--diff_radius", type=float, default=0.06, help="Neighbor radius for background comparison")
    parser.add_argument("--dbscan_eps", type=float, default=0.2, help="DBSCAN eps")
    parser.add_argument("--dbscan_min_points", type=int, default=19, help="DBSCAN min points")
    # post-process
    parser.add_argument("--min_cluster_size", type=int, default=15, help="Drop clusters smaller than this")
    parser.add_argument("--merge_dist", type=float, default=0.6, help="Merge clusters whose centroids are within this")
    args = parser.parse_args()

    ensure_dir(args.output_root)

    # read background & crop
    background_raw = o3d.io.read_point_cloud(args.background)
    crop_region = o3d.io.read_point_cloud(args.crop)

    # preprocess background once
    background = preprocess_background(background_raw, crop_region, args.voxel, args.max_far_dist)

    # iterate scenes/frames
    scenes = sorted([d for d in os.listdir(args.input_root) if os.path.isdir(os.path.join(args.input_root, d))])
    for scene in scenes:
        scene_in = os.path.join(args.input_root, scene, "pcd")
        if not os.path.isdir(scene_in):
            continue
        scene_out = os.path.join(args.output_root, scene)
        ensure_dir(scene_out)

        frames = [f for f in sorted(os.listdir(scene_in)) if f.lower().endswith((".pcd", ".ply"))]
        for fname in tqdm(frames, desc=f"[{scene}]"):
            fpath = os.path.join(scene_in, fname)
            frame_raw = o3d.io.read_point_cloud(fpath)

            # preprocess frame
            frame = preprocess_frame(frame_raw, crop_region, args.voxel, args.max_far_dist)

            # difference against background
            diff_np = difference_points(background, frame, args.diff_radius)

            # cluster -> drop tiny -> merge close
            clusters, _ = dbscan_clusters(diff_np, args.dbscan_eps, args.dbscan_min_points)
            clusters = remove_small_clusters(clusters, args.min_cluster_size)
            clusters = merge_close_clusters(clusters, args.merge_dist)

            # save
            stem = os.path.splitext(fname)[0]
            save_clusters(clusters, scene_out, stem)

    print("Done.")


if __name__ == "__main__":
    main()
