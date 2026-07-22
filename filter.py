import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PLY = r"C:\Users\leona\Downloads\output\kaggle\working\output\43059ab5-8\point_cloud\iteration_30000\point_cloud.ply"
OUTPUT_PLY = r"C:\Users\leona\Downloads\output\kaggle\working\output\43059ab5-8\point_cloud\iteration_30000\cube_filtered_output.ply"

MAX_DISTANCE = 10.0

# Opacity threshold
MIN_OPACITY = 0.01

# Maximum allowed scale magnitude
MAX_SCALE_NORM = 15.0
    
# Maximum anisotropy ratio
MAX_ANISOTROPY = 80.0

# Radius-based outlier removal
NEIGHBOR_RADIUS = 0.5
MIN_NEIGHBORS = 5

# ============================================================
# LOAD PLY
# ============================================================

print("Loading PLY...")

plydata = PlyData.read(INPUT_PLY)
vertex = plydata["vertex"]

num_points = len(vertex)

print(f"Loaded {num_points} gaussians")

# ============================================================
# EXTRACT ATTRIBUTES
# ============================================================

positions = np.vstack([
    vertex["x"],
    vertex["y"],
    vertex["z"]
]).T

opacity = np.array(vertex["opacity"])

scales = np.vstack([
    vertex["scale_0"],
    vertex["scale_1"],
    vertex["scale_2"]
]).T

rotations = np.vstack([
    vertex["rot_0"],
    vertex["rot_1"],
    vertex["rot_2"],
    vertex["rot_3"]
]).T


distances = np.linalg.norm(positions, axis=1)

print(positions[1])
def is_within_box(points, x1, x2, y1, y2, z1, z2):
    return (
        (points[:, 0] >= x1) & (points[:, 0] <= x2) &
        (points[:, 1] >= y1) & (points[:, 1] <= y2) &
        (points[:, 2] >= z1) & (points[:, 2] <= z2)
    )

#RECTANGLE = np.array([[-5, -5, -1], [5, -5, -1], [5, 5, -1], [-5, 5, -1], [-5, -5, 5], [5, -5, 5], [5, 5, 5], [-5, 5, 5]])
#distances_to_corners = np.linalg.norm(positions[:, None, :] - RECTANGLE[None, :, :], axis=2)

#distance_mask = np.any(distances_to_corners < MAX_DISTANCE, axis=1)
dist_mask = distances < MAX_DISTANCE

cube_mask = ~is_within_box(positions, -10, 10, -10, 10, -4, 6)
#rect_mask = np.all((positions >= RECTANGLE[0]) & (positions <= RECTANGLE[1]) & (positions <= RECTANGLE[2]) & (positions <= RECTANGLE[3]) & (positions <= RECTANGLE[4]) & (positions <= RECTANGLE[5]) & (positions <= RECTANGLE[6]) & (positions <= RECTANGLE[7]), axis=1)
print(f"Remaining after distance filter: {dist_mask.sum()}")
print(f"Remaining after cube filter: {cube_mask.sum()}")
#print(f"Remaining after rectangle filter: {distance_mask.sum()}")

# ============================================================
# FILTER 1: OPACITY
# ============================================================

print("Filtering low-opacity gaussians...")

mask_opacity = opacity > MIN_OPACITY

print(f"Remaining after opacity filter: {mask_opacity.sum()}")

# ============================================================
# FILTER 2: SCALE MAGNITUDE
# ============================================================

print("Filtering oversized gaussians...")

scale_norm = np.linalg.norm(scales, axis=1)

mask_scale = scale_norm < MAX_SCALE_NORM

print(f"Remaining after scale filter: {mask_scale.sum()}")

# ============================================================
# FILTER 3: ANISOTROPY
# ============================================================

print("Filtering highly anisotropic gaussians...")

anisotropy = np.max(scales, axis=1) / (
    np.min(scales, axis=1) + 1e-8
)

mask_anisotropy = anisotropy < MAX_ANISOTROPY

print(f"Remaining after anisotropy filter: {mask_anisotropy.sum()}")

# ============================================================
# COMBINE BASIC FILTERS
# ============================================================

combined_mask = (
    mask_opacity &
    mask_scale &
    mask_anisotropy &
    dist_mask
)

filtered_positions = positions[combined_mask]

print(f"Remaining after basic filtering: {len(filtered_positions)}")

# ============================================================
# FILTER 4: SPATIAL OUTLIER REMOVAL
# ============================================================

print("Removing isolated spatial outliers...")

tree = cKDTree(filtered_positions)

neighbor_counts = np.array([
    len(tree.query_ball_point(p, NEIGHBOR_RADIUS))
    for p in filtered_positions
])

mask_neighbors = neighbor_counts > MIN_NEIGHBORS

print(f"Remaining after neighbor filtering: {mask_neighbors.sum()}")

# ============================================================
# FINAL MASK
# ============================================================

final_indices = np.where(combined_mask)[0]
final_indices = final_indices[mask_neighbors]

print(f"Final gaussian count: {len(final_indices)}")

# ============================================================
# BUILD NEW PLY
# ============================================================

print("Saving filtered PLY...")

new_vertex_data = vertex[final_indices]

new_ply = PlyData([
    PlyElement.describe(new_vertex_data, "vertex")
], text=False)

new_ply.write(OUTPUT_PLY)

print(f"Saved filtered file to: {OUTPUT_PLY}")

# ============================================================
# OPTIONAL STATS
# ============================================================

removed = num_points - len(final_indices)

print("================================================")
print(f"Original gaussians : {num_points}")
print(f"Filtered gaussians : {len(final_indices)}")
print(f"Removed gaussians  : {removed}")
print("================================================")