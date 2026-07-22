"""
Poisson Reconstruction from filtered Gaussian splatting point cloud.

This script reads a Gaussian-splatting .ply file, applies the same
opacity and radius filters, generates a filtered point cloud, and then
reconstructs a mesh using Open3D Poisson surface reconstruction.
"""

import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData
import open3d as o3d


def load_and_filter_gaussians(ply_path: str, opacity_threshold: float = 0.5,
                              radius: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Load PLY gaussians and filter by opacity and radius."""
    print(f"Loading PLY file: {ply_path}")
    ply = PlyData.read(ply_path)
    v = ply["vertex"]

    print(f"Total gaussians: {len(v):,}")
    print(f"Available attributes: {v.data.dtype.names}")

    positions = np.column_stack([v["x"], v["y"], v["z"]])

    center = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)

    mask_radius = distances < radius
    positions_filtered = positions[mask_radius]
    v_filtered = v[mask_radius]
    print(f"After radius filter ({radius}): {len(positions_filtered):,} gaussians")

    if "opacity" in v.data.dtype.names:
        opacity = v_filtered["opacity"]
        if np.all(opacity < 1.0) and np.all(opacity > -20):
            opacity = 1.0 / (1.0 + np.exp(-opacity))
        mask_opacity = opacity >= opacity_threshold
        positions_filtered = positions_filtered[mask_opacity]
        v_filtered = v_filtered[mask_opacity]
        print(f"After opacity filter (>= {opacity_threshold}): {len(positions_filtered):,} gaussians")
    else:
        print("Warning: No opacity attribute found in PLY file")

    normals = None
    if all(name in v.data.dtype.names for name in ("nx", "ny", "nz")):
        normals = np.column_stack([v_filtered["nx"], v_filtered["ny"], v_filtered["nz"]])
        print("Loaded normals from PLY file")
    else:
        print("No normals found in PLY file; normals will be estimated")

    return positions_filtered, normals


def create_point_cloud(positions: np.ndarray, normals: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from filtered positions and normals."""
    print("Creating filtered point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def normalize_normals(pcd: o3d.geometry.PointCloud):
    """Normalize any existing normals on the point cloud."""
    if len(pcd.normals) == 0:
        return
    normals = np.asarray(pcd.normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] = normals[valid] / lengths[valid, None]
    pcd.normals = o3d.utility.Vector3dVector(normals)


def clean_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Remove invalid and duplicate points from the point cloud."""
    print("Cleaning point cloud...")
    pcd.remove_non_finite_points()
    pcd.remove_duplicated_points()
    pcd.remove_radius_outlier(nb_points=16, radius=0.01)
    print(f"Cleaned point cloud has {len(pcd.points):,} points")
    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.2, max_nn: int = 30):
    """Estimate and orient normals for the point cloud."""
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(100)


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: Path):
    """Save filtered point cloud to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving filtered point cloud to {output_path}")
    o3d.io.write_point_cloud(str(output_path), pcd)


def run_poisson(pcd: o3d.geometry.PointCloud, depth: int = 9, scale: float = 1.1,
                linear_fit: bool = False) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Perform Poisson surface reconstruction."""
    print(f"Running Poisson reconstruction (depth={depth}, scale={scale}, linear_fit={linear_fit})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit
    )
    print(f"Reconstruction complete: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    return mesh, np.asarray(densities)


def filter_mesh_by_density(mesh: o3d.geometry.TriangleMesh, densities: np.ndarray,
                           density_threshold: float = 0.01) -> o3d.geometry.TriangleMesh:
    """Filter low-density vertices from reconstructed mesh."""
    if densities.size == 0:
        print("No density values returned; skipping density filtering.")
        return mesh
    print(f"Filtering mesh by density: threshold={density_threshold}")
    density_min = densities.min()
    density_max = densities.max()
    if density_max <= density_min:
        print("Density values are constant; skipping density filtering.")
        return mesh
    densities = (densities - density_min) / (density_max - density_min)
    vertices_to_keep = densities >= density_threshold
    if len(vertices_to_keep) != len(mesh.vertices):
        mesh.remove_vertices_by_mask(~vertices_to_keep)
    print(f"Filtered mesh has {len(mesh.vertices):,} vertices and {len(mesh.triangles):,} triangles")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Poisson reconstruction from filtered Gaussian splatting PLY")
    parser.add_argument("--input", type=str,
                        default="C:\\Users\\leona\\Desktop\\graphics\\outputZip\\kaggle\\working\\output\\point_cloud\\iteration_30000\\point_cloud.ply",
                        help="Input Gaussian splatting PLY file")
    parser.add_argument("--output-dir", type=str,
                        default="C:\\Users\\leona\\Desktop\\graphics\\mesh_output",
                        help="Output directory for point cloud and reconstructed mesh")
    parser.add_argument("--opacity-threshold", type=float, default=0.1,
                        help="Opacity threshold for filtering gaussians")
    parser.add_argument("--radius", type=float, default=10.0,
                        help="Radius from scene center for filtering gaussians")
    parser.add_argument("--poisson-depth", type=int, default=11,
                        help="Poisson reconstruction depth")
    parser.add_argument("--density-threshold", type=float, default=0.001,
                        help="Density threshold for filtering low-confidence mesh vertices")
    parser.add_argument("--save-point-cloud", action="store_true",
                        help="Save the filtered point cloud to disk")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions, normals = load_and_filter_gaussians(
        args.input,
        opacity_threshold=args.opacity_threshold,
        radius=args.radius
    )

    pcd = create_point_cloud(positions, normals)
    pcd = clean_point_cloud(pcd)

    if args.save_point_cloud:
        save_point_cloud(pcd, output_dir / "filtered_gaussians.ply")

    if len(pcd.normals) == 0:
        estimate_normals(pcd)
    else:
        normalize_normals(pcd)
        print("Point cloud already contains normals")

    mesh, densities = run_poisson(pcd, depth=args.poisson_depth)
    if len(mesh.vertices) == 0 or densities.size == 0:
        print("Poisson reconstruction returned no geometry; retrying with estimated normals...")
        estimate_normals(pcd)
        mesh, densities = run_poisson(pcd, depth=args.poisson_depth)

    mesh = filter_mesh_by_density(mesh, densities, density_threshold=args.density_threshold)

    output_mesh_path = output_dir / f"mesh_poisson_depth{args.poisson_depth}.ply"
    print(f"Saving reconstructed mesh to {output_mesh_path}")
    o3d.io.write_triangle_mesh(str(output_mesh_path), mesh)
    print("Done.")


if __name__ == "__main__":
    main()
