"""Small point-cloud viewer; no geometry modifications or measurements."""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from reconstruction.validation import validate_ply


def main():
    parser = argparse.ArgumentParser(description="Inspect a PLY cloud or mesh vertices in arbitrary scene units")
    parser.add_argument("path", type=Path)
    parser.add_argument("--cameras", type=Path, help="camera_poses.json (auto-discovered for pipeline outputs)")
    parser.add_argument("--save", type=Path, help="Save a three-view PNG instead of opening an interactive window")
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--bounds-percentile", type=float, default=.5, help="Zoom past extreme outliers for display only; use 0 to show full bounds")
    parser.add_argument("--vertical-axis", choices=("x", "y", "z"), default="y", help="Display orientation only; COLMAP often starts with camera Y down")
    parser.add_argument("--invert-vertical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open3d", action="store_true", help="Optional Open3D viewer with actual mesh faces")
    args = parser.parse_args()
    print(validate_ply(args.path))
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    if not 0 <= args.bounds_percentile < 10:
        parser.error("--bounds-percentile must be in [0, 10)")
    if args.open3d:
        try:
            import open3d as o3d
        except ImportError:
            parser.error("Install open3d for --open3d, or omit it for the matplotlib viewer")
        data = PlyData.read(args.path)
        geometry = o3d.io.read_triangle_mesh(str(args.path)) if "face" in data else o3d.io.read_point_cloud(str(args.path))
        o3d.visualization.draw_geometries([geometry])
        return
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vertices = PlyData.read(args.path, mmap="r")["vertex"].data
    indices = np.linspace(0, len(vertices) - 1, min(len(vertices), args.max_points), dtype=int)
    points = np.column_stack([vertices[k][indices] for k in ("x", "y", "z")])
    colors = np.column_stack([vertices[k][indices] for k in ("red", "green", "blue")]) / 255 if "red" in vertices.dtype.names else "steelblue"
    cameras_path = args.cameras or args.path.parent.parent / "sparse" / "camera_poses.json"
    cameras = None
    if cameras_path.is_file():
        cameras = np.array([im["position_world"] for im in json.loads(cameras_path.read_text())["images"]])
    fig = plt.figure(figsize=(17, 6) if args.save else (10, 8))
    low, high = np.percentile(points, [args.bounds_percentile, 100 - args.bounds_percentile], axis=0)
    if cameras is not None:
        low = np.minimum(low, cameras.min(axis=0))
        high = np.maximum(high, cameras.max(axis=0))
    margin = np.maximum((high - low) * .05, 1e-6)
    low, high = low - margin, high + margin
    views = [(20, 45), (0, 0), (90, -90)] if args.save else [(20, 45)]
    for i, (elevation, azimuth) in enumerate(views):
        if i > 0:
            a, b = (0, 2) if i == 1 else (0, 1)
            ax = fig.add_subplot(1, len(views), i + 1)
            ax.scatter(points[:, a], points[:, b], c=colors, s=.5, linewidths=0, rasterized=True)
            if cameras is not None:
                ax.plot(cameras[:, a], cameras[:, b], color="crimson", linewidth=.8)
            ax.set(xlabel=f'{"XYZ"[a]} (arbitrary units)', ylabel=f'{"XYZ"[b]} (arbitrary units)',
                   xlim=(low[a], high[a]), ylim=(low[b], high[b]))
            if args.invert_vertical and "xyz"[b] == args.vertical_axis:
                ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.grid(alpha=.2)
            continue
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        ax.scatter(*points.T, c=colors, s=.8, linewidths=0, rasterized=True)
        if cameras is not None:
            ax.plot(*cameras.T, color="crimson", linewidth=.8, label="Camera path")
            ax.scatter(*cameras[::10].T, color="crimson", s=4)
            ax.legend()
        ax.set(xlabel="X (arbitrary units)", ylabel="Y (arbitrary units)", zlabel="Z (arbitrary units)")
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
        ax.view_init(elev=elevation, azim=azimuth, vertical_axis=args.vertical_axis)
        ax.set_box_aspect(np.maximum(high - low, 1e-6))
        if args.invert_vertical:
            getattr(ax, f"invert_{args.vertical_axis}axis")()
    fig.suptitle(f"{args.path.name} — {len(vertices):,} vertices; {len(indices):,} sampled for display; {args.bounds_percentile:g}–{100 - args.bounds_percentile:g}% bounds")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=160)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
