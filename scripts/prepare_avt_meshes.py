from pathlib import Path
import csv

import numpy as np
import SimpleITK as sitk
import trimesh
from skimage.measure import marching_cubes


ROOT = Path("data/AVT/raw")
OUT = Path("data/AVT/meshes")
CSV = Path("data/AVT/mesh_qc.csv")

OUT.mkdir(parents=True, exist_ok=True)


def edge_topology(faces):
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])

    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)

    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))

    return boundary, nonmanifold


def segmentation_to_binary(img):
    arr = sitk.GetArrayFromImage(img)

    # Vector image: [z, y, x, channels]
    if img.GetNumberOfComponentsPerPixel() > 1:
        arr = np.any(arr != 0, axis=-1)

    # 4D Slicer segmentation: usually [layer, z, y, x]
    while arr.ndim > 3:
        arr = np.any(arr != 0, axis=0)

    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected segmentation array shape: {arr.shape}")

    return arr.astype(np.uint8)


def vertices_to_physical(vertices_zyx, img):
    """
    marching_cubes returns coordinates as z,y,x.
    Convert them into physical x,y,z coordinates using
    spacing/origin/direction from the NRRD.
    """

    spacing = np.asarray(img.GetSpacing()[:3], dtype=float)
    origin = np.asarray(img.GetOrigin()[:3], dtype=float)

    dim = img.GetDimension()
    full_direction = np.asarray(img.GetDirection(), dtype=float).reshape(dim, dim)
    direction = full_direction[:3, :3]

    # z,y,x -> x,y,z
    indices_xyz = vertices_zyx[:, [2, 1, 0]]

    scaled = indices_xyz * spacing[None, :]
    physical = origin[None, :] + scaled @ direction.T

    return physical


def process(seg_path):
    case = seg_path.name.replace(".seg.nrrd", "")
    cohort = seg_path.parent.parent.name

    print(f"\n=== {cohort}/{case} ===", flush=True)

    img = sitk.ReadImage(str(seg_path))
    binary = segmentation_to_binary(img)

    voxels = int(binary.sum())

    if voxels == 0:
        raise RuntimeError("Empty segmentation")

    # Padding guarantees a closed marching-cubes surface even if
    # segmentation touches the image boundary.
    padded = np.pad(binary, 1, mode="constant", constant_values=0)

    verts, faces, _, _ = marching_cubes(
        padded.astype(np.float32),
        level=0.5,
        allow_degenerate=False,
    )

    # compensate for one-voxel padding
    verts -= 1.0

    verts = vertices_to_physical(verts, img)

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=False,
    )

    # Minimal deterministic cleanup.
    # No smoothing, hole filling or decimation here.
    if len(mesh.faces):
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()

    if len(mesh.faces):
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()

    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
    except Exception:
        pass

    boundary, nonmanifold = edge_topology(np.asarray(mesh.faces))

    try:
        components = len(mesh.split(only_watertight=False))
    except Exception:
        components = -1

    finite = bool(np.isfinite(mesh.vertices).all())

    row = {
        "cohort": cohort,
        "case": case,
        "source": str(seg_path),
        "voxels": voxels,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(components),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "watertight": bool(mesh.is_watertight),
        "winding": bool(mesh.is_winding_consistent),
        "finite": finite,
    }

    # Initial resource/QC flag.
    # Do NOT interpret this as final cohort selection yet.
    row["under_80k"] = int(len(mesh.vertices) <= 80000)

    row["basic_qc"] = int(
        finite
        and len(mesh.vertices) > 0
        and len(mesh.faces) > 0
        and boundary == 0
        and nonmanifold == 0
        and mesh.is_watertight
        and mesh.is_winding_consistent
    )

    output = OUT / f"{cohort}_{case}.ply"
    mesh.export(output)

    print(
        f"V={row['vertices']} "
        f"F={row['faces']} "
        f"components={components} "
        f"boundary={boundary} "
        f"nonmanifold={nonmanifold} "
        f"watertight={row['watertight']} "
        f"winding={row['winding']} "
        f"<=80k={row['under_80k']}",
        flush=True,
    )

    return row


def main():
    segs = sorted(ROOT.rglob("*.seg.nrrd"))

    print(f"Found {len(segs)} segmentations")

    if len(segs) != 56:
        print(
            f"WARNING: expected 56 AVT cases, found {len(segs)}",
            flush=True,
        )

    rows = []

    for i, seg in enumerate(segs, 1):
        print(f"\n[{i}/{len(segs)}] {seg}", flush=True)

        try:
            row = process(seg)
            row["error"] = ""
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

            row = {
                "cohort": seg.parent.parent.name,
                "case": seg.name.replace(".seg.nrrd", ""),
                "source": str(seg),
                "voxels": "",
                "vertices": "",
                "faces": "",
                "components": "",
                "boundary_edges": "",
                "nonmanifold_edges": "",
                "watertight": "",
                "winding": "",
                "finite": "",
                "under_80k": 0,
                "basic_qc": 0,
                "error": repr(e),
            }

        rows.append(row)

    fields = [
        "cohort",
        "case",
        "source",
        "voxels",
        "vertices",
        "faces",
        "components",
        "boundary_edges",
        "nonmanifold_edges",
        "watertight",
        "winding",
        "finite",
        "under_80k",
        "basic_qc",
        "error",
    ]

    with CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========================================")
    print(f"Processed: {len(rows)}")
    print(f"CSV:       {CSV}")
    print(f"Meshes:    {OUT}")
    print("========================================")


if __name__ == "__main__":
    main()
