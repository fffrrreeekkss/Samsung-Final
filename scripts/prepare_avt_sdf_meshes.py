from pathlib import Path
import csv
import gc
import re

import numpy as np
import SimpleITK as sitk
import trimesh
from skimage.measure import marching_cubes


ROOT = Path("data/AVT/raw")

# Старые meshes НЕ перезаписываем
OUT = Path("data/AVT/meshes_sdf_1mm")
CSV = Path("data/AVT/mesh_qc_sdf_1mm.csv")

TARGET_SPACING_MM = 1.0
SMOOTH_SIGMA_MM = 0.5
MARGIN_MM = 12.0

OUT.mkdir(parents=True, exist_ok=True)


def safe_name(s):
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    return s.strip("_")


def edge_topology(faces):
    faces = np.asarray(faces, dtype=np.int64)

    if len(faces) == 0:
        return 0, 0

    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])

    edges = np.sort(edges, axis=1)

    _, counts = np.unique(
        edges,
        axis=0,
        return_counts=True,
    )

    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))

    return boundary, nonmanifold


def vertices_to_physical(vertices_zyx, img):
    """
    marching_cubes:
        array coordinates = z, y, x

    Convert to physical x, y, z in mm.
    """
    spacing = np.asarray(img.GetSpacing(), dtype=float)
    origin = np.asarray(img.GetOrigin(), dtype=float)
    direction = np.asarray(
        img.GetDirection(),
        dtype=float,
    ).reshape(3, 3)

    xyz = vertices_zyx[:, [2, 1, 0]]

    return (
        origin[None, :]
        + (xyz * spacing[None, :]) @ direction.T
    )


def crop_mask(mask):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)

    if not stats.HasLabel(1):
        raise RuntimeError("Empty segmentation")

    bbox = stats.GetBoundingBox(1)

    index = list(bbox[:3])
    size = list(bbox[3:])

    spacing = mask.GetSpacing()
    full_size = mask.GetSize()

    for d in range(3):
        margin_vox = int(
            np.ceil(MARGIN_MM / spacing[d])
        )

        start = max(
            0,
            index[d] - margin_vox,
        )

        end = min(
            full_size[d],
            index[d] + size[d] + margin_vox,
        )

        index[d] = int(start)
        size[d] = int(end - start)

    cropped = sitk.RegionOfInterest(
        mask,
        size=size,
        index=index,
    )

    # Добавляем гарантированный background
    cropped = sitk.ConstantPad(
        cropped,
        [2, 2, 2],
        [2, 2, 2],
        0,
    )

    return cropped


def resample_sdf_to_isotropic(sdf):
    old_size = sdf.GetSize()
    old_spacing = sdf.GetSpacing()

    new_spacing = (
        TARGET_SPACING_MM,
        TARGET_SPACING_MM,
        TARGET_SPACING_MM,
    )

    new_size = [
        int(
            np.ceil(
                (old_size[d] - 1)
                * old_spacing[d]
                / TARGET_SPACING_MM
            )
        ) + 1
        for d in range(3)
    ]

    r = sitk.ResampleImageFilter()

    r.SetOutputSpacing(new_spacing)
    r.SetSize(new_size)
    r.SetOutputOrigin(sdf.GetOrigin())
    r.SetOutputDirection(sdf.GetDirection())

    r.SetTransform(sitk.Transform())

    # Signed distance интерполируем линейно
    r.SetInterpolator(sitk.sitkLinear)

    # Положительное расстояние = снаружи сосуда
    r.SetDefaultPixelValue(1000.0)

    return r.Execute(sdf)


def process(seg_path):
    cohort = seg_path.parent.parent.name
    case = seg_path.name.replace(".seg.nrrd", "")

    print()
    print("=" * 70)
    print(f"{cohort}/{case}")
    print("=" * 70, flush=True)

    img = sitk.ReadImage(str(seg_path))

    if img.GetDimension() != 3:
        raise RuntimeError(
            f"Expected 3D image, got dimension={img.GetDimension()}"
        )

    if img.GetNumberOfComponentsPerPixel() != 1:
        raise RuntimeError(
            "Expected scalar segmentation"
        )

    original_spacing = img.GetSpacing()

    # Любое ненулевое значение считаем vessel segmentation
    mask = sitk.Cast(
        img > 0,
        sitk.sitkUInt8,
    )

    arr_original = sitk.GetArrayViewFromImage(mask)

    voxels = int(np.count_nonzero(arr_original))

    if voxels == 0:
        raise RuntimeError("Empty segmentation")

    voxel_volume = float(np.prod(original_spacing))
    mask_volume_mm3 = voxels * voxel_volume

    print(
        "original spacing:",
        tuple(round(x, 5) for x in original_spacing),
        flush=True,
    )

    print(
        "original foreground voxels:",
        voxels,
        flush=True,
    )

    # -----------------------------
    # 1. Crop вокруг сосудов
    # -----------------------------
    mask_crop = crop_mask(mask)

    print(
        "cropped size:",
        mask_crop.GetSize(),
        flush=True,
    )

    # -----------------------------
    # 2. Signed distance в mm
    #
    # inside = negative
    # outside = positive
    # surface = 0
    # -----------------------------
    sdf = sitk.SignedMaurerDistanceMap(
        mask_crop,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )

    # -----------------------------
    # 3. 1 mm isotropic grid
    # -----------------------------
    sdf_iso = resample_sdf_to_isotropic(sdf)

    print(
        "isotropic size:",
        sdf_iso.GetSize(),
        flush=True,
    )

    print(
        "isotropic spacing:",
        sdf_iso.GetSpacing(),
        flush=True,
    )

    # -----------------------------
    # 4. Mild physical smoothing
    # -----------------------------
    sdf_iso = sitk.SmoothingRecursiveGaussian(
        sdf_iso,
        SMOOTH_SIGMA_MM,
    )

    # -----------------------------
    # 5. Zero-isosurface
    # -----------------------------
    arr = sitk.GetArrayFromImage(
        sdf_iso
    ).astype(np.float32)

    if not (
        float(arr.min()) <= 0.0 <= float(arr.max())
    ):
        raise RuntimeError(
            f"Zero level missing: "
            f"min={arr.min()} max={arr.max()}"
        )

    verts, faces, _, _ = marching_cubes(
        arr,
        level=0.0,
        allow_degenerate=False,
    )

    verts = vertices_to_physical(
        verts,
        sdf_iso,
    )

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=False,
    )

    # -----------------------------
    # 6. Minimal deterministic cleanup
    # -----------------------------
    if len(mesh.faces):
        mesh.update_faces(
            mesh.unique_faces()
        )
        mesh.remove_unreferenced_vertices()

    if len(mesh.faces):
        mesh.update_faces(
            mesh.nondegenerate_faces()
        )
        mesh.remove_unreferenced_vertices()

    try:
        trimesh.repair.fix_normals(
            mesh,
            multibody=True,
        )
    except Exception:
        pass

    # -----------------------------
    # QC
    # -----------------------------
    boundary, nonmanifold = edge_topology(
        mesh.faces
    )

    try:
        components = mesh.split(
            only_watertight=False
        )
        n_components = len(components)
    except Exception:
        n_components = -1

    finite = bool(
        np.isfinite(mesh.vertices).all()
    )

    watertight = bool(
        mesh.is_watertight
    )

    winding = bool(
        mesh.is_winding_consistent
    )

    mesh_volume_mm3 = (
        abs(float(mesh.volume))
        if watertight
        else np.nan
    )

    if (
        np.isfinite(mesh_volume_mm3)
        and mask_volume_mm3 > 0
    ):
        volume_difference_percent = (
            100.0
            * (
                mesh_volume_mm3
                - mask_volume_mm3
            )
            / mask_volume_mm3
        )
    else:
        volume_difference_percent = np.nan

    under_80k = int(
        len(mesh.vertices) <= 80000
    )

    basic_qc = int(
        finite
        and len(mesh.vertices) > 0
        and len(mesh.faces) > 0
        and boundary == 0
        and nonmanifold == 0
        and watertight
        and winding
    )

    single_component = int(
        n_components == 1
    )

    eligible = int(
        basic_qc == 1
        and single_component == 1
        and under_80k == 1
    )

    filename = (
        f"{safe_name(cohort)}_"
        f"{safe_name(case)}.ply"
    )

    output = OUT / filename
    mesh.export(output)

    row = {
        "cohort": cohort,
        "case": case,
        "source": str(seg_path),
        "mesh": str(output),

        "original_spacing_x": original_spacing[0],
        "original_spacing_y": original_spacing[1],
        "original_spacing_z": original_spacing[2],

        "target_spacing_mm": TARGET_SPACING_MM,
        "sdf_sigma_mm": SMOOTH_SIGMA_MM,

        "mask_voxels": voxels,
        "mask_volume_mm3": mask_volume_mm3,

        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(n_components),

        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,

        "watertight": watertight,
        "winding": winding,
        "finite": finite,

        "mesh_volume_mm3": mesh_volume_mm3,
        "volume_difference_percent": volume_difference_percent,

        "under_80k": under_80k,
        "single_component": single_component,
        "basic_qc": basic_qc,
        "eligible": eligible,
    }

    print()
    print(
        f"V={row['vertices']} "
        f"F={row['faces']} "
        f"components={n_components}"
    )

    print(
        f"boundary={boundary} "
        f"nonmanifold={nonmanifold} "
        f"watertight={watertight} "
        f"winding={winding}"
    )

    print(
        f"volume difference = "
        f"{volume_difference_percent:.2f}%"
    )

    print(
        f"<=80k={under_80k} "
        f"eligible={eligible}"
    )

    print(
        "saved:",
        output,
        flush=True,
    )

    del img
    del mask
    del mask_crop
    del sdf
    del sdf_iso
    del arr
    del mesh

    gc.collect()

    return row


def main():
    segs = sorted(
        ROOT.rglob("*.seg.nrrd")
    )

    print(
        f"Found {len(segs)} AVT segmentations"
    )

    if len(segs) != 56:
        print(
            f"WARNING: expected 56, "
            f"found {len(segs)}"
        )

    rows = []

    for i, seg in enumerate(segs, 1):

        print()
        print(
            f"[{i}/{len(segs)}] {seg}",
            flush=True,
        )

        try:
            row = process(seg)
            row["error"] = ""

        except Exception as e:

            print(
                f"ERROR: {repr(e)}",
                flush=True,
            )

            row = {
                "cohort": seg.parent.parent.name,
                "case": seg.name.replace(
                    ".seg.nrrd",
                    "",
                ),
                "source": str(seg),
                "mesh": "",

                "original_spacing_x": "",
                "original_spacing_y": "",
                "original_spacing_z": "",

                "target_spacing_mm":
                    TARGET_SPACING_MM,

                "sdf_sigma_mm":
                    SMOOTH_SIGMA_MM,

                "mask_voxels": "",
                "mask_volume_mm3": "",

                "vertices": "",
                "faces": "",
                "components": "",

                "boundary_edges": "",
                "nonmanifold_edges": "",

                "watertight": "",
                "winding": "",
                "finite": "",

                "mesh_volume_mm3": "",
                "volume_difference_percent": "",

                "under_80k": 0,
                "single_component": 0,
                "basic_qc": 0,
                "eligible": 0,

                "error": repr(e),
            }

        rows.append(row)

        # Перезаписываем CSV после каждого пациента,
        # чтобы результат не потерялся при прерывании.
        fields = list(rows[0].keys())

        with CSV.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(rows)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print("Processed:", len(rows))
    print("Meshes:", OUT)
    print("QC:", CSV)


if __name__ == "__main__":
    main()
