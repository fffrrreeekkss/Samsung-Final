from pathlib import Path
import numpy as np
import SimpleITK as sitk
import trimesh
from skimage.measure import marching_cubes


SRC = Path("data/AVT/raw/KiTS/K19/K19.seg.nrrd")
OUT = Path("data/AVT/K19_linear_1mm.ply")

TARGET_SPACING = 1.0
MARGIN_MM = 12.0


def edge_topology(faces):
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])

    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)

    return (
        int(np.sum(counts == 1)),
        int(np.sum(counts > 2)),
    )


def to_physical(vertices_zyx, img):
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


img = sitk.ReadImage(str(SRC))
mask = sitk.Cast(img > 0, sitk.sitkUInt8)

# Original segmentation volume
voxels = int(
    np.count_nonzero(
        sitk.GetArrayViewFromImage(mask)
    )
)

mask_volume = (
    voxels * np.prod(mask.GetSpacing())
)

# ------------------------
# Crop around segmentation
# ------------------------
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(mask)

bbox = stats.GetBoundingBox(1)

index = list(bbox[:3])
size = list(bbox[3:])

spacing = mask.GetSpacing()
full_size = mask.GetSize()

for d in range(3):
    margin = int(
        np.ceil(MARGIN_MM / spacing[d])
    )

    start = max(
        0,
        index[d] - margin,
    )

    end = min(
        full_size[d],
        index[d] + size[d] + margin,
    )

    index[d] = int(start)
    size[d] = int(end - start)

mask = sitk.RegionOfInterest(
    mask,
    size=size,
    index=index,
)

# Explicit background around the object
mask = sitk.ConstantPad(
    mask,
    [2, 2, 2],
    [2, 2, 2],
    0,
)

print("Original spacing:", img.GetSpacing())
print("Crop size:", mask.GetSize())

# ------------------------
# Resample BINARY FIELD
# ------------------------
field = sitk.Cast(
    mask,
    sitk.sitkFloat32,
)

old_size = field.GetSize()
old_spacing = field.GetSpacing()

new_spacing = (
    TARGET_SPACING,
    TARGET_SPACING,
    TARGET_SPACING,
)

new_size = [
    int(
        np.ceil(
            (old_size[d] - 1)
            * old_spacing[d]
            / TARGET_SPACING
        )
    ) + 1
    for d in range(3)
]

r = sitk.ResampleImageFilter()

r.SetOutputSpacing(new_spacing)
r.SetSize(new_size)
r.SetOutputOrigin(field.GetOrigin())
r.SetOutputDirection(field.GetDirection())

r.SetTransform(sitk.Transform())

# Key difference:
# directly interpolate 0/1 segmentation
r.SetInterpolator(sitk.sitkLinear)
r.SetDefaultPixelValue(0.0)

field_iso = r.Execute(field)

print("New size:", field_iso.GetSize())
print("New spacing:", field_iso.GetSpacing())

# ------------------------
# 0.5 isosurface
# ------------------------
arr = sitk.GetArrayFromImage(
    field_iso
).astype(np.float32)

print(
    "Interpolated range:",
    float(arr.min()),
    float(arr.max()),
)

verts, faces, _, _ = marching_cubes(
    arr,
    level=0.5,
    allow_degenerate=False,
)

verts = to_physical(
    verts,
    field_iso,
)

mesh = trimesh.Trimesh(
    vertices=verts,
    faces=faces,
    process=False,
)

mesh.update_faces(mesh.unique_faces())
mesh.remove_unreferenced_vertices()

mesh.update_faces(mesh.nondegenerate_faces())
mesh.remove_unreferenced_vertices()

try:
    trimesh.repair.fix_normals(
        mesh,
        multibody=True,
    )
except Exception:
    pass

boundary, nonmanifold = edge_topology(
    mesh.faces
)

components = len(
    mesh.split(
        only_watertight=False
    )
)

mesh_volume = (
    abs(float(mesh.volume))
    if mesh.is_watertight
    else np.nan
)

volume_diff = (
    100.0
    * (mesh_volume - mask_volume)
    / mask_volume
)

mesh.export(OUT)

print()
print("================ RESULT ================")
print("OUTPUT:", OUT)
print("vertices:", len(mesh.vertices))
print("faces:", len(mesh.faces))
print("components:", components)
print("boundary edges:", boundary)
print("nonmanifold edges:", nonmanifold)
print("watertight:", mesh.is_watertight)
print("winding:", mesh.is_winding_consistent)
print("mask volume mm3:", mask_volume)
print("mesh volume mm3:", mesh_volume)
print("volume difference %:", volume_diff)
