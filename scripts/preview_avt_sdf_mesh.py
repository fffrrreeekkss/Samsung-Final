from pathlib import Path
import numpy as np
import SimpleITK as sitk
import trimesh
from skimage.measure import marching_cubes


SRC = Path("data/AVT/raw/KiTS/K19/K19.seg.nrrd")
OUT = Path("data/AVT/K19_sdf_1mm.ply")

TARGET_SPACING = 1.0
SMOOTH_SIGMA_MM = 0.5
MARGIN_MM = 12.0


def to_physical(vertices_zyx, img):
    spacing = np.asarray(img.GetSpacing(), dtype=float)
    origin = np.asarray(img.GetOrigin(), dtype=float)
    direction = np.asarray(img.GetDirection(), dtype=float).reshape(3, 3)

    xyz = vertices_zyx[:, [2, 1, 0]]
    return origin[None, :] + (xyz * spacing[None, :]) @ direction.T


img = sitk.ReadImage(str(SRC))

# Binary GT segmentation
mask = sitk.Cast(img > 0, sitk.sitkUInt8)

# ---------- crop around vessel ----------
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(mask)

if not stats.HasLabel(1):
    raise RuntimeError("Empty segmentation")

bbox = stats.GetBoundingBox(1)
# bbox = x,y,z,sizeX,sizeY,sizeZ

index = list(bbox[:3])
size = list(bbox[3:])
spacing = img.GetSpacing()
full_size = img.GetSize()

for d in range(3):
    margin_vox = int(np.ceil(MARGIN_MM / spacing[d]))

    start = max(0, index[d] - margin_vox)
    end = min(full_size[d], index[d] + size[d] + margin_vox)

    index[d] = start
    size[d] = end - start

mask = sitk.RegionOfInterest(
    mask,
    size=[int(x) for x in size],
    index=[int(x) for x in index],
)

# Ensure background outside segmentation
mask = sitk.ConstantPad(
    mask,
    [2, 2, 2],
    [2, 2, 2],
    0,
)

print("Original spacing:", img.GetSpacing())
print("Cropped size:", mask.GetSize())
print("Cropped spacing:", mask.GetSpacing())

# ---------- signed distance in millimetres ----------
sdf = sitk.SignedMaurerDistanceMap(
    mask,
    insideIsPositive=False,
    squaredDistance=False,
    useImageSpacing=True,
)

# ---------- resample signed distance to 1 mm isotropic ----------
old_size = sdf.GetSize()
old_spacing = sdf.GetSpacing()

new_spacing = (TARGET_SPACING,) * 3

new_size = [
    int(np.ceil(
        (old_size[d] - 1) * old_spacing[d] / TARGET_SPACING
    )) + 1
    for d in range(3)
]

resampler = sitk.ResampleImageFilter()
resampler.SetOutputSpacing(new_spacing)
resampler.SetSize(new_size)
resampler.SetOutputOrigin(sdf.GetOrigin())
resampler.SetOutputDirection(sdf.GetDirection())
resampler.SetTransform(sitk.Transform())
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(1000.0)

sdf_iso = resampler.Execute(sdf)

# Mild smoothing in physical units.
# Applied to the distance field, not directly to mesh vertices.
sdf_iso = sitk.SmoothingRecursiveGaussian(
    sdf_iso,
    SMOOTH_SIGMA_MM,
)

print("Isotropic size:", sdf_iso.GetSize())
print("Isotropic spacing:", sdf_iso.GetSpacing())

# ---------- zero level surface ----------
arr = sitk.GetArrayFromImage(sdf_iso).astype(np.float32)

verts, faces, _, _ = marching_cubes(
    arr,
    level=0.0,
    allow_degenerate=False,
)

verts = to_physical(verts, sdf_iso)

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
    trimesh.repair.fix_normals(mesh, multibody=True)
except Exception:
    pass

OUT.parent.mkdir(parents=True, exist_ok=True)
mesh.export(OUT)

print()
print("OUTPUT:", OUT)
print("vertices:", len(mesh.vertices))
print("faces:", len(mesh.faces))
print("watertight:", mesh.is_watertight)
print("winding:", mesh.is_winding_consistent)

components = mesh.split(only_watertight=False)
print("components:", len(components))
