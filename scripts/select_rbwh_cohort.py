from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import trimesh


CSV = Path("data/RBWH_STL/rbwh_qc.csv")
OUT = Path("data/RBWH_STL/reference")
MANIFEST = Path("data/RBWH_STL/selected_cases.csv")

SEED = 2027
N = 20

MIN_VERTICES = 3000
MAX_VERTICES = 20000

OUT.mkdir(parents=True, exist_ok=True)


def true_subject(name):
    m = re.search(r"(sub-\d+)", str(name))
    if not m:
        raise RuntimeError(f"Cannot parse subject from: {name}")
    return m.group(1)


df = pd.read_csv(CSV)

# Use ID encoded in the actual STL filename.
# This avoids odd nested directory cases in the public dataset.
df["true_subject"] = df["name"].apply(true_subject)

eligible = df[
    (df["finite"] == True) &
    (df["watertight"] == True) &
    (df["winding"] == True) &
    (df["components"] == 1) &
    (df["boundary_edges"] == 0) &
    (df["nonmanifold_edges"] == 0) &
    (df["vertices"] >= MIN_VERTICES) &
    (df["vertices"] <= MAX_VERTICES)
].copy()

print("Eligible STL files:", len(eligible))
print("Eligible unique subjects:", eligible["true_subject"].nunique())

if eligible["true_subject"].nunique() < N:
    raise RuntimeError("Not enough eligible independent subjects")


# -------------------------------------------------
# One mesh per subject
# If subject has several aneurysms, choose one
# deterministically using the fixed random seed.
# -------------------------------------------------
rng = np.random.default_rng(SEED)

chosen_per_subject = []

for subject, group in eligible.groupby("true_subject"):
    group = group.sort_values("name").reset_index(drop=True)

    idx = int(rng.integers(0, len(group)))

    chosen_per_subject.append(
        group.iloc[idx]
    )

pool = pd.DataFrame(chosen_per_subject)
pool = pool.sort_values("vertices").reset_index(drop=True)


# -------------------------------------------------
# Stratify by mesh complexity.
# Four size strata, five subjects from each.
# -------------------------------------------------
pool["size_stratum"] = pd.qcut(
    pool["vertices"],
    q=4,
    labels=["Q1", "Q2", "Q3", "Q4"],
    duplicates="drop",
)

selected_parts = []

for stratum, group in pool.groupby(
    "size_stratum",
    observed=True
):
    group = group.copy()

    if len(group) < 5:
        raise RuntimeError(
            f"Not enough cases in {stratum}: {len(group)}"
        )

    indices = rng.choice(
        len(group),
        size=5,
        replace=False,
    )

    x = group.iloc[indices].copy()
    selected_parts.append(x)

selected = pd.concat(
    selected_parts,
    ignore_index=True,
)

selected = selected.sort_values(
    "vertices"
).reset_index(drop=True)

if len(selected) != N:
    raise RuntimeError(
        f"Expected {N}, selected {len(selected)}"
    )


# -------------------------------------------------
# Export welded meshes as PLY
# -------------------------------------------------
rows = []

for i, r in selected.iterrows():
    patient = i + 1

    src = Path(r["file"])

    m = trimesh.load(
        src,
        process=True,
    )

    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(
            tuple(m.geometry.values())
        )

    m.merge_vertices()
    m.remove_unreferenced_vertices()

    m.update_faces(
        m.unique_faces()
    )
    m.remove_unreferenced_vertices()

    m.update_faces(
        m.nondegenerate_faces()
    )
    m.remove_unreferenced_vertices()

    dst = OUT / f"{patient}.ply"
    m.export(dst)

    rows.append({
        "patient_id": patient,
        "subject": r["true_subject"],
        "session": r["session"],
        "original_name": r["name"],
        "source_stl": str(src),
        "reference_ply": str(dst),
        "vertices": len(m.vertices),
        "faces": len(m.faces),
        "size_stratum": str(r["size_stratum"]),
    })

manifest = pd.DataFrame(rows)
manifest.to_csv(
    MANIFEST,
    index=False,
)

print()
print("============= FINAL COHORT =============")
print(
    manifest[
        [
            "patient_id",
            "subject",
            "session",
            "vertices",
            "faces",
            "size_stratum",
        ]
    ].to_string(index=False)
)

print()
print("Patients:", len(manifest))
print("Unique subjects:", manifest["subject"].nunique())
print("Min vertices:", manifest["vertices"].min())
print("Median vertices:", manifest["vertices"].median())
print("Max vertices:", manifest["vertices"].max())

print()
print("References:", OUT)
print("Manifest:", MANIFEST)
