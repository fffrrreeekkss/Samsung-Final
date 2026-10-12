from pathlib import Path
import pandas as pd
import numpy as np
import trimesh


ROOT = Path("data/RBWH_STL/derivatives")
OUT = Path("data/RBWH_STL/rbwh_qc.csv")


def subject_from_path(p):
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    return "unknown"


def session_from_path(p):
    for part in p.parts:
        if part.startswith("ses-"):
            return part
    return "unknown"


def edge_topology(faces):
    faces = np.asarray(faces, dtype=np.int64)

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


files = []

for p in ROOT.rglob("*.stl"):
    name = p.name.lower()

    if (
        "aneurysm" in name
        and "parent" in name
        and "artery" in name
    ):
        files.append(p)


print("Candidate STL files:", len(files))

rows = []

for i, p in enumerate(sorted(files), 1):

    print(
        f"[{i}/{len(files)}] {p}",
        flush=True,
    )

    try:
        # IMPORTANT:
        # STL duplicates vertices triangle-by-triangle.
        # process=True welds identical vertices.
        m = trimesh.load(
            p,
            process=True,
        )

        if isinstance(m, trimesh.Scene):
            if not m.geometry:
                raise RuntimeError("Empty scene")

            m = trimesh.util.concatenate(
                tuple(m.geometry.values())
            )

        # Explicit weld/cleanup
        m.merge_vertices()
        m.remove_unreferenced_vertices()

        if len(m.faces):
            m.update_faces(
                m.unique_faces()
            )
            m.remove_unreferenced_vertices()

        if len(m.faces):
            m.update_faces(
                m.nondegenerate_faces()
            )
            m.remove_unreferenced_vertices()

        boundary, nonmanifold = edge_topology(
            m.faces
        )

        components = len(
            m.split(
                only_watertight=False
            )
        )

        finite = bool(
            np.isfinite(m.vertices).all()
        )

        extent = (
            m.bounds[1]
            - m.bounds[0]
        )

        rows.append({
            "subject": subject_from_path(p),
            "session": session_from_path(p),
            "file": str(p),
            "name": p.name,

            "vertices": int(len(m.vertices)),
            "faces": int(len(m.faces)),

            "components": int(components),

            "boundary_edges": int(boundary),
            "nonmanifold_edges": int(nonmanifold),

            "watertight": bool(m.is_watertight),
            "winding": bool(
                m.is_winding_consistent
            ),
            "finite": finite,

            "extent_x": float(extent[0]),
            "extent_y": float(extent[1]),
            "extent_z": float(extent[2]),

            "error": "",
        })

    except Exception as e:

        rows.append({
            "subject": subject_from_path(p),
            "session": session_from_path(p),
            "file": str(p),
            "name": p.name,

            "vertices": np.nan,
            "faces": np.nan,
            "components": np.nan,

            "boundary_edges": np.nan,
            "nonmanifold_edges": np.nan,

            "watertight": False,
            "winding": False,
            "finite": False,

            "extent_x": np.nan,
            "extent_y": np.nan,
            "extent_z": np.nan,

            "error": repr(e),
        })


df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)


print()
print("================ SUMMARY ================")
print("STL candidates:", len(df))
print(
    "Unique subjects:",
    df["subject"].nunique()
)
print(
    "Unique subject/session:",
    len(
        df[
            ["subject", "session"]
        ].drop_duplicates()
    )
)
print(
    "Finite:",
    int(df["finite"].sum())
)
print(
    "Single component:",
    int((df["components"] == 1).sum())
)
print(
    "Nonmanifold=0:",
    int(
        (df["nonmanifold_edges"] == 0).sum()
    )
)
print(
    "Watertight:",
    int(df["watertight"].sum())
)

print()
print("Vertex range AFTER WELD:")
print(df["vertices"].describe())

print()
print("================ FILES ================")

print(
    df[
        [
            "subject",
            "session",
            "vertices",
            "faces",
            "components",
            "boundary_edges",
            "nonmanifold_edges",
            "watertight",
            "winding",
            "name",
        ]
    ]
    .sort_values(
        ["subject", "session"]
    )
    .to_string(index=False)
)

print()
print("Saved:", OUT)
