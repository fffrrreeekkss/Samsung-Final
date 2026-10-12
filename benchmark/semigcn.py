"""Full-surface SeMIGCN: working v2 MGCN path, temporary upstream artifacts only."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import random
import sys
import os

import numpy as np
import trimesh
import pymeshlab
from scipy import sparse

from .mesh_utils import ROOT, save, fill
from .metrics import distances


def sparse_compatibility(Mesh, meshnet, torch):
    """Equivalent uniform graph operators without quadratic memory allocation."""

    def build_v2v(self):
        n = len(self.vs)
        edges = self.edges

        adj = sparse.coo_matrix(
            (
                np.ones(2 * len(edges)),
                (
                    np.r_[edges[:, 0], edges[:, 1]],
                    np.r_[edges[:, 1], edges[:, 0]],
                ),
            ),
            shape=(n, n),
        ).tocsr()

        self.v2v = [
            adj.indices[adj.indptr[i]:adj.indptr[i + 1]].tolist()
            for i in range(n)
        ]

        degree = np.asarray(adj.sum(axis=1)).ravel()

        if np.any(degree == 0):
            raise ValueError("Isolated model vertex")

        def tensor(a):
            a = a.tocoo()
            return torch.sparse_coo_tensor(
                torch.tensor(
                    np.vstack((a.row, a.col)),
                    dtype=torch.long,
                ),
                torch.tensor(
                    a.data,
                    dtype=torch.float32,
                ),
                a.shape,
            ).coalesce()

        self.Adj = tensor(adj)
        self.v_dims = torch.tensor(
            degree,
            dtype=torch.float32,
        )
        self.Diag = tensor(
            sparse.diags(1 / degree)
        )
        self.AdjI = tensor(
            adj + sparse.eye(n)
        )
        self.Lap = tensor(
            sparse.eye(n)
            - sparse.diags(1 / degree) @ adj
        )

    Mesh.build_v2v = build_v2v

    def pool_forward(self, value):
        matrix = self.pool_hash.coalesce()

        denom = (
            torch.sparse.sum(matrix, dim=1)
            .to_dense()
            .reshape(-1, 1)
        )

        return torch.sparse.mm(matrix, value) / denom

    meshnet.MeshPool.forward = pool_forward


def _edge_topology(mesh):
    """
    Return:
        boundary_edges,
        nonmanifold_edges
    """

    faces = np.asarray(mesh.faces, dtype=np.int64)

    if len(faces) == 0:
        return 0, 0

    edges = np.vstack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        )
    )

    edges = np.sort(edges, axis=1)

    _, counts = np.unique(
        edges,
        axis=0,
        return_counts=True,
    )

    boundary = int(
        np.sum(counts == 1)
    )

    nonmanifold = int(
        np.sum(counts > 2)
    )

    return boundary, nonmanifold


def _mesh_from_meshlab(ms):
    """Convert current PyMeshLab mesh to trimesh without processing."""

    m = ms.current_mesh()

    vertices = np.asarray(
        m.vertex_matrix(),
        dtype=np.float64,
    )

    faces = np.asarray(
        m.face_matrix(),
        dtype=np.int64,
    )

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )



def _pymeshlab_percentage(value):
    """Compatibility wrapper for old/new PyMeshLab percentage parameters."""
    if hasattr(pymeshlab, "PercentageValue"):
        return pymeshlab.PercentageValue(float(value))
    if hasattr(pymeshlab, "Percentage"):
        return pymeshlab.Percentage(float(value))
    return float(value)


def _isotropic_remesh_initial(initial, percent=0.6):
    """
    Apply the isotropic remeshing stage used by the original SeMIGCN
    preprocessing pipeline.

    `percent` is the MeshLab target edge length expressed as a percentage
    of the model bounding-box diagonal (original SeMIGCN default: 0.6).
    """
    initial = initial.copy()

    before_v = len(initial.vertices)
    before_f = len(initial.faces)
    before_edges = np.asarray(initial.edges_unique_length, dtype=np.float64)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(initial.vertices, dtype=np.float64),
            face_matrix=np.asarray(initial.faces, dtype=np.int32),
        ),
        "semigcn_initial_before_remesh",
    )

    target = _pymeshlab_percentage(percent)

    done = False
    last_error = None

    # Current PyMeshLab API.
    if hasattr(ms, "meshing_isotropic_explicit_remeshing"):
        try:
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=target
            )
            done = True
        except Exception as exc:
            last_error = exc

    # Compatibility with older PyMeshLab filter names used by the
    # official SeMIGCN repository.
    if not done:
        for filter_name in (
            "meshing_isotropic_explicit_remeshing",
            "remeshing_isotropic_explicit_remeshing",
        ):
            try:
                ms.apply_filter(
                    filter_name,
                    targetlen=target,
                )
                done = True
                break
            except Exception as exc:
                last_error = exc

    if not done:
        raise RuntimeError(
            "PyMeshLab isotropic remeshing failed"
        ) from last_error

    remeshed = _mesh_from_meshlab(ms)

    if len(remeshed.vertices) == 0 or len(remeshed.faces) == 0:
        raise ValueError(
            "Isotropic remeshing produced an empty mesh"
        )

    remeshed.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(
        remeshed,
        multibody=True,
    )

    boundary, nonmanifold = _edge_topology(remeshed)

    after_edges = np.asarray(
        remeshed.edges_unique_length,
        dtype=np.float64,
    )

    def cv(x):
        if len(x) == 0:
            return float("nan")
        mean = float(np.mean(x))
        if mean <= 0:
            return float("nan")
        return float(np.std(x) / mean)

    print(
        "[SeMIGCN REMESH]"
        f" percent={float(percent):.3f}"
        f" V={before_v}->{len(remeshed.vertices)}"
        f" F={before_f}->{len(remeshed.faces)}"
        f" edge_cv={cv(before_edges):.4f}->{cv(after_edges):.4f}"
        f" boundary={boundary}"
        f" nonmanifold={nonmanifold}"
        f" watertight={remeshed.is_watertight}"
        f" winding={remeshed.is_winding_consistent}",
        flush=True,
    )

    # Do not abort here. PyMeshLab isotropic remeshing can introduce a
    # handful of duplicate/non-manifold edges even when its input is valid.
    # The caller immediately passes this mesh through _repair_initial(),
    # which is the dedicated topology/orientation repair stage.
    if (
        boundary != 0
        or nonmanifold != 0
        or not remeshed.is_watertight
        or not remeshed.is_winding_consistent
    ):
        print(
            "[SeMIGCN REMESH] topology cleanup required; "
            "passing remeshed surface to _repair_initial()",
            flush=True,
        )

    return remeshed

def _repair_initial(initial):
    """
    Make the SeMIGCN full-surface initializer closed and consistently
    oriented.

    Important:
    - the common case (closed mesh with inconsistent winding) changes only
      face orientation;
    - geometry/topology fallback is used only if the initializer is actually
      non-watertight/non-manifold;
    - GT and damaged input are not modified here.
    """

    initial = initial.copy()

    boundary, nonmanifold = _edge_topology(initial)

    print(
        "[SeMIGCN INIT] before repair:"
        f" V={len(initial.vertices)}"
        f" F={len(initial.faces)}"
        f" boundary={boundary}"
        f" nonmanifold={nonmanifold}"
        f" watertight={initial.is_watertight}"
        f" winding={initial.is_winding_consistent}",
        flush=True,
    )

    # ---------------------------------------------------------
    # 1. First fix ONLY face orientation.
    #
    # This is the expected fix for patient 7:
    # the surface can already be closed/manifold but have inconsistent
    # winding after hole filling.
    # ---------------------------------------------------------

    trimesh.repair.fix_normals(
        initial,
        multibody=True,
    )

    boundary, nonmanifold = _edge_topology(initial)

    if (
        boundary == 0
        and nonmanifold == 0
        and initial.is_watertight
        and initial.is_winding_consistent
    ):
        print(
            "[SeMIGCN INIT] orientation repaired successfully:"
            f" boundary={boundary}"
            f" nonmanifold={nonmanifold}"
            f" watertight={initial.is_watertight}"
            f" winding={initial.is_winding_consistent}",
            flush=True,
        )

        return initial

    # ---------------------------------------------------------
    # 2. Topology fallback.
    #
    # Only reached if fix_normals() was not enough.
    # ---------------------------------------------------------

    print(
        "[SeMIGCN INIT] orientation-only repair was insufficient; "
        "running topology fallback",
        flush=True,
    )

    ms = pymeshlab.MeshSet()

    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(
                initial.vertices,
                dtype=np.float64,
            ),
            face_matrix=np.asarray(
                initial.faces,
                dtype=np.int32,
            ),
        ),
        "semigcn_initial",
    )

    # ---------------------------------------------------------
    # 2a. Resolve non-manifold edges as conservatively as possible.
    # ---------------------------------------------------------

    for _ in range(5):
        current = _mesh_from_meshlab(ms)

        _, nm = _edge_topology(current)

        if nm == 0:
            break

        ms.meshing_repair_non_manifold_edges(
            method="Split Vertices"
        )

        ms.meshing_repair_non_manifold_vertices(
            vertdispratio=0
        )

        ms.meshing_remove_unreferenced_vertices()

    current = _mesh_from_meshlab(ms)

    _, nm = _edge_topology(current)

    # ---------------------------------------------------------
    # 2b. Hard fallback for residual non-manifold edges.
    # ---------------------------------------------------------

    if nm > 0:
        print(
            f"[SeMIGCN INIT] residual non-manifold edges after "
            f"Split Vertices: {nm}; using Remove Faces",
            flush=True,
        )

        ms.meshing_repair_non_manifold_edges(
            method="Remove Faces"
        )

        ms.meshing_repair_non_manifold_vertices(
            vertdispratio=0
        )

        ms.meshing_remove_unreferenced_vertices()

    current = _mesh_from_meshlab(ms)

    boundary, nonmanifold = _edge_topology(current)

    if nonmanifold != 0:
        raise ValueError(
            "SeMIGCN initialization topology repair failed: "
            f"{nonmanifold} non-manifold edges remain"
        )

    # ---------------------------------------------------------
    # 2c. Close any holes created by the topology repair.
    # ---------------------------------------------------------

    if boundary > 0 or not current.is_watertight:
        print(
            "[SeMIGCN INIT] closing residual holes:"
            f" boundary={boundary}",
            flush=True,
        )

        ms.meshing_close_holes(
            maxholesize=1000,
            selfintersection=False,
            refinehole=False,
        )

        ms.meshing_remove_unreferenced_vertices()

    repaired = _mesh_from_meshlab(ms)

    # ---------------------------------------------------------
    # 3. Final orientation normalization.
    # ---------------------------------------------------------

    trimesh.repair.fix_normals(
        repaired,
        multibody=True,
    )

    boundary, nonmanifold = _edge_topology(repaired)

    print(
        "[SeMIGCN INIT] after repair:"
        f" V={len(repaired.vertices)}"
        f" F={len(repaired.faces)}"
        f" boundary={boundary}"
        f" nonmanifold={nonmanifold}"
        f" watertight={repaired.is_watertight}"
        f" winding={repaired.is_winding_consistent}",
        flush=True,
    )

    # ---------------------------------------------------------
    # 4. Strict validation.
    # ---------------------------------------------------------

    if boundary != 0:
        raise ValueError(
            "SeMIGCN initialization still has "
            f"{boundary} boundary edges after repair"
        )

    if nonmanifold != 0:
        raise ValueError(
            "SeMIGCN initialization still has "
            f"{nonmanifold} non-manifold edges after repair"
        )

    if not repaired.is_watertight:
        raise ValueError(
            "SeMIGCN initialization is still not watertight "
            "after repair"
        )

    if not repaired.is_winding_consistent:
        raise ValueError(
            "SeMIGCN initialization still has inconsistent "
            "face winding after repair"
        )

    return repaired


def repair(damaged, cfg, seed):
    with TemporaryDirectory(
        prefix="semigcn_"
    ) as directory:
        return _train(
            damaged,
            cfg,
            seed,
            Path(directory),
        )


def _train(org, cfg, seed, work):
    import torch

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        cfg.get("device", "cpu")
    )

    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            "SeMIGCN device must be cpu or cuda"
        )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA explicitly requested but unavailable"
            )

        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )

    # ---------------------------------------------------------
    # SeMIGCN imports
    # ---------------------------------------------------------

    sys.path.insert(
        0,
        str(ROOT / "semigcn"),
    )

    from util.mesh import Mesh
    from util import (
        meshnet,
        datamaker,
        loss,
        models,
    )

    sparse_compatibility(
        Mesh,
        meshnet,
        torch,
    )

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.set_num_threads(
        cfg["threads"]
    )

    torch.use_deterministic_algorithms(
        True
    )

    torch.backends.cudnn.benchmark = False

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ---------------------------------------------------------
    # Full-surface initialization
    # ---------------------------------------------------------

    initial = fill(
        org,
        refine=True,
    )

    # First make the hole-filled initializer topologically valid.
    initial = _repair_initial(initial)

    # The official SeMIGCN preprocessing performs isotropic remeshing
    # before normalization/training. This is important because hole-filling
    # can create highly non-uniform triangles that destabilize MGCN and
    # collapse the inpainted patch.
    initial = _isotropic_remesh_initial(
        initial,
        percent=float(
            cfg.get(
                "remesh_percent",
                0.6,
            )
        ),
    )

    # Re-check orientation/topology after remeshing.
    initial = _repair_initial(initial)

    if len(initial.vertices) > cfg["max_vertices"]:
        raise ValueError(
            "Model vertex resource limit exceeded after remeshing"
        )

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------

    center = initial.bounds.mean(0)

    scale = float(
        initial.edges_unique_length.mean()
    )

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Invalid SeMIGCN normalization scale"
        )

    normalized = initial.copy()

    normalized.vertices = (
        initial.vertices - center
    ) / scale

    original = org.copy()

    original.vertices = (
        org.vertices - center
    ) / scale

    # ---------------------------------------------------------
    # Known / unknown mask
    # ---------------------------------------------------------

    known = (
        distances(
            normalized.vertices,
            original,
        )
        < cfg["mask_distance_edge_units"]
    )

    if known.all() or not known.any():
        raise ValueError(
            "Known/unknown mask invalid"
        )

    known_count = int(np.sum(known))
    unknown_count = int(len(known) - known_count)

    print(
        "[SeMIGCN MASK]"
        f" known={known_count}"
        f" unknown={unknown_count}"
        f" unknown_pct={100.0 * unknown_count / len(known):.2f}%",
        flush=True,
    )

    # ---------------------------------------------------------
    # Smoothed mesh
    # ---------------------------------------------------------

    smooth = normalized.copy()

    e = smooth.edges_unique
    n = len(smooth.vertices)

    a = sparse.coo_matrix(
        (
            np.ones(2 * len(e)),
            (
                np.r_[e[:, 0], e[:, 1]],
                np.r_[e[:, 1], e[:, 0]],
            ),
        ),
        shape=(n, n),
    ).tocsr()

    degree = np.asarray(
        a.sum(1)
    ).ravel()

    if np.any(degree == 0):
        raise ValueError(
            "Isolated vertex before SeMIGCN smoothing"
        )

    operator = (
        sparse.diags(
            1 / degree
        )
        @ a
    )

    for _ in range(
        cfg["smooth_iterations"]
    ):
        smooth.vertices = (
            operator
            @ smooth.vertices
        )

    # ---------------------------------------------------------
    # Temporary upstream files
    # ---------------------------------------------------------

    name = work.name

    for mesh, suffix in [
        (normalized, "initial"),
        (original, "original"),
        (smooth, "smooth"),
    ]:
        save(
            mesh,
            work / f"{name}_{suffix}.obj",
        )

    (
        work / f"{name}_vmask.json"
    ).write_text(
        json.dumps(
            known.tolist()
        )
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    mesh_dic, dataset = (
        datamaker.create_dataset(
            str(work),
            dm_size=cfg["dummy_masks"],
            kn=cfg["neighbourhoods"],
            cache=False,
        )
    )

    print(
        "SEMIGCN DEVICE\n"
        f"device = {device}\n"
        f"gpu = "
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'none'}",
        flush=True,
    )

    # ---------------------------------------------------------
    # Model
    #
    # Mesh simplification and construction remain CPU operations.
    # Move the finished model and stored tensors once afterwards.
    # ---------------------------------------------------------

    model = meshnet.MGCN(
        torch.device("cpu"),
        mesh_dic["smo_mesh"],
        mesh_dic["ini_mesh"],
        mesh_dic["v_mask"],
        skip=False,
    ).to(device)

    model.device = device

    for module in model.modules():

        for tensor_name in (
            "edge_index1",
            "edge_index2",
        ):
            if hasattr(
                module,
                tensor_name,
            ):
                setattr(
                    module,
                    tensor_name,
                    getattr(
                        module,
                        tensor_name,
                    ).to(device),
                )

        if (
            device.type == "cuda"
            and isinstance(
                module,
                meshnet.ChebConv,
            )
        ):
            # Memory-saving CUDA path only.
            #
            # CPU retains historical/default decomposed_layers=1.
            module.decomposed_layers = 4

            from functools import partial
            from torch.utils.checkpoint import checkpoint

            original_forward = module.forward

            module.forward = partial(
                checkpoint,
                original_forward,
                use_reentrant=False,
            )

    # ---------------------------------------------------------
    # Move model-held tensor lists
    # ---------------------------------------------------------

    for tensor_list_name in (
        "poss_list",
        "smposs_list",
        "v_masks_list",
        "f_masks_list",
        "edge_inds",
        "p_hashes",
        "up_hashes",
    ):
        setattr(
            model,
            tensor_list_name,
            [
                value.to(device)
                for value in getattr(
                    model,
                    tensor_list_name,
                )
            ],
        )

    for tensor_name in (
        "poss",
        "smposs",
        "v_masks",
        "f_masks",
    ):
        setattr(
            model,
            tensor_name,
            getattr(
                model,
                tensor_name,
            ).to(device),
        )

    # ---------------------------------------------------------
    # Dataset tensors
    # ---------------------------------------------------------

    for attr_name, value in vars(
        dataset
    ).items():

        if torch.is_tensor(value):
            setattr(
                dataset,
                attr_name,
                value.detach()
                .to(device)
                .requires_grad_(
                    value.requires_grad
                ),
            )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
    )

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["scheduler_step"],
            gamma=cfg["scheduler_gamma"],
        )
    )

    # ---------------------------------------------------------
    # Training tensors
    # ---------------------------------------------------------

    masks = mesh_dic[
        "vmask_dummy"
    ].to(device)

    vmask = mesh_dic[
        "v_mask"
    ].to(device)

    fmask = mesh_dic[
        "f_mask"
    ].to(device)

    weights = cfg[
        "multi_scale_weights"
    ]

    faces = torch.as_tensor(
        mesh_dic["ini_mesh"].faces,
        dtype=torch.long,
        device=device,
    )

    normals = torch.as_tensor(
        mesh_dic["ini_mesh"].fn,
        device=device,
    )

    if (
        masks.shape[1]
        % cfg["batch_size"]
    ):
        raise ValueError(
            "Mask count must divide batch size"
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(
        cfg["epochs"]
    ):

        model.train()

        epoch_loss = 0.0

        permutation = torch.randperm(
            masks.shape[1]
        )

        batches = permutation.reshape(
            -1,
            cfg["batch_size"],
        )

        for batch in batches:

            optimizer.zero_grad()

            for idx in batch:

                mask = (
                    masks[
                        :,
                        int(idx),
                    ].reshape(-1, 1)
                    * vmask.reshape(-1, 1)
                )

                positions = model(
                    dataset,
                    mask,
                )

                norm = (
                    models.compute_fn(
                        positions[0],
                        faces,
                    )
                )

                position_loss = sum(
                    loss.mask_pos_rec_loss(
                        pos,
                        model.poss_list[i],
                        model.v_masks_list[
                            i
                        ]
                        .reshape(-1)
                        .bool(),
                    )
                    * weights[i]
                    for i, pos
                    in enumerate(
                        positions
                    )
                )

                total = (
                    position_loss
                    + cfg["normal_weight"]
                    * loss.mask_norm_rec_loss(
                        norm,
                        normals,
                        fmask,
                    )
                )

                if not torch.isfinite(
                    total
                ):
                    raise ValueError(
                        "Nonfinite full-surface training loss"
                    )

                total.backward()

                epoch_loss += float(
                    total.detach()
                )

            if any(
                parameter.grad is not None
                and not torch.isfinite(
                    parameter.grad
                ).all()
                for parameter
                in model.parameters()
            ):
                raise ValueError(
                    "Nonfinite model gradients"
                )

            optimizer.step()

        scheduler.step()

        print(
            f'Epoch {epoch + 1}/{cfg["epochs"]}: '
            f'{epoch_loss / masks.shape[1]:.6g}',
            flush=True,
        )

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    model.eval()

    with torch.no_grad():
        predicted = (
            model(
                dataset,
                vmask.reshape(-1, 1),
            )[0]
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    if not np.isfinite(
        predicted
    ).all():
        raise ValueError(
            "Nonfinite inference"
        )

    # ---------------------------------------------------------
    # Restore original physical coordinates
    # ---------------------------------------------------------
    # Official SeMIGCN refinement
    #
    # The raw network output is an intermediate reconstruction.
    # The original SeMIGCN pipeline applies Laplacian mesh_merge()
    # after inference to preserve the observed part of the surface
    # and blend the prediction into the missing region.
    # ---------------------------------------------------------

    predicted_tensor = torch.from_numpy(
        predicted
    ).float()

    refined_tensor = Mesh.mesh_merge(
        mesh_dic["ini_mesh"].Lap,
        mesh_dic["ini_mesh"],
        predicted_tensor,
        vmask.detach().cpu(),
        w=float(cfg.get("mu", 1.0)),
    )

    refined = (
        refined_tensor
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    if not np.isfinite(refined).all():
        raise ValueError(
            "Nonfinite SeMIGCN refinement"
        )

    # ---------------------------------------------------------
    # Restore original physical coordinates
    # ---------------------------------------------------------

    return trimesh.Trimesh(
        refined * scale + center,
        initial.faces,
        process=False,
    )
