import inspect
import numpy as np
import trimesh
import yaml
from benchmark.mesh_utils import ROOT,load,save,fill,topology,boundary_cycles
from benchmark.defects import PatchGenerator
from benchmark.metrics import evaluate

def test_ply_lossless(tmp_path):
    m=trimesh.creation.icosphere(subdivisions=1);m.vertices+=np.pi
    p=tmp_path/'m.ply';save(m,p);b=load(p)
    assert np.array_equal(m.vertices,b.vertices) and np.array_equal(m.faces,b.faces)

def test_metrics_identity_and_fixed_roi():
    m=trimesh.creation.icosphere(subdivisions=2)
    a=evaluate(m,m,[(np.array([0,0,1.]),.6)],4,200)
    b=evaluate(m,m,[(np.array([0,0,1.]),.6)],4,200)
    assert a==b and a['repair_assd']<1e-12 and a['global_hd95']<1e-12

def test_full_surface_network_and_baselines():
    from benchmark.semigcn import repair
    assert list(inspect.signature(repair).parameters)==['damaged','cfg','seed']
    m=trimesh.creation.icosphere(subdivisions=3)
    settings=dict(local_eligibility=False,max_normal_angle_degrees=75,min_faces=8,area_tolerance=.3,max_removed_fraction=.1,local_neighbourhood_multiplier=2.)
    g=PatchGenerator(m,settings);d,ids,_=g.patch(.008,123);d2,ids2,_=g.patch(.008,123)
    assert np.array_equal(ids,ids2) and np.array_equal(d.vertices,d2.vertices)
    assert len(boundary_cycles(d)[0])==1
    d.metadata['synthetic_loops']=[loop.tolist() for loop in boundary_cycles(d)[0]]
    before=d.vertices.copy()
    for refine in [False,True]:
        out=fill(d,refine=refine);assert out.is_watertight and topology(out)==(0,0)
    cfg=yaml.safe_load((ROOT/'config/experiment.yaml').read_text())['semigcn']
    cfg.update(epochs=1,dummy_masks=5,neighbourhoods=[1])
    out=repair(d,cfg,123)
    assert np.isfinite(out.vertices).all() and len(out.faces)>len(d.faces)
    assert np.array_equal(before,d.vertices)

def test_only_synthetic_loop_filled_with_unrelated_nonmanifold_edges():
    sphere=trimesh.creation.icosphere(subdivisions=2)
    keep=sphere.triangles_center[:,2]<.85
    damaged=trimesh.Trimesh(sphere.vertices,sphere.faces[keep],process=False)
    loop=boundary_cycles(damaged)[0][0]
    n=len(damaged.vertices)
    v=np.vstack([damaged.vertices,np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,-1,0]])+10])
    f=np.vstack([damaged.faces,np.array([[0,1,2],[1,0,3],[0,1,4]])+n])
    damaged=trimesh.Trimesh(v,f,process=False);damaged.metadata['synthetic_loops']=[loop.tolist()]
    from benchmark.mesh_utils import edges
    old=edges(damaged);original_boundary={tuple(e) for e in old[1][old[3]==1]}
    synthetic={tuple(sorted(e)) for e in zip(loop,np.roll(loop,-1))}
    for refine in [False,True]:
        out=fill(damaged,refine=refine);new=edges(out)
        assert np.array_equal(out.vertices[:len(v)],v) and np.array_equal(out.faces[:len(f)],f)
        assert topology(out)[0]==topology(damaged)[0]==1
        assert {tuple(e) for e in new[1][new[3]==1]}==original_boundary-synthetic
        assert np.array_equal(fill(damaged,refine=refine).faces,out.faces)
