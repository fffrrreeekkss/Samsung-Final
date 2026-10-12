"""Native-unit, unsquared point-to-triangle distances in a fixed union ROI."""
import numpy as np
import trimesh
from .mesh_utils import topology

def sample(mesh,n,seed,regions=None):
    rng=np.random.default_rng(seed);pool=mesh
    if regions:
        triangles=mesh.triangles;selected=np.zeros(len(triangles),bool)
        for center,radius in regions:
            closest=trimesh.triangles.closest_point(triangles,np.broadcast_to(center,(len(triangles),3)))
            selected|=np.linalg.norm(closest-center,axis=1)<=radius
        pool=trimesh.Trimesh(mesh.vertices,mesh.faces[selected],process=False)
    if not len(pool.faces):raise ValueError('No surface in fixed repair ROI')
    batches=[];normals=[];total=0
    for _ in range(100):
        ids=rng.choice(len(pool.faces),n,p=pool.area_faces/pool.area)
        uv=rng.random((n,2));flip=uv.sum(1)>1;uv[flip]=1-uv[flip];t=pool.triangles[ids]
        points=t[:,0]+uv[:,:1]*(t[:,1]-t[:,0])+uv[:,1:]*(t[:,2]-t[:,0])
        good=np.ones(n,bool) if regions is None else np.any([np.linalg.norm(points-c,axis=1)<=r for c,r in regions],axis=0)
        batches.append(points[good]);normals.append(pool.face_normals[ids[good]]);total+=good.sum()
        if total>=n:return np.vstack(batches)[:n],np.vstack(normals)[:n]
    raise ValueError('Insufficient repair ROI samples')

def proximity(points,mesh):
    query=trimesh.proximity.ProximityQuery(mesh)
    chunks=[query.on_surface(points[i:i+500]) for i in range(0,len(points),500)]
    return np.concatenate([c[1] for c in chunks]),np.concatenate([c[2] for c in chunks])

def distances(points,mesh):return proximity(points,mesh)[0]

def evaluate(gt,pred,regions,seed,n=10000):
    result={}
    for prefix,roi in [('repair',regions),('global',None)]:
        a,na=sample(gt,n,seed,roi);b,nb=sample(pred,n,seed+1,roi)
        da,fa=proximity(a,pred);db,fb=proximity(b,gt)
        result[prefix+'_assd']=float((da.mean()+db.mean())/2)
        result[prefix+'_hd95']=float(max(np.quantile(da,.95),np.quantile(db,.95)))
        if prefix=='repair':
            result.update(repair_gt_to_pred=float(da.mean()),repair_pred_to_gt=float(db.mean()))
            angle_a=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',na,pred.face_normals[fa]),-1,1)))
            angle_b=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',nb,gt.face_normals[fb]),-1,1)))
            result['repair_normal_error_deg']=float((angle_a.mean()+angle_b.mean())/2)
    result['nonmanifold_edges']=topology(pred)[0]
    result['self_intersections']=np.nan
    if not result['nonmanifold_edges']:
        import pymeshlab as ml
        ms=ml.MeshSet();ms.add_mesh(ml.Mesh(pred.vertices,pred.faces));ms.compute_selection_by_self_intersections_per_face()
        result['self_intersections']=int(ms.current_mesh().selected_face_number())
    return result
