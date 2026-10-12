"""Deterministic vessel-scale connected defects; large has three separated holes."""
import heapq
import math
import numpy as np
import trimesh
from scipy.sparse.csgraph import connected_components
from .mesh_utils import edges,boundary_cycles,face_graph,complete_face_graph

class PatchGenerator:
    def __init__(self, mesh, settings):
        self.mesh=mesh; self.c=settings; self.graph=face_graph(mesh,weighted=True)
        self.areas=mesh.area_faces; self.centers=mesh.triangles_center
        self.normals=mesh.face_normals
        _,_,inv,count=edges(mesh)
        self.edge_inverse=inv.reshape(-1,3); self.edge_counts=count
        self.forbidden=np.any(count[inv].reshape(-1,3) != 2,axis=1)
        self.complete_graph=complete_face_graph(mesh)
        self.component_count,self.component_labels=connected_components(self.complete_graph,directed=False)
        self.component_areas=np.bincount(self.component_labels,weights=self.areas)
        self.local_policy=bool(settings.get('local_eligibility',False))
        if self.local_policy:
            from scipy.spatial import cKDTree
            component_faces=np.bincount(self.component_labels)
            min_area=max(settings['candidate_area_fractions'])/settings['max_component_removed_fraction']*mesh.area
            too_small=(self.component_areas<min_area)|(component_faces<settings['min_component_faces'])
            self.forbidden |= too_small[self.component_labels]
            self.forbidden |= self.areas<=settings['degenerate_relative_area']*np.linalg.norm(np.ptp(mesh.vertices,axis=0))**2
            # Explicit topological collar plus a spatial collar against nearby bad sheets/islands.
            for _ in range(settings['exclusion_rings']): self.forbidden |= np.asarray(self.graph@self.forbidden.astype(float)).ravel()>0
            if self.forbidden.any():
                bad_vertices=np.unique(mesh.faces[self.forbidden])
                separation=cKDTree(mesh.vertices[bad_vertices]).query(self.centers)[0]
                self.forbidden |= separation < settings['exclusion_distance_edge_factor']*np.median(mesh.edges_unique_length)
        self.valid=~self.forbidden

    def patch(self, fraction, seed):
        rng=np.random.default_rng(seed); m=self.mesh; target=m.area*fraction
        probabilities=self.areas*self.valid
        if probabilities.sum()==0: raise ValueError('No valid local region remains after exclusion collar')
        start=int(rng.choice(len(m.faces),p=probabilities/probabilities.sum()))
        if self.forbidden[start]: raise ValueError('Seed touches original boundary/nonmanifold edge')
        seen=set(); queue=[(0.0,start)]; dist={start:0.0}; area=0.; patch=[]
        normal_cos=math.cos(math.radians(self.c['max_normal_angle_degrees']))
        while queue and area < target:
            d,u=heapq.heappop(queue)
            if u in seen: continue
            seen.add(u)
            if self.forbidden[u] or self.normals[u] @ self.normals[start] < normal_cos:
                continue
            patch.append(u); area+=self.areas[u]
            a,b=self.graph.indptr[u:u+2]
            for v,w in zip(self.graph.indices[a:b],self.graph.data[a:b]):
                nd=d+w
                if nd < dist.get(int(v),np.inf):
                    dist[int(v)]=nd; heapq.heappush(queue,(nd,int(v)))
        ids=np.sort(patch)
        component=int(self.component_labels[start])
        if len(ids)<self.c['min_faces']: raise ValueError('Too few faces at requested scale')
        if abs(area/target-1)>self.c['area_tolerance']: raise ValueError('Area outside severity tolerance')
        if area/m.area>self.c['max_removed_fraction']: raise ValueError('Excess removed area')
        if self.local_policy and area/self.component_areas[component]>self.c['max_component_removed_fraction']:
            raise ValueError('Patch removes too much of selected component')
        patchmesh=trimesh.Trimesh(m.vertices,m.faces[ids],process=False)
        loops,bad=boundary_cycles(patchmesh)
        if bad or len(loops)!=1: raise ValueError('Patch is not a single disk boundary')
        if connected_components(self.graph[ids][:,ids],directed=False,return_labels=False)!=1:
            raise ValueError('Patch disconnected')
        used=np.unique(m.faces[ids]); eu=len(used)-len(patchmesh.edges_unique)+len(ids)
        if eu!=1: raise ValueError('Patch Euler characteristic is not a disk')
        keep=np.ones(len(m.faces),bool); keep[ids]=False
        # Preserve the existing component count; globally disconnected references are allowed.
        if connected_components(self.complete_graph[keep][:,keep],directed=False,return_labels=False)!=self.component_count:
            raise ValueError('Defect disconnects reference')
        damaged=trimesh.Trimesh(m.vertices,m.faces[keep],process=False)
        affected,removed_counts=np.unique(self.edge_inverse[ids],return_counts=True)
        if np.any(self.edge_counts[affected]!=2): raise ValueError('Patch meets pre-existing pathology')
        new_boundary_count=int(np.sum(removed_counts==1))
        if new_boundary_count!=len(loops[0]): raise ValueError('Synthetic boundary is not uniquely identified')
        damaged.remove_unreferenced_vertices()
        return damaged,ids,dict(removed_surface_area=area)

def generate(mesh,severity,seed,settings):
    cfg=dict(local_eligibility=True,candidate_area_fractions=[1e-5],max_component_removed_fraction=.15,min_component_faces=200,
             degenerate_relative_area=1e-14,exclusion_rings=4,exclusion_distance_edge_factor=2,max_normal_angle_degrees=80,
             min_faces=12,area_tolerance=.3,max_removed_fraction=.08,local_neighbourhood_multiplier=2.)
    g=PatchGenerator(mesh,cfg);removed=set();regions=[];holes=[];synthetic=[]
    count=settings['large_holes'] if severity=='large' else 1
    edge=float(np.median(mesh.edges_unique_length));diag=float(np.linalg.norm(np.ptp(mesh.vertices,axis=0)))
    for hole in range(count):
        found=False
        for attempt in range(settings['max_attempts']):
            trial=int(np.random.SeedSequence([seed,hole,attempt]).generate_state(1)[0]);rng=np.random.default_rng(trial)
            p=g.areas*g.valid;p=p/p.sum();start=int(rng.choice(len(mesh.faces),p=p))
            origin=mesh.triangles_center[start];normal=mesh.face_normals[start]
            hits,_,fi=mesh.ray.intersects_location([origin-normal*edge*1e-4],[-normal],multiple_hits=True)
            ds=np.linalg.norm(hits-origin,axis=1);valid=(ds>edge*.5)&(mesh.face_normals[fi]@normal<-.5)
            if not valid.any():continue
            vessel_radius=float(ds[valid].min()/2)
            if not edge*2<vessel_radius<diag*.15:continue
            target=np.pi*(settings['small_radius_ratio']*vessel_radius)**2
            if severity=='medium':target*=settings['medium_area_factor']
            try:
                _,ids,meta=g.patch(target/mesh.area,trial)
                if holes and meta['removed_surface_area']<minimum_hole_area:continue
                points=mesh.vertices[np.unique(mesh.faces[ids])];center=points.mean(0);radius=float(np.linalg.norm(points-center,axis=1).max())
                if any(np.linalg.norm(center-c)<1.5*(radius+r) for c,r in holes):continue
                if removed.intersection(ids):continue
                synthetic.append(boundary_cycles(trimesh.Trimesh(mesh.vertices,mesh.faces[ids],process=False))[0][0])
                removed.update(ids.tolist());holes.append((center,radius));regions.append((center,radius*2))
                if hole==0:minimum_hole_area=meta['removed_surface_area']
                found=True;break
            except ValueError:continue
        if not found:raise ValueError(f'Could not create separated hole {hole+1}/{count}')
    ids=np.array(sorted(removed));keep=np.ones(len(mesh.faces),bool);keep[ids]=False
    if mesh.area_faces[ids].sum()/mesh.area>.15:raise ValueError('Excess removed area')
    used,inverse=np.unique(mesh.faces[keep],return_inverse=True)
    damaged=trimesh.Trimesh(mesh.vertices[used],inverse.reshape(-1,3),process=False)
    damaged.metadata['synthetic_loops']=[np.searchsorted(used,loop).tolist() for loop in synthetic]
    original_loops,bad=boundary_cycles(mesh);new_loops,new_bad=boundary_cycles(damaged)
    if len(new_loops)!=len(original_loops)+count or new_bad!=bad:raise ValueError('Incorrect synthetic hole count')
    if connected_components(complete_face_graph(damaged),directed=False,return_labels=False)!=g.component_count:raise ValueError('Damage disconnected original components')
    size=dict(holes=count,removed_faces=len(ids),removed_area=float(mesh.area_faces[ids].sum()),area_fraction=float(mesh.area_faces[ids].sum()/mesh.area))
    return damaged,regions,size
