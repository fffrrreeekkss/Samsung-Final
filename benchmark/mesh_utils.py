"""Source loading, minimal cleanup, double-precision PLY and MeshLab baselines."""
from pathlib import Path
import re
import numpy as np
import trimesh
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]

def hypersurface(path):
    b = Path(path).read_bytes()
    if not b.startswith(b'# HyperSurface'):
        raise ValueError('Unsupported .surf variant (expected Amira HyperSurface)')
    binary = b'BINARY' in b.split(b'\n', 1)[0]
    m = re.search(rb'(?m)^Vertices[ \t]+(\d+)[ \t]*\r?\n', b)
    if m is None:
        raise ValueError('Missing Vertices block')
    n, offset = int(m[1]), m.end()
    if binary:
        # HyperSurface writes an optional empty separator line before vertices.
        if b[offset:offset+2] == b'\r\n': offset += 2
        elif b[offset:offset+1] == b'\n': offset += 1
        endian = '<' if b'LITTLE-ENDIAN' in b[:80] else '>'
        vertices = np.frombuffer(b, endian+'f4', n*3, offset).astype(float).reshape(-1, 3)
        offset += n*12
    else:
        tail = b[offset:].decode('ascii'); lines = tail.splitlines(keepends=True)
        coords, consumed = [], 0
        for line in lines:
            consumed += len(line)
            if line.strip(): coords.extend(map(float, line.split()))
            if len(coords) >= 3*n: break
        if len(coords) != 3*n: raise ValueError('Malformed ASCII coordinates')
        vertices = np.asarray(coords).reshape(-1, 3); offset += consumed
    p = re.search(rb'\bPatches[ \t]+(\d+)', b[offset:])
    if p is None: raise ValueError('Missing patch count')
    patch_count = int(p[1]); faces = []
    for _ in range(patch_count):
        t = re.search(rb'(?m)^[ \t]*Triangles[ \t]+(\d+)[ \t]*\r?\n', b[offset:])
        if t is None: raise ValueError('Missing triangle block')
        nf = int(t[1]); offset += t.end()
        if binary:
            f = np.frombuffer(b, endian+'i4', nf*3, offset).astype(np.int64).reshape(-1, 3)
            offset += nf*12
        else:
            lines = b[offset:].decode('ascii').splitlines(keepends=True); values=[]
            for line in lines:
                offset += len(line)
                if line.strip(): values.extend(map(int, line.split()))
                if len(values) >= nf*3: break
            f = np.asarray(values, dtype=np.int64).reshape(-1, 3)
        faces.append(f-1)
    faces = np.concatenate(faces)
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= n:
        raise ValueError('Invalid coordinates or triangle indices')
    return trimesh.Trimesh(vertices, faces, process=False)

def load(path):
    path = Path(path)
    if path.suffix.lower() == '.surf': return hypersurface(path)
    m = trimesh.load(path, force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh): raise ValueError('Not a triangular mesh')
    if path.suffix.lower() == '.stl':
        # STL is triangle soup. Exact-coordinate indexing only; no rounding/welding tolerance.
        v, inv = np.unique(np.asarray(m.vertices), axis=0, return_inverse=True)
        m = trimesh.Trimesh(v, inv[m.faces], process=False)
    return m


def save(mesh,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix=='.obj':
        path.write_text(trimesh.exchange.obj.export_obj(mesh,digits=17,include_color=False));return
    if path.suffix!='.ply':raise ValueError('Final meshes must be PLY')
    with path.open('w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(mesh.vertices)}\nproperty double x\nproperty double y\nproperty double z\nelement face {len(mesh.faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        np.savetxt(f,mesh.vertices,fmt='%.17g')
        np.savetxt(f,np.c_[np.full(len(mesh.faces),3),mesh.faces],fmt='%d')

def edges(mesh):
    oriented=mesh.faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    unique,inv,count=np.unique(np.sort(oriented,axis=1),axis=0,return_inverse=True,return_counts=True)
    return oriented,unique,inv,count

def face_graph(mesh,weighted=False):
    adj=mesh.face_adjacency
    w=np.linalg.norm(mesh.triangles_center[adj[:,0]]-mesh.triangles_center[adj[:,1]],axis=1) if weighted else np.ones(len(adj))
    return sparse.coo_matrix((np.r_[w,w],(np.r_[adj[:,0],adj[:,1]],np.r_[adj[:,1],adj[:,0]])),shape=(len(mesh.faces),)*2).tocsr()

def complete_face_graph(mesh):
    _,_,inv,count=edges(mesh);order=np.argsort(inv,kind='stable');starts=np.r_[0,np.cumsum(count)]
    a=[];b=[]
    for k in np.flatnonzero(count>1):
        ids=order[starts[k]:starts[k+1]]//3
        a.extend([ids[0]]*(len(ids)-1));b.extend(ids[1:])
    return sparse.coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(len(mesh.faces),)*2).tocsr()

def boundary_cycles(mesh):
    oriented, unique, inv, count = edges(mesh)
    boundary = oriented[count[inv] == 1]
    adjacency = {}
    for a,b in boundary:
        adjacency.setdefault(int(a), []).append(int(b)); adjacency.setdefault(int(b), []).append(int(a))
    unseen = set(adjacency); loops=[]; invalid=0
    while unseen:
        first = min(unseen); stack=[first]; comp=set()
        while stack:
            u=stack.pop()
            if u in comp: continue
            comp.add(u); stack.extend(v for v in adjacency[u] if v not in comp)
        unseen -= comp
        if any(len(adjacency[u]) != 2 for u in comp): invalid += 1; continue
        loop=[first]; prev=None; u=first
        while True:
            nxt = next(v for v in sorted(adjacency[u]) if v != prev)
            if nxt == first: break
            loop.append(nxt); prev,u=u,nxt
        loops.append(np.array(loop, dtype=int))
    return loops, invalid


def topology(mesh):
    _,_,_,count=edges(mesh)
    return int((count>2).sum()),int((count==1).sum())

def fill(mesh,refine=False):
    """Append only the explicitly identified synthetic patches; never repair globally."""
    loops=mesh.metadata.get('synthetic_loops')
    if not loops:raise ValueError('Synthetic boundary loops required; refusing to guess anatomical holes')
    oriented,unique,inv,count=edges(mesh)
    boundary={tuple(sorted(e)):tuple(e[::-1]) for e in oriented[count[inv]==1]}
    vertices=mesh.vertices.tolist();added=[]
    target=float(mesh.edges_unique_length.mean())*.8
    for supplied in loops:
        ring=list(map(int,supplied))
        if len(set(ring))!=len(ring) or len(ring)<3:raise ValueError('Invalid synthetic loop')
        keys=[tuple(sorted((a,b))) for a,b in zip(ring,np.roll(ring,-1))]
        if not all(k in boundary for k in keys):raise ValueError('Synthetic loop is not an exposed damaged boundary')
        if boundary[keys[0]]!=(ring[0],ring[1]):ring.reverse()
        if not all(boundary[tuple(sorted((a,b)))]==(a,b) for a,b in zip(ring,np.roll(ring,-1))):raise ValueError('Inconsistent boundary winding')
        coords=mesh.vertices[ring];center=coords.mean(0)
        _,_,vh=np.linalg.svd(coords-center,full_matrices=False)
        xy=(mesh.vertices-center)@vh[:2].T
        def cross(a,b):return a[0]*b[1]-a[1]*b[0]
        sign=np.sign(sum(cross(xy[a],xy[b]) for a,b in zip(ring,np.roll(ring,-1))))
        eps=np.sum(np.ptp(coords,axis=0)**2)*1e-12
        faces=[]
        while len(ring)>3:
            for i,b in enumerate(ring):
                a,c=ring[i-1],ring[(i+1)%len(ring)]
                if sign*cross(xy[b]-xy[a],xy[c]-xy[b])<=eps:continue
                if any(all(sign*cross(xy[y]-xy[x],xy[v]-xy[x])>=-eps for x,y in [(a,b),(b,c),(c,a)]) for v in ring if v not in (a,b,c)):continue
                faces.append([a,b,c]);ring.pop(i);break
            else:raise ValueError('Synthetic polygon has no valid projected ear')
        faces.append(ring)
        # Same old initialization scale:0.8 mean input edge. Split only new faces.
        for _ in range(10 if refine else 0):
            next_faces=[];changed=False
            for face in faces:
                tri=np.asarray([vertices[i] for i in face])
                if np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2<=np.sqrt(3)*target**2/4:
                    next_faces.append(face);continue
                mid=len(vertices);vertices.append(tri.mean(0).tolist());a,b,c=face
                next_faces.extend([[a,b,mid],[b,c,mid],[c,a,mid]]);changed=True
            faces=next_faces
            if not changed:break
        added.extend(faces)
    out=trimesh.Trimesh(np.asarray(vertices),np.vstack([mesh.faces,added]),process=False)
    if not np.array_equal(out.vertices[:len(mesh.vertices)],mesh.vertices) or not np.array_equal(out.faces[:len(mesh.faces)],mesh.faces):raise ValueError('Initializer changed observed geometry')
    return out
