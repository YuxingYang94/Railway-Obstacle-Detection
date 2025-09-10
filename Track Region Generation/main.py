#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

# ================== settings ==================
RAW_DIR     = "./input"
PLANE_DIR   = "./plane"
ALIGNED_DIR = "./align"
CROP_DIR    = ALIGNED_DIR

LABEL_FIELD = "pred"

EPS, MIN_SAMPLES, MIN_DIAG = 0.5, 50, 5.0
PARALLEL_XNORM_TH, NEED_PARALLEL_CNT = 0.01, 3
PREFER_Z_UP = True
# =====================================================

os.makedirs(PLANE_DIR, exist_ok=True)
os.makedirs(ALIGNED_DIR, exist_ok=True)


def norm(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def pca_dir(pts):
    p = PCA(n_components=3).fit(pts)
    return norm(p.components_[0])

def bounding_diag(pts):
    mn, mx = np.min(pts,0), np.max(pts,0)
    return float(np.linalg.norm(mx-mn))

def cluster_points(pts):
    if pts.size == 0: return []
    labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(pts).labels_
    clusters = []
    for lab in set(labels):
        if lab == -1: continue
        cl = pts[labels==lab]
        if bounding_diag(cl) >= MIN_DIAG: clusters.append(cl)
    return clusters

def avg_dir(vectors):
    return norm(np.mean(np.stack(vectors,0),0)) if vectors else np.zeros(3)

def sign_align(vectors, ref):
    ref = norm(ref); out=[]
    for v in vectors:
        v = norm(v)
        out.append(-v if np.dot(v,ref)<0 else v)
    return out

def many_parallel(dirs):
    if len(dirs)<NEED_PARALLEL_CNT: return False
    for i,di in enumerate(dirs):
        cnt=1
        for j,dj in enumerate(dirs):
            if i==j: continue
            if np.linalg.norm(np.cross(di,dj))<PARALLEL_XNORM_TH: cnt+=1
        if cnt>=NEED_PARALLEL_CNT: return True
    return False

def plane_normal(main_dir, centers):
    if len(centers)<2: return np.array([0,0,1])
    v = np.asarray(centers[0])-np.asarray(centers[1])
    n = norm(np.cross(main_dir,v))
    if PREFER_Z_UP and np.dot(n,[0,0,1])<0: n=-n
    return n

def save_plane(base, avg_d, n, dirs):
    path = os.path.join(PLANE_DIR,f"{base}_plane.txt")
    data={"Average_Principal_Direction":avg_d.tolist(),
          "Normal_Vector":n.tolist(),
          "Cluster_Principal_Directions":[d.tolist() for d in dirs]}
    with open(path,"w") as f: f.write(json.dumps(data,indent=2,ensure_ascii=False))

def read_plane(txt_path):
    with open(txt_path,"r") as f:
        d=json.load(f)
    return np.array(d["Average_Principal_Direction"]), np.array(d["Normal_Vector"])


# ---------- Step1: extract plane ----------
def generate_plane_file(fname, ply):
    pts=np.column_stack((ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]))
    labels=np.asarray(ply["vertex"][LABEL_FIELD])
    pts=pts[labels!=0]

    clusters=cluster_points(pts)
    if not clusters: return
    dirs,centers=[],[]
    for cl in clusters:
        dirs.append(pca_dir(cl))
        mn,mx=np.min(cl,0),np.max(cl,0)
        centers.append((mn+mx)/2)
    if many_parallel(dirs):
        davg=avg_dir(dirs)
        dirs=sign_align(dirs,davg)
        n=plane_normal(davg,centers)
        save_plane(os.path.splitext(fname)[0],davg,n,dirs)


# ---------- Step2:  aligned main direction ----------
def align_and_save(fname, ply):
    txt_path=os.path.join(PLANE_DIR,os.path.splitext(fname)[0]+"_plane.txt")
    if not os.path.isfile(txt_path): return
    avg_d, normal = read_plane(txt_path)

    all_pts=np.column_stack((ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"], ply["vertex"][LABEL_FIELD]))

    y_axis=np.array([0,1,0])
    rv1=np.cross(avg_d,y_axis)
    ang1=np.arccos(np.dot(avg_d,y_axis)/(np.linalg.norm(avg_d)*np.linalg.norm(y_axis)))
    R1=np.eye(3) if np.linalg.norm(rv1)<1e-10 else R.from_rotvec(rv1/np.linalg.norm(rv1)*ang1).as_matrix()
    rot_pts=np.dot(all_pts[:,:3],R1.T)
    rot_normal=np.dot(normal,R1.T)

    z_axis=np.array([0,0,1])
    rv2=np.cross(rot_normal,z_axis)
    ang2=np.arccos(np.dot(rot_normal,z_axis)/(np.linalg.norm(rot_normal)*np.linalg.norm(z_axis)))
    R2=np.eye(3) if np.linalg.norm(rv2)<1e-10 else R.from_rotvec(rv2/np.linalg.norm(rv2)*ang2).as_matrix()
    final_pts=np.dot(rot_pts,R2.T)

    arr=np.array([(p[0],p[1],p[2],int(l)) for p,l in zip(final_pts,all_pts[:,3])],
                 dtype=[('x','f4'),('y','f4'),('z','f4'),('pred','i4')])
    out_path=os.path.join(ALIGNED_DIR,os.path.splitext(fname)[0]+"_aligned.ply")
    PlyData([PlyElement.describe(arr,"vertex")]).write(out_path)
    return out_path


# ---------- Step3: generate crop ----------
def compute_bbox(points):
    return np.min(points,0), np.max(points,0)

def expand_bbox(min_bounds,max_bounds):
    min_b,max_c = min_bounds.copy(), max_bounds.copy()
    min_b[0]-=1; min_b[2]-=1
    max_c[0]+=1; max_c[1]+=2; max_c[2]+=4
    return min_b,max_c

def generate_points_in_bbox(min_b,max_c,density=1):
    x=np.arange(min_b[0],max_c[0],density)
    y=np.arange(min_b[1],max_c[1],density)
    z=np.arange(min_b[2],max_c[2],density)
    gx,gy,gz=np.meshgrid(x,y,z)
    return np.vstack([gx.ravel(),gy.ravel(),gz.ravel()]).T

def crop_from_aligned(ply_path):
    ply= PlyData.read(ply_path)
    pts=np.column_stack((ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]))
    preds=np.asarray(ply["vertex"]["pred"])
    valid=pts[preds!=0]
    if valid.shape[0]==0: return

    labels=DBSCAN(eps=0.2,min_samples=200).fit_predict(valid)
    clusters=[]
    for lab in np.unique(labels):
        cl=valid[labels==lab]
        if len(cl)<800: continue
        min_b,max_c=compute_bbox(cl)
        if (max_c[1]-min_b[1])<5: continue
        clusters.append(cl)
    if not clusters: return

    merged=np.vstack(clusters)
    min_b,max_c=compute_bbox(merged)
    min_b,max_c=expand_bbox(min_b,max_c)
    bbox_pts=generate_points_in_bbox(min_b,max_c)

    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(bbox_pts)
    out_path=ply_path.replace(".ply","_crop.pcd")
    o3d.io.write_point_cloud(out_path,pcd)
    print("保存:",out_path)


# ---------- main ----------
def main():
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".ply"): continue
        ply= PlyData.read(os.path.join(RAW_DIR,fname))
        generate_plane_file(fname,ply)
        aligned_path=align_and_save(fname,ply)
        if aligned_path: crop_from_aligned(aligned_path)

if __name__=="__main__":
    main()

