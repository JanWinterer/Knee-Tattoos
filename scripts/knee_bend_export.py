"""
Export the SKEL knee flexion (0 -> 90 deg) of one leg for Blender.
Compute all poses in one SKEL call (batch) and check the result.

Run from the repo root:
    python scripts/knee_bend_export.py --gender male --side right
"""
import argparse
import os

import numpy as np
import torch

from skel.kin_skel import pose_param_names, skel_joints_name
from skel_setup import load_skel

N_POSES = 46  # 0, 2, 4, ..., 90 degrees
MAX_DEG = 90
# SKEL bone indices of each leg: femur, tibia, talus, calcn, toes
LEG_BONES = {'right': [1, 2, 3, 4, 5], 'left': [6, 7, 8, 9, 10]}


def dominant_bone(weights):
    """For each vertex, the index of the bone that moves it most: (V,)."""
    w = weights.to_dense() if weights.is_sparse else weights
    return w.argmax(dim=1).numpy()


def submesh(faces, vert_bone, bones):
    """Keep faces whose 3 vertices all belong to `bones`, and renumber them.

    Returns (vert_ids, new_faces):
        vert_ids   original indices of the kept vertices, in order
        new_faces  faces using the new numbering 0..len(vert_ids)-1
    """
    faces = faces.numpy()
    keep_f = np.isin(vert_bone, bones)[faces].all(axis=1)  # True for faces fully on the leg
    faces = faces[keep_f]
    vert_ids = np.unique(faces)  # kept vertices, sorted
    remap = -np.ones(len(vert_bone), dtype=np.int64)  # old index -> new index (-1 = dropped)
    remap[vert_ids] = np.arange(len(vert_ids))
    return vert_ids, remap[faces]


parser = argparse.ArgumentParser(description='Export the SKEL knee flexion of one leg for Blender.')
parser.add_argument('--gender', choices=['male', 'female'], default='male')
parser.add_argument('--side', choices=['right', 'left'], default='right')
args = parser.parse_args()
GENDER, SIDE = args.gender, args.side
BONES = LEG_BONES[SIDE]
KNEE_PARAM = f'knee_angle_{SIDE[0]}'  # SKEL pose names end in _r or _l
ANKLE = BONES[2]  # the ankle joint is where the talus starts
OUT = f'output/knee_bend/knee_bend_anim_{GENDER}_{SIDE}.npz'  # each model/leg gets its own file

skel = load_skel(GENDER)

# Knee angle of every pose, evenly spaced from 0 to 90 degrees: (46,)
knee_deg = np.linspace(0, MAX_DEG, N_POSES)

# One row per pose: (46, 46). Only the chosen knee column is non-zero.
pose = torch.zeros(N_POSES, skel.num_q_params)
pose[:, pose_param_names.index(KNEE_PARAM)] = torch.tensor(np.radians(knee_deg), dtype=torch.float32)
betas = torch.zeros(N_POSES, skel.num_betas)   # same average shape for every pose
trans = torch.zeros(N_POSES, 3)

with torch.no_grad():
    out = skel(pose, betas, trans)

print('knee angles:', knee_deg[:4], '...', knee_deg[-1])
print('skin_verts:', out.skin_verts.shape)    # expect (46, 6890, 3)
print('ankle first / last pose:', out.joints[0, ANKLE].numpy().round(3), out.joints[-1, ANKLE].numpy().round(3))

# Keep only the chosen leg. The selection depends on the skinning weights only (fixed for the model),
# so the same vertices, in the same order, are kept for every pose.
skin_bone = dominant_bone(skel.skin_weights)  # (6890,)
bone_bone = dominant_bone(skel.skel_weights_rigid)  # (247252,)
skin_vid, skin_f = submesh(skel.skin_f, skin_bone, BONES)
bone_vid, bone_f = submesh(skel.skel_f, bone_bone, BONES)

print(f'{SIDE}-leg skin:', len(skin_vid), 'verts,', len(skin_f), 'faces')
print(f'{SIDE}-leg bone:', len(bone_vid), 'verts,', len(bone_f), 'faces')

# Rigid transform of every SKEL bone in world coordinates: rotation = joints_ori, translation = joint centre.
# Femur = bone 1 (right) or 6 (left), tibia = 2 or 7. Maps bone-frame points to world: p_world = T @ p_bone.
n_joints = out.joints.shape[1]
bone_T = np.zeros((N_POSES, n_joints, 4, 4), dtype=np.float32)
bone_T[..., :3, :3] = out.joints_ori.numpy()
bone_T[..., :3, 3] = out.joints.numpy()
bone_T[..., 3, 3] = 1.0

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(
    OUT,
    # --- motion ---
    knee_deg=knee_deg.astype(np.float32),  # (P,)        knee flexion [deg], pose p = frame p
    poses=pose.numpy(),  # (P, 46)     full SKEL pose vector [rad]
    pose_param_names=np.array(pose_param_names),  # (46,)       name of each pose column
    # --- model configuration ---
    gender=np.array(GENDER),
    side=np.array(SIDE),  # which leg: 'right' or 'left'
    betas=betas[0].numpy(),  # (10,)       shape coefficients (0 = average)
    trans=trans.numpy(),  # (P, 3)      global translation [m]
    # --- skin (chosen leg) ---
    skin_verts=out.skin_verts.numpy()[:, skin_vid],  # (P, Vs, 3)  [m]
    skin_faces=skin_f.astype(np.int32),  # (Fs, 3)     same for every pose
    skin_vert_ids=skin_vid.astype(np.int32),  # (Vs,)       index in the full SKEL skin mesh
    skin_label=skin_bone[skin_vid].astype(np.int32),  # (Vs,)       dominant SKEL bone per skin vertex
    # --- bones (chosen leg) ---
    bone_verts=out.skel_verts.numpy()[:, bone_vid],  # (P, Vb, 3)  [m]
    bone_faces=bone_f.astype(np.int32),  # (Fb, 3)
    bone_vert_ids=bone_vid.astype(np.int32),  # (Vb,)       index in the full SKEL bone mesh
    bone_label=bone_bone[bone_vid].astype(np.int32),  # (Vb,)       SKEL bone per bone vertex (see joint_names)
    # --- joints and bone poses (all 24 SKEL bones) ---
    joints=out.joints.numpy(),  # (P, 24, 3)  joint centres [m]
    joint_names=np.array(skel_joints_name),  # (24,)
    bone_transforms=bone_T,  # (P, 24, 4, 4) rigid bone-to-world transforms
    # --- conventions ---
    units=np.array('metres; angles in poses = radians, knee_deg = degrees'),
    coordinate_system=np.array('SKEL world: y up, x = subject left (right leg is -x), z = forward'),
)
print(f'Saved {OUT}')
