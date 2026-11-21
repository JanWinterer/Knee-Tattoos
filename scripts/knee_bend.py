"""
Pose SKEL with a specific knee bend and export the meshes.
SKEL takes three inputs and returns a posed 3D body:
    pose  (B, 46)  joint angles in radians, named as in skel.kin_skel.pose_param_names
    betas (B, 10)  body shape coefficients, 0 = average body
    trans (B, 3)   global translation in metres
where B is the batch size (number of bodies computed at once; here 1).

Coordinates: metres, y up, right side of the body is -x, z forward.
"""
import math
import os

import torch
import trimesh

from skel.kin_skel import pose_param_names
from skel_setup import load_skel

KNEE_DEG = 90 # right knee flexion in degrees (0 = straight)

skel = load_skel('male')

# Inputs: one body (batch size 1), all zeros = reference pose, average shape, at the origin.
pose = torch.zeros(1, skel.num_q_params)  # (1, 46) joint angles [rad]
betas = torch.zeros(1, skel.num_betas)  # (1, 10) shape coefficients
trans = torch.zeros(1, 3)  # (1, 3)  global position [m]

# Bend the right knee: find its column by name (it is column 6) and set the angle in radians.
# This must happen before skel(...) runs, because SKEL reads `pose` at that moment.
pose[0, pose_param_names.index('knee_angle_r')] = math.radians(KNEE_DEG)

with torch.no_grad():
    out = skel(pose, betas, trans)

# Outputs (first dimension = batch):
# skin_verts  (1, 6890, 3)     skin surface points [m]
# skel_verts  (1, 247252, 3)   bone surface points [m]
# joints      (1, 24, 3)       joint centres [m]: 0 pelvis, 1 hip_r, 2 knee_r, 3 ankle_r, ...
print('skin vertices:', out.skin_verts.shape)
print('bone vertices:', out.skel_verts.shape)
print('joints:       ', out.joints.shape)

skin = trimesh.Trimesh(out.skin_verts[0].numpy(), skel.skin_f.numpy())
bones = trimesh.Trimesh(out.skel_verts[0].numpy(), skel.skel_f.numpy())

os.makedirs('output/knee_bend', exist_ok=True)
skin.export(f'output/knee_bend/skin_male_kneeR{KNEE_DEG}.obj')
bones.export(f'output/knee_bend/bones_male_kneeR{KNEE_DEG}.obj')
print(f'Saved output/knee_bend/skin_male_kneeR{KNEE_DEG}.obj and bones_male_kneeR{KNEE_DEG}.obj')

for i, name in [(1, 'hip_r'), (2, 'knee_r'), (3, 'ankle_r')]:
    print(f'{name:8s} {out.joints[0, i].numpy().round(3)}')