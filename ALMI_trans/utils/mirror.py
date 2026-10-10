"""Left-right mirroring of ALMI-X state-action vectors and captions (symmetry augmentation).

The Unitree H1-2 is left-right symmetric and its left and right joints use the same axes, so reflecting a
trajectory in the robot's sagittal plane gives another valid trajectory: left and right joints swap, and
joints that turn about the x axis (roll) or z axis (yaw) change sign, while pitch joints (y axis) keep it.
A "wave left" trajectory mirrored this way is a "wave right" trajectory, which the captions follow by
swapping the words left and right.

71-dim observation: base angular velocity (3), projected gravity (3), joint offsets (21), joint
velocities (21), last action (21), gait phase (2: sin of the left and right leg phases).
"""
import re

import numpy as np
import torch

# joint order: 6 left leg, 6 right leg (hip yaw, hip pitch, hip roll, knee, ankle pitch, ankle roll),
# torso yaw, 4 left arm, 4 right arm (shoulder pitch, shoulder roll, shoulder yaw, elbow pitch)
_LEG_SIGN = [-1, 1, -1, 1, 1, -1]
_ARM_SIGN = [1, -1, -1, 1]
JOINT_PERM = list(range(6, 12)) + list(range(0, 6)) + [12] + list(range(17, 21)) + list(range(13, 17))
JOINT_SIGN = np.array(_LEG_SIGN * 2 + [-1] + _ARM_SIGN * 2, np.float32)

OBS_PERM = ([0, 1, 2, 3, 4, 5]
            + [6 + j for j in JOINT_PERM] + [27 + j for j in JOINT_PERM] + [48 + j for j in JOINT_PERM]
            + [70, 69])
OBS_SIGN = np.concatenate([[-1, 1, -1], [1, -1, 1], JOINT_SIGN, JOINT_SIGN, JOINT_SIGN, [1, 1]]).astype(np.float32)


def mirror_obs(x):
    """Mirror (..., 71) observation-action vectors (numpy array or torch tensor)."""
    if isinstance(x, torch.Tensor):
        return x[..., OBS_PERM] * torch.as_tensor(OBS_SIGN, device=x.device, dtype=x.dtype)
    return x[..., OBS_PERM] * OBS_SIGN


def mirror_caption(text):
    """Swap the words left and right, e.g. 'go left fast and wave right' -> 'go right fast and wave left'."""
    return re.sub(r"\b(left|right)\b", lambda m: "right" if m.group(1) == "left" else "left", text)


assert sorted(OBS_PERM) == list(range(71))
assert np.allclose(mirror_obs(mirror_obs(np.arange(71, dtype=np.float32))), np.arange(71))
