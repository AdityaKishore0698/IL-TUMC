# Symmetry test of the MuJoCo H1-2 model

## 1. Static check (left minus mirrored right)

* Bodies compared: 25; joints: 10; actuators: 10; missing counterparts: 0
* Largest body mass difference: 4e-06 kg (L_pinky_intermediate)
* Largest body inertia difference: 5.87e-07 kg m² (left_wrist_yaw_link)
* Largest body com difference: 0.00969 m (L_thumb_proximal_base)
* Largest body pos difference: 0.0207 m (L_thumb_proximal)
* Bodies whose geom count differs: ['left_knee_link']; whose collision-geom count differs: [('left_knee_link', -1)]
* Largest joint range difference: 0 (left_hip_yaw_joint)
* Largest joint damping difference: 0 (left_hip_yaw_joint)
* Largest joint armature difference: 0 (left_hip_yaw_joint)
* Largest joint friction difference: 0 (left_hip_yaw_joint)
* Largest actuator ctrlrange difference: 0 (left_hip_yaw_joint)
* Largest actuator forcerange difference: 0 (left_hip_yaw_joint)
* Largest actuator gear difference: 0 (left_hip_yaw_joint)
* Deployment PD gains symmetric: kp True, kd True

## 2. Dynamic check: a recorded wave on its own arm vs mirrored onto the other arm

Robot standing on PD at the default pose; 40 ALMI-X single-hand waves, 6 s each.

| | Recorded wave on its own arm | Same wave mirrored onto the other arm |
|---|---|---|
| left waves (n=20): max torso tilt, median | left arm: 1.000 | right arm: 1.000 |
| left waves: falls | 20 | 20 |
| right waves (n=20): max torso tilt, median | right arm: 1.000 | left arm: 1.000 |
| right waves: falls | 20 | 20 |

* Base roll, |own + mirrored| (0 if symmetric): median 6.0413 rad, max 6.2461 rad; roll magnitude median 3.0805 rad
* Lateral drift, |own + mirrored|: median 0.0081 m, max 0.0361 m; drift magnitude median 0.0145 m
* All runs with the LEFT arm waving: max tilt median 1.000; with the RIGHT arm waving: 1.000
