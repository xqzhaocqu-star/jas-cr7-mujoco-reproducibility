# Third-party notices

CR7 meshes and xacro originate from the ROKAE ROS 2 robot description,
https://github.com/RokaeRobot/rokae_ros2 (also in the owner's fork
https://github.com/xqzhaocqu-star/rokae_cr7_zhao_sim).
The source workspace distributes these under Apache-2.0; its license is copied
verbatim in `third_party/ROKAE_LICENSE`. Original xacro notices are retained.
The bundled MuJoCo scene derives from these geometries; inertias use the
uniform-density convex-hull approximation described in the model metadata.
Hashes of the meshes and source audit files are in `simulation/models/asset_sha256.json`.
Historical absolute source paths inside the audit JSON are provenance strings,
not runtime dependencies.

The independent PPC implementation follows Verginis, Mastellaro and Dimarogonas,
“Robust Cooperative Manipulation without Force/Torque Measurements: Control Design
and Experiments”, IEEE TCST 28(3), 713–729 (2020).
Primary reference: https://arxiv.org/abs/1710.11088 .
See `simulation/docs/external_ppc_baseline.md` for equation mappings and limitations.

No blanket open-source license is granted here for the new research controller
code. This repository is initially private; the owner can choose its release
license separately. The third-party license above remains applicable.
