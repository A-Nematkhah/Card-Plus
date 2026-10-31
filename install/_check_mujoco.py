import os
from os.path import join

mujoco_path = os.getenv("MUJOCO_PY_MUJOCO_PATH") or join(os.path.expanduser("~"), ".mujoco", "mujoco210")
lib_path = join(mujoco_path, "bin")
print("mujoco_path repr:", repr(mujoco_path))
print("lib_path repr:", repr(lib_path))
print("exists mujoco:", os.path.exists(mujoco_path))
print("exists bin:", os.path.exists(lib_path))
parts = os.environ.get("PATH", "").split(";")
print("lib_path in PATH:", lib_path in parts)
for p in parts:
    if "mujoco" in p.lower():
        print(" PATH entry:", repr(p))

import mujoco_py
print("mujoco_py OK")
