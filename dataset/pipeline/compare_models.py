import os
import sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("PYOPENGL_PLATFORM","egl")
from vla_branch_lab.assets import configure_libero
from vla_branch_lab.config import load_config
configure_libero(load_config(ROOT/"configs/base.yaml"))
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
s=benchmark.get_benchmark_dict()["libero_spatial"]()
e=OffScreenRenderEnv(bddl_file_name=s.get_task_bddl_file_path(0),camera_heights=128,camera_widths=128,control_freq=20,hard_reset=False,render_gpu_device_id=3)
e.seed(20261009);e.reset()
a=e.sim.model._model
b=mujoco.MjModel.from_xml_path(str((Path(sys.argv[1])/"model.xml").resolve()))
for name in ["body_mass","body_inertia","body_ipos","body_iquat","body_pos","body_quat","geom_pos","geom_quat","geom_size","geom_contype","geom_conaffinity","geom_condim","geom_solref","geom_solimp","geom_friction","jnt_pos","jnt_axis","qpos0","eq_data"]:
    x=np.asarray(getattr(a,name));y=np.asarray(getattr(b,name))
    print(name,x.shape,y.shape,"max",np.max(np.abs(x-y)) if x.size and x.shape==y.shape else "shape_diff",flush=True)
    if x.shape==y.shape and x.ndim>1:
        ids=np.flatnonzero(np.max(np.abs(x-y),axis=1)>1e-7)
        for i in ids[:5]:
            print(" DIFF",i,x[i],y[i],flush=True)
print("OPTIONS",[(n,getattr(a.opt,n),getattr(b.opt,n)) for n in ["timestep","iterations","cone","integrator","gravity"]],flush=True)
Path("reference.xml").write_text(e.env.model.get_xml())
e.close()
