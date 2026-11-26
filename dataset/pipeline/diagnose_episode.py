import sys
from pathlib import Path
import h5py
import mujoco
import numpy as np
folder=Path(sys.argv[1])
model=mujoco.MjModel.from_xml_path(str((folder/"model.xml").resolve()))
print("PAIRS",model.npair)
for i in range(model.npair):
    g1,g2=model.pair_geom1[i],model.pair_geom2[i]
    print(i,mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g1)),int(model.geom_contype[g1]),int(model.geom_conaffinity[g1]),"group",int(model.geom_group[g1]),mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g2)),model.pair_friction[i].tolist(),model.pair_solref[i].tolist())
with h5py.File(folder/"trajectory.hdf5") as f:
    e=f["observations/eef"][:]; b=f["observations/bowl"][:]; states=f["observations/state"][:]
    contact=f["observations/contact_sides"][:]
    phase=f["phase"][:]
    for i in range(0,len(e),20):
        print("FRAME",i,"time",states[i,0],"eef",e[i],"bowl",b[i],"contact",contact[i])
