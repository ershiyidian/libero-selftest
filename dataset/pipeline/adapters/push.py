from pathlib import Path
import sys

import numpy as np
import mujoco


ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/name) for name in ("pipeline","planner","physics")]
from real_task_support_v5 import RealTaskEpisode,UnsafeCandidate
from constrained_iterative import mjdata_from_flat


class Episode(RealTaskEpisode):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        data=mujoco.MjData(self.model._model)
        self.source_plate_rot=[]
        self.source_contact_count=[]
        for state in self.expert_states:
            mjdata_from_flat(self.model._model,state,data)
            self.source_plate_rot.append(data.xmat[self.target_body_id].reshape(3,3).copy())
            count=0
            for contact in data.contact[:data.ncon]:
                bodies={int(self.model.geom_bodyid[contact.geom1]),int(self.model.geom_bodyid[contact.geom2])}
                if self.target_body_id in bodies and any((self.model.body_id2name(body) or "").startswith("gripper0_") for body in bodies):
                    count+=1
            self.source_contact_count.append(count)
        self.source_plate_rot=np.asarray(self.source_plate_rot)

    def reference_contact(self):
        candidates=[]
        for index in range(1,len(self.source_xyz)-4):
            if self.source_contact_count[index]==0:
                continue
            object_motion=self.source_target_xyz[index+3]-self.source_target_xyz[index]
            hand_motion=self.source_xyz[index+3]-self.source_xyz[index]
            distance=float(np.linalg.norm(object_motion[:2]))
            if distance<0.004:
                continue
            tracking=float(np.linalg.norm(object_motion[:2]-hand_motion[:2]))
            if tracking<0.003:
                candidates.append(index)
        if not candidates:
            raise UnsafeCandidate("原专家中没有找到真实盘沿接触")
        reference=min(candidates,key=lambda index:np.linalg.norm(self.source_target_xyz[index]-self.target_pos()))
        offset=self.source_plate_rot[reference].T@(self.source_xyz[reference]-self.source_target_xyz[reference])
        orientation=self.source_plate_rot[reference].T@self.source_rot[reference]
        current=self.target_rot()
        return self.target_pos()+current@offset,current@orientation

    def direction(self):
        delta=self.source_target_xyz[-1]-self.target_pos()
        delta[2]=0
        length=float(np.linalg.norm(delta))
        if length<0.001:
            raise UnsafeCandidate("推动目标已到达但原任务判定未满足")
        return delta/length

    def perform_probe(self,scale=1.0):
        start=len(self.actions)
        hand=self.eef().copy()
        before=self.target_pos().copy()
        direction=self.direction()
        rotation=self.rot().copy()
        grip=float(self.expert_actions[self.source_fork,6])
        self._move_plain(hand+direction*0.007*scale,rotation,grip,steps=9,phase="probe")
        self.hold(steps=4,phase="probe")
        displacement=float(np.dot(self.target_pos()-before,direction))
        tip=float(np.dot(self.eef()-hand,direction))
        witness={"method":self.method_id,"method_title":self.method_card["title"],"start_step":start,"end_step":len(self.actions),"actual_steps":len(self.actions)-start,"actual_motion":displacement,"actual_ee_motion_m":tip,"object_relative_shift_m":abs(tip-displacement),"object_relative_rotation_rad":0.0,"two_finger_contact_end":False,"kind":"push","probe_mechanism":self.method_card["mechanism_id"],"probe_amplitude_scale":float(scale)}
        self.events.append(witness)
        return witness

    def probe_status(self,witness):
        if witness["actual_motion"]>=0.003:
            return "pass"
        if witness["actual_ee_motion_m"]>=0.003:
            return "negative"
        return "inconclusive"

    def recontact(self,phase="improve"):
        reference=self.source_fork
        target,rotation=self.reference_contact()
        target+=self.direction()*0.003
        grip=float(self.expert_actions[reference,6])
        elevated=max(float(self.eef()[2]),float(target[2]+0.025))
        self.move([*self.eef()[:2],elevated],rotation,grip,steps=7,phase=phase)
        self.move([*target[:2],elevated],rotation,grip,steps=9,phase=phase)
        self._move_plain(target,rotation,grip,steps=10,phase=phase)
        self.hold(target,rotation,grip,steps=4,phase=phase)

    def correct_and_continue(self,witness,attempt=0):
        start=len(self.actions)
        self.recontact()
        self.events.append({"method":"task_adjustment","start_step":start,"end_step":len(self.actions),"family":"push","attempt":attempt,"trigger":witness["result"],"physical_robot_actions":True})

    def continue_source(self):
        reference=self.source_fork
        grip=float(self.expert_actions[reference,6])
        for index in range(180):
            if self.env.check_success():
                return
            if self.remaining_budget()<30:
                raise UnsafeCandidate("推动与最终撤手超过五倍动作预算")
            target,rotation=self.reference_contact()
            target+=self.direction()*0.008
            self.execute(self.command_to(target,rotation,grip),"expert_suffix")
        raise UnsafeCandidate("盘子未到达原任务目标")
