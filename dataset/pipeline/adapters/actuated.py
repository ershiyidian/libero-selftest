from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/name) for name in ("pipeline","planner","physics")]
from real_task_support_v5 import RealTaskEpisode,UnsafeCandidate
from constrained_iterative import mjdata_from_flat,bodies_below


class Episode(RealTaskEpisode):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.joint is None:
            raise ValueError("机构执行器需要真实关节")
        self.fixture_body=int(self.model.jnt_bodyid[self.joint])
        self.fixture_bodies=bodies_below(self.model._model,self.fixture_body)
        self.fixture_source_xyz=[]
        self.fixture_source_rot=[]
        shadow=mujoco.MjData(self.model._model)
        for state in self.expert_states:
            mjdata_from_flat(self.model._model,state,shadow)
            self.fixture_source_xyz.append(shadow.xpos[self.fixture_body].copy())
            self.fixture_source_rot.append(shadow.xmat[self.fixture_body].reshape(3,3).copy())
        self.fixture_source_xyz=np.asarray(self.fixture_source_xyz)
        self.fixture_source_rot=np.asarray(self.fixture_source_rot)
        self.goal_q=float(self.expert_states[-1,1+int(self.model.jnt_qposadr[self.joint])])

    def fixture_contacts(self):
        total=0
        for contact in self.env.sim.data.contact[:self.env.sim.data.ncon]:
            bodies={int(self.model.geom_bodyid[contact.geom1]),int(self.model.geom_bodyid[contact.geom2])}
            if bodies & self.fixture_bodies and any((self.model.body_id2name(body) or "").startswith("gripper0_") for body in bodies):
                total+=1
        return total

    def movement_direction(self):
        index=self.source_fork
        future=min(index+12,len(self.source_xyz)-1)
        direction=self.source_xyz[future]-self.source_xyz[index]
        if np.linalg.norm(direction)<0.002:
            direction=self.source_xyz[-1]-self.source_xyz[index]
        magnitude=float(np.linalg.norm(direction))
        if magnitude<0.001:
            raise UnsafeCandidate("专家没有提供有效机构运动方向")
        return direction/magnitude

    def perform_probe(self,scale=1.0):
        start=len(self.actions)
        q_before=self.joint_pos()
        hand_before=self.eef().copy()
        rotation=self.rot().copy()
        grip=float(self.expert_actions[self.source_fork,6])
        direction=self.movement_direction()
        goal=hand_before+direction*(0.009*float(scale))
        contacts=[]
        for step in range(8):
            blend=(step+1)/8
            self.execute(self.command_to(hand_before+(goal-hand_before)*blend,rotation,grip),"probe")
            contacts.append(self.fixture_contacts())
        self.hold(goal,rotation,grip,steps=3,phase="probe")
        q_after=self.joint_pos()
        sign=np.sign(self.goal_q-q_before)
        progress=float((q_after-q_before)*sign)
        tip=float(np.dot(self.eef()-hand_before,direction))
        witness={
            "method":self.method_id,"method_title":self.method_card["title"],
            "start_step":start,"end_step":len(self.actions),"actual_steps":len(self.actions)-start,
            "actual_motion":progress,"actual_ee_motion_m":tip,
            "fixture_q_before":q_before,"fixture_q_after":q_after,
            "actual_contact_frames":sum(value>0 for value in contacts),
            "object_relative_shift_m":abs(tip-progress),"object_relative_rotation_rad":0.0,
            "two_finger_contact_end":self.fixture_contacts()>0,
            "probe_amplitude_scale":float(scale),"kind":self.family,
            "probe_mechanism":self.method_card["mechanism_id"],
        }
        self.events.append(witness)
        return witness

    def probe_status(self,witness):
        threshold=0.003 if self.family=="drawer" else 0.02
        if witness["actual_motion"]>=threshold and witness["two_finger_contact_end"]:
            return "pass"
        if witness["actual_ee_motion_m"]>=0.003 or witness["actual_contact_frames"]>=3:
            return "negative"
        return "inconclusive"

    def current_reference_grasp(self):
        index=self.source_fork
        source_rotation=self.fixture_source_rot[index]
        offset=source_rotation.T@(self.source_xyz[index]-self.fixture_source_xyz[index])
        rotation=source_rotation.T@self.source_rot[index]
        current_rotation=np.asarray(self.env.sim.data.body_xmat[self.fixture_body]).reshape(3,3)
        current_position=np.asarray(self.env.sim.data.body_xpos[self.fixture_body])
        return current_position+current_rotation@offset,current_rotation@rotation

    def correct_and_continue(self,witness,attempt=0):
        start=len(self.actions)
        grip=float(self.expert_actions[self.source_fork,6])
        direction=self.movement_direction()
        rotation=self.rot().copy()
        self._move_plain(self.eef()-direction*0.006,rotation,grip,steps=7,phase="improve")
        target,orientation=self.current_reference_grasp()
        self._move_plain(target,orientation,grip,steps=10,phase="improve")
        self.hold(target-direction*0.004,orientation,grip,steps=9,phase="improve")
        self.events.append({"method":"task_adjustment","start_step":start,"end_step":len(self.actions),"family":self.family,"attempt":attempt,"trigger":witness["result"],"actual_fixture_q":self.joint_pos(),"physical_robot_actions":True})

    def continue_source(self):
        direction=self.movement_direction()
        orientation=self.source_rot[self.source_fork]
        grip=float(self.expert_actions[self.source_fork,6])
        for index in range(180):
            if self.env.check_success() and abs(self.goal_q-self.joint_pos())<0.006:
                return
            if self.remaining_budget()<30:
                raise UnsafeCandidate("机构操作与撤手超过五倍动作预算")
            target,orientation=self.current_reference_grasp()
            # 以机构实际位置更新目标，跟随把手运动；每一步均保留碰撞检测。
            step=0.010 if self.family=="drawer" else 0.006
            self.execute(self.command_to(target+direction*step,orientation,grip),"expert_suffix")
        raise UnsafeCandidate("在可用控制步内机构没有到达原任务目标")

    def post_success_safety(self):
        if not self.env.check_success():
            return
        begin=len(self.actions)
        anchor=self.eef().copy()
        orientation=self.rot().copy()
        self.hold(grip=-1.0,steps=4,phase="release")
        self._move_plain(anchor+[0,0,0.035],orientation,-1.0,steps=10,phase="withdraw")
        self._move_plain(self.eef()+self.movement_direction()*0.025,orientation,-1.0,steps=8,phase="withdraw")
        self.hold(grip=-1.0,steps=12,phase="final_hold")
        self.events.append({"method":"post_task_release_observe","start_step":begin,"end_step":len(self.actions),"task_success_after_release":self.env.check_success(),"actual_withdrawal_m":float(np.linalg.norm(self.eef()-anchor))})
