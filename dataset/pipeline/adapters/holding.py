from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/name) for name in ("pipeline","planner","physics")]
from real_task_support_v5 import RealTaskEpisode,UnsafeCandidate
from constrained_iterative import mjdata_from_flat


class Episode(RealTaskEpisode):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.family!="support" or self.target_body_id is None:
            raise ValueError("持物执行器需要一个真实的任务目标物体")
        shadow=mujoco.MjData(self.model._model)
        positions=[]
        rotations=[]
        for state in self.expert_states:
            mjdata_from_flat(self.model._model,state,shadow)
            positions.append(shadow.xpos[self.target_body_id].copy())
            rotations.append(shadow.xmat[self.target_body_id].reshape(3,3).copy())
        self.source_object_xyz=np.asarray(positions)
        self.source_object_rot=np.asarray(rotations)
        self.corrected_grasp=False
        self.approach_history=[]
        self.source_support_geoms=set()
        mjdata_from_flat(self.model._model,self.expert_states[0],shadow)
        for contact in shadow.contact[:shadow.ncon]:
            first=int(self.model.geom_bodyid[contact.geom1])
            second=int(self.model.geom_bodyid[contact.geom2])
            if first==self.target_body_id and contact.frame[2]<-0.65:
                self.source_support_geoms.add(int(contact.geom2))
            elif second==self.target_body_id and contact.frame[2]>0.65:
                self.source_support_geoms.add(int(contact.geom1))
        if not self.source_support_geoms:
            raise ValueError("原始物体没有可验证的起始支撑")

    def execute_expert_prefix(self):
        changes=[item for item in self.condition_parameters["mutations"] if item["kind"]=="approach_offset"]
        if not changes:
            return super().execute_expert_prefix()
        offset=np.asarray(changes[0]["offset_m"],dtype=float)
        closes=np.flatnonzero((self.expert_actions[1:,6]>0)&(self.expert_actions[:-1,6]<=0))+1
        closes=closes[closes<self.source_fork]
        if not len(closes):
            raise ValueError("找不到原专家的实际首次闭合过程")
        close=int(closes[-1])
        begin=max(0,close-10)
        for index in range(self.source_fork):
            action=self.expert_actions[index].copy()
            if index<begin:
                self.execute(action,"expert_prefix")
                continue
            weight=min(1.0,(index-begin+1)/6)
            reference=min(index+1,len(self.source_xyz)-1)
            action=self.command_to(self.source_xyz[reference]+offset*weight,self.source_rot[reference],action[6])
            self.execute(action,"expert_prefix")
            self.approach_history.append({"source_step":index,"requested_offset_m":(offset*weight).tolist(),"actual_ee_m":self.eef().tolist()})
        self.events.append({
            "method":"condition_precontact_offset","start_step":begin,
            "end_step":len(self.actions),"offset_m":offset.tolist(),
            "actual_control_steps":len(self.approach_history),
            "source_contact_step":close,"physical_robot_actions":True,
        })

    def perform_probe(self,scale=1.0):
        start=len(self.actions)
        before_pos=self.target_pos().copy()
        initial_relative,initial_rotation=self.relative_object_pose()
        anchor=self.eef().copy()
        orientation=self.rot().copy()
        grip=float(self.actions[-1][6])
        # 即使夹持没有成立，也真实执行低高度试抬来观察物体是否跟随。
        displacement=np.array([0.0,0.0,0.006*scale])
        self._move_plain(anchor+displacement,orientation,grip,steps=8,phase="probe")
        self.hold(steps=4,phase="probe")
        actual_hand=float(self.eef()[2]-anchor[2])
        actual_object=float(self.target_pos()[2]-before_pos[2])
        if self.method_card["mechanism_id"]=="task_turn":
            turned=Rotation.from_rotvec([0.0,0.0,0.055*scale]).as_matrix()@orientation
            self._move_plain(self.eef(),turned,grip,steps=7,phase="probe")
            self.hold(steps=3,phase="probe")
        self._move_plain(anchor,orientation,grip,steps=8,phase="probe")
        self.hold(anchor,orientation,grip,steps=3,phase="probe")
        records=self.records[start:]
        relative=np.asarray([r["eef_rot"].T@(r["target"]-r["eef"]) for r in records])
        relative_rot=np.asarray([r["eef_rot"].T@r["object_rot"] for r in records])
        shifts=np.linalg.norm(relative-relative[0],axis=1)
        turns=Rotation.from_matrix(relative_rot@relative_rot[0].T).magnitude()
        maximum_shift=float(shifts.max())
        maximum_rotation=float(turns.max())
        contact_fraction=float(np.mean([r["contact_both"] for r in records]))
        witness={
            "method":self.method_id,"method_title":self.method_card["title"],
            "start_step":start,"end_step":len(self.actions),
            "actual_steps":len(self.actions)-start,"actual_motion":actual_object,
            "object_relative_shift_m":maximum_shift,
            "object_relative_rotation_rad":maximum_rotation,
            "two_finger_contact_end":self.has_both_contacts(),
            "bilateral_contact_fraction":contact_fraction,
            "actual_ee_lift_m":actual_hand,"actual_object_lift_m":actual_object,
            "probe_amplitude_scale":float(scale),"kind":self.family,
            "probe_mechanism":"grasp_load_check",
            "end_observable":{"image_available":True,"proprio_available":True},
        }
        self.events.append(witness)
        return witness

    def probe_status(self,witness):
        motion=witness["actual_ee_lift_m"]
        following=witness["actual_object_lift_m"]
        if motion<0.002:
            return "inconclusive"
        if (not witness["two_finger_contact_end"] or
            following<0.45*motion or witness["object_relative_shift_m"]>0.006 or
            witness["object_relative_rotation_rad"]>0.16):
            return "negative"
        if following<0.7*motion or witness["object_relative_shift_m"]>0.0035:
            return "inconclusive"
        return "pass"

    def source_support(self):
        raw=self.env.sim.data._data
        total=0.0
        for index,contact in enumerate(raw.contact[:raw.ncon]):
            first=int(self.model.geom_bodyid[contact.geom1])
            second=int(self.model.geom_bodyid[contact.geom2])
            normal=float(contact.frame[2])
            valid=(first==self.target_body_id and int(contact.geom2) in self.source_support_geoms and normal<-0.65) or (second==self.target_body_id and int(contact.geom1) in self.source_support_geoms and normal>0.65)
            if valid:
                force=np.zeros(6)
                mujoco.mj_contactForce(self.model._model,raw,index,force)
                total+=abs(float(force[0]))
        return total>0.025 and abs(float(raw.cvel[self.target_body_id,5]))<0.05

    def correct_and_continue(self,witness,attempt=0):
        start=len(self.actions)
        orientation=self.rot().copy()
        if not self.source_support():
            anchor=self.eef().copy()
            for index in range(35):
                if self.source_support():
                    break
                target=anchor-np.array([0.0,0.0,0.0015*(index+1)])
                self.execute(self.command_to(target,orientation,1.0),"improve")
            if not self.source_support():
                raise UnsafeCandidate("重抓前无法形成原始承载面的真实支撑")
        supported=[]
        for _ in range(4):
            self.hold(steps=1,phase="improve")
            supported.append(self.source_support())
        if not all(supported):
            raise UnsafeCandidate("重抓松手前的支撑不连续")
        self.hold(grip=-1.0,steps=7,phase="improve")
        self._move_plain(self.eef()+[0,0,0.025],orientation,-1.0,steps=10,phase="improve")
        reference=self.source_fork
        source_rotation=self.source_object_rot[reference]
        offset=source_rotation.T@(self.source_xyz[reference]-self.source_object_xyz[reference])
        rotation=source_rotation.T@self.source_rot[reference]
        target_rotation=self.target_rot().copy()
        grasp=self.target_pos()+target_rotation@offset
        grasp_rotation=target_rotation@rotation
        # 根据最新物体位置重建专家抓取，已执行过的偏移不会重复应用。
        self.move(grasp+[0,0,0.025],grasp_rotation,-1.0,steps=10,phase="improve")
        self._move_plain(grasp,grasp_rotation,-1.0,steps=12,phase="improve")
        self.hold(grasp,grasp_rotation,grip=1.0,steps=9,phase="improve")
        if not self.has_both_contacts():
            self.events.append({"method":"supported_regrasp","start_step":start,"end_step":len(self.actions),"result":"contact_requires_retest","attempt":attempt})
            return
        self._move_plain(self.eef()+[0,0,0.012],grasp_rotation,1.0,steps=9,phase="improve")
        self.corrected_grasp=True
        self.events.append({
            "method":"supported_regrasp","start_step":start,"end_step":len(self.actions),
            "actual_grasp_m":grasp.tolist(),"support_confirmed_frames":4,
            "two_finger_contact":self.has_both_contacts(),"attempt":attempt,
            "physical_robot_actions":True,
        })

    def continue_source(self):
        releases=np.flatnonzero(self.expert_actions[self.source_fork:,6]<0)
        end=self.source_fork+int(releases[0]) if len(releases) else len(self.source_xyz)-1
        indices=list(range(self.source_fork,end,4))+[end]
        for index in indices:
            self.move_object(self.source_object_xyz[index],self.source_object_rot[index],phase="expert_suffix",max_steps=35)
        self.guided_support_transfer()

    def move_object(self,position,orientation,phase="expert_suffix",max_steps=220):
        position=np.asarray(position,dtype=float)
        for _ in range(max_steps):
            current=self.target_pos()
            current_rotation=self.target_rot()
            delta=position-current
            angular=Rotation.from_matrix(orientation@current_rotation.T).as_rotvec()
            distance=float(np.linalg.norm(delta));angle=float(np.linalg.norm(angular))
            position_tolerance=0.006 if phase=="expert_suffix" else 0.003
            rotation_tolerance=0.09 if phase=="expert_suffix" else 0.05
            if distance<position_tolerance and angle<rotation_tolerance:
                return
            if self.remaining_budget()<42:
                raise UnsafeCandidate("物体相对轨迹需要更多控制步，无法保留完整放置预算")
            object_step=current+delta*min(1.0,0.004/max(distance,1e-9))
            object_rotation=Rotation.from_rotvec(angular*min(1.0,0.045/max(angle,1e-9))).as_matrix()@current_rotation
            relative_position=current_rotation.T@(self.eef()-current)
            relative_rotation=current_rotation.T@self.rot()
            hand=object_step+object_rotation@relative_position
            hand_rotation=object_rotation@relative_rotation
            self.execute(self.command_to(hand,hand_rotation,1.0),phase)
        raise UnsafeCandidate(f"物体参考路径没有到达：位置误差{distance*1000:.2f}毫米，角度误差{np.degrees(angle):.2f}度，阶段{phase}")

    def guided_support_transfer(self):
        start=len(self.actions)
        goal=self.source_object_xyz[-1].copy()
        orientation=self.source_object_rot[-1].copy()
        near=goal.copy();near[2]+=0.012
        self.move_object(near,orientation,phase="support_seek")
        stable=0
        for _ in range(80):
            if self.goal_support_stable():
                self.hold(grip=1.0,steps=1,phase="support_settle")
                stable+=1
                if stable>=4:
                    break
            else:
                stable=0
                self.execute(self.command_to(self.eef()-[0,0,0.0015],self.rot(),1.0),"support_seek")
        if stable<4:
            raise UnsafeCandidate("实际承载面上没有形成连续支撑")
        before=self.support_transfer_snapshot()
        release_begin=len(self.actions)
        anchor=self.eef().copy()
        hand_rotation=self.rot().copy()
        completed=False
        for _ in range(48):
            opening=float(self.env.robots[0].gripper.current_action[0])
            if opening>=0.99 and not self.has_both_contacts() and self.eef()[2]-anchor[2]>0.012:
                completed=True
                break
            if self.remaining_budget()<28:
                raise UnsafeCandidate("逐步卸载与撤手超过完整动作预算")
            choices=[]
            for grip in (-1.0,0.0):
                for rise in (0.0,0.002,0.006,0.012):
                    target=self.eef()+[0,0,rise]
                    command=self.command_to(target,hand_rotation,grip).astype(np.float32)
                    predicted=self.predictive_control.predict(command,"supported_release")
                    if predicted["risk"]>1.0 or predicted["minimum_goal_support_force_n"]<0.015:
                        continue
                    next_open=float(predicted["gripper_action"][0])
                    height=float(predicted["eef_position"][2]-anchor[2])
                    desired_height=max(0.0,(next_open+1)/2)*0.020
                    score=next_open-15*abs(height-desired_height)
                    choices.append((score,command))
            if not choices:
                raise UnsafeCandidate("没有找到保持实际支撑的安全张爪与退出动作")
            _,command=max(choices,key=lambda item:item[0])
            self.execute(command,"supported_release")
        if not completed:
            raise UnsafeCandidate("夹爪没有在受支撑状态下完成打开与分离")
        self.events.append({"method":"goal_supported_transfer","start_step":start,"end_step":len(self.actions),"release_begin_step":release_begin,"contact_confirmed_before_open_step":release_begin,"staged_release_actual_actions":len(self.actions)-release_begin,"support_pre_release":before,"result":"pass"})

    def post_success_safety(self):
        if not self.env.check_success():
            return
        start=len(self.actions)
        anchor=self.eef().copy()
        self._move_plain(anchor+[0,0,0.045],self.rot(),-1.0,steps=14,phase="withdraw")
        self.hold(grip=-1.0,steps=12,phase="final_hold")
        self.events.append({"method":"post_task_release_observe","start_step":start,"end_step":len(self.actions),"task_success_after_release":self.env.check_success(),"actual_withdrawal_m":float(np.linalg.norm(self.eef()-anchor))})
