import os
import sys
import json
import math
from math import pi
from dotenv import load_dotenv
import pybullet as physics
import HarfangHighLevel as hl
from openai import OpenAI

# ------------------------------
# Small utilities
# ------------------------------

def clamp(v, vmin, vmax): return max(vmin, min(v, vmax))
def deg(x): return x * 180.0 / math.pi
def rad(x): return x * math.pi / 180.0


#OPENAI_API_KEY=sk-projectXXXX... in .env file (command for creating .env-->nano .env)

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')


SYSTEM_PROMPT = (
    "You control a simulated robot 'Reachy' (PyBullet + Harfang). "
    "Your job is to convert the USER command into a JSON **array** of actions. "
    "Return ONLY JSON. No prose. If no action, return [].\n\n"
    "Actions you may emit:\n"
    "  - {\"type\":\"set_target\",\"args\":{\"hand\":\"left|right\",\"x\":float,\"y\":float,\"z\":float}}\n"
    "  - {\"type\":\"delta_target\",\"args\":{\"hand\":\"left|right\",\"dx\":float,\"dy\":float,\"dz\":float}}\n"
    "  - {\"type\":\"set_wrist\",\"args\":{\"hand\":\"left|right\",\"pitch\":float_degrees|null,\"roll\":float_degrees|null}}\n"
    "  - {\"type\":\"set_gripper\",\"args\":{\"hand\":\"left|right\",\"value\":float_0_to_1}}    # 0=open, 1=closed\n"
    "  - {\"type\":\"calibrate\",\"args\":{}}\n"
    "  - {\"type\":\"swap_hands\",\"args\":{}}\n"
    "  - {\"type\":\"set_keyboard_override\",\"args\":{\"enabled\":true|false}}\n"
    "  - {\"type\":\"set_joint_offset\",\"args\":{\"name\":str,\"value_deg\":float}}\n"
    "Constraints:\n"
    "- Coordinates (x,y,z) are in meters in the Harfang scene frame.\n"
    "- Keep values reasonable: wrist pitch∈[-30°, +34°], wrist roll∈[-29°, +29°] approx.\n"
    "- If the command is ambiguous (e.g., no hand specified), prefer the right hand.\n"
    "- NEVER include text outside JSON."
)

class LLMBridge:
    def __init__(self):
        self.enabled = bool(os.environ.get("OPENAI_API_KEY")) and (OpenAI is not None)
        self.client = OpenAI() if self.enabled else None

    def parse(self, text: str):
        """
        Returns a list[dict] of actions parsed from text.
        Will fall back to a small rule-based parser if API is not available.
        """
        t = (text or "").strip()
        if not t:
            return []

        if self.enabled:
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": t},
                    ],
                    response_format={"type": "json_object"}  # ensures JSON
                )
                content = resp.choices[0].message.content or "[]"

            
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    if "actions" in parsed and isinstance(parsed["actions"], list):
                        return parsed["actions"]
                    # Or a single action
                    return [parsed]
                if isinstance(parsed, list):
                    return parsed
                return []
            except Exception as e:
                print("[LLM] OpenAI call failed, falling back:", e)

        return self._fallback_rules(t)

    # ---------- tiny offline fallback so the app still works ----------

    @staticmethod
    def _is_float(s):
        try:
            float(s); return True
        except:
            return False

    @staticmethod
    def _last_float(t):
        toks = t.replace(",", " ").split()
        floats = []
        for tok in toks:
            try:
                floats.append(float(tok))
            except:
                pass
        return floats[-1] if floats else None

    @staticmethod
    def _percent_or_default(t, default):
        # finds number optionally followed by %
        raw = t.replace("%", " % ").split()
        vals = []
        for i, tok in enumerate(raw):
            try:
                f = float(tok)
                if i + 1 < len(raw) and raw[i+1] == "%":
                    vals.append(clamp(f/100.0, 0.0, 1.0))
                else:
                    vals.append(f)
            except:
                pass
        return vals[-1] if vals else default

    def _fallback_rules(self, t: str):
        text = t.lower()
        actions = []

        if ("move" in text or "set" in text) and "hand" in text and "to" in text:
            hand = "right" if "right" in text else ("left" if "left" in text else "right")
            nums = [float(s) for s in text.replace(",", " ").split() if self._is_float(s)]
            if len(nums) >= 3:
                x, y, z = nums[-3:]
                actions.append({"type":"set_target","args":{"hand":hand,"x":x,"y":y,"z":z}})

        dirs = {"up":(0,+1,0), "down":(0,-1,0), "left":(0,0,+1), "right":(0,0,-1), "forward":(+1,0,0), "back":(-1,0,0)}
        for k, vec in dirs.items():
            if k in text:
                mag = self._last_float(text) or 0.05
                dx, dy, dz = (v*mag for v in vec)
                hand = "right" if "right" in text else ("left" if "left" in text else "right")
                actions.append({"type":"delta_target","args":{"hand":hand,"dx":dx,"dy":dy,"dz":dz}})

        if "wrist" in text:
            hand = "right" if "right" in text else ("left" if "left" in text else "right")
            pitch = None; roll = None
            if "pitch" in text: pitch = self._last_float(text) or 0.0
            if "roll"  in text: roll  = self._last_float(text) or 0.0
            if pitch is not None or roll is not None:
                actions.append({"type":"set_wrist","args":{"hand":hand,"pitch":pitch,"roll":roll}})

        if "gripper" in text:
            hand = "right" if "right" in text else ("left" if "left" in text else "right")
            if "open" in text:
                v = self._percent_or_default(text, 0.0)
            elif "close" in text:
                v = self._percent_or_default(text, 1.0)
            else:
                v = self._percent_or_default(text, 0.5)
            actions.append({"type":"set_gripper","args":{"hand":hand,"value":clamp(v,0.0,1.0)}})

        if "calibrate" in text:
            actions.append({"type":"calibrate","args":{}})

        if "swap" in text and "hand" in text:
            actions.append({"type":"swap_hands","args":{}})

        if "keyboard" in text:
            enabled = ("on" in text) or ("enable" in text) or ("true" in text)
            if ("off" in text) or ("disable" in text) or ("false" in text):
                enabled = False
            actions.append({"type":"set_keyboard_override","args":{"enabled":enabled}})

        return actions


# ------------------------------
# Digital twin app
# ------------------------------

class ReachySim:
    reachy_path = "models/reachy2.URDF"

    def __init__(self):
        # --- Bullet ---
        physics.connect(physics.DIRECT)
        physics.setPhysicsEngineParameter(enableConeFriction=0)
        physics.setAdditionalSearchPath(".")

        self.reachyId = physics.loadURDF(self.reachy_path, useFixedBase=True)
        self.numJoints = physics.getNumJoints(self.reachyId)
        for j in range(self.numJoints):
            physics.resetJointState(self.reachyId, j, 0.0)

        self.right_ee = 8
        self.left_ee  = 17

        # Robust qIndex mapping for IK results
        self.movable = []
        for i in range(self.numJoints):
            jinfo = physics.getJointInfo(self.reachyId, i)
            jtype, qidx = jinfo[2], jinfo[3]
            if jtype in (physics.JOINT_REVOLUTE, physics.JOINT_PRISMATIC) and qidx > -1:
                self.movable.append((i, qidx))
        self.movable.sort(key=lambda t: t[1])
        self.idx_by_joint = {i: k for k, (i, _) in enumerate(self.movable)}

        # --- Harfang / Scene ---
        self.want_vr = True
        for a in sys.argv[1:]:
            if a == "no_vr":
                self.want_vr = False

        hl.Init(1920, 1080, self.want_vr)
        hl.LoadSceneFromAssets("reachy.scn", hl.gVal.scene, hl.gVal.res, hl.GetForwardPipelineInfo())
        hl.AddFpsCamera(2, 1.5, 0, pi/8, -pi/2)
        hl.gVal.scene.SetCurrentCamera(hl.gVal.camera)

        self.reachy_3D_link_to_join = {
            "pedestal":"chest",
            "r_shoulder_pitch":"right_shoulder_pitch_joint",
            "r_shoulder_roll":"right_shoulder_roll_joint",
            "r_arm_yaw":"right_arm_yaw_joint",
            "r_elbow_pitch":"right_elbow_pitch_joint",
            "r_forearm_yaw":"right_forearm_yaw_joint",
            "r_wrist_pitch":"right_wrist_pitch_joint",
            "r_wrist_roll":"right_wrist_roll_joint",
            "r_gripper":"right_gripper_joint",
            "l_shoulder_pitch":"left_shoulder_pitch_joint",
            "l_shoulder_roll":"left_shoulder_roll_joint",
            "l_arm_yaw":"left_arm_yaw_joint",
            "l_elbow_pitch":"left_elbow_pitch_joint",
            "l_forearm_yaw":"left_forearm_yaw_joint",
            "l_wrist_pitch":"left_wrist_pitch_joint",
            "l_wrist_roll":"left_wrist_roll_joint",
            "l_gripper":"left_gripper_joint",
        }

        self.joint_node_offset = {
            "pedestal": {"r":-pi/2.0, "inv":False},
            "r_shoulder_pitch": {"r":pi/2.0, "inv":False},
            "r_shoulder_roll": {"r":pi/2.0, "inv":False},
            "r_arm_yaw":{"r":0, "inv":False},
            "r_elbow_pitch":{"r":0, "inv":False},
            "r_forearm_yaw":{"r":0, "inv":False},
            "r_wrist_pitch":{"r":0, "inv":False},
            "r_wrist_roll":{"r":0, "inv":False},
            "r_gripper":{"r":0, "inv":False},
            "l_shoulder_pitch": {"r":pi/2.0, "inv":True},
            "l_shoulder_roll": {"r":pi/2.0, "inv":True},
            "l_arm_yaw": {"r":pi, "inv":False},
            "l_elbow_pitch":{"r":0, "inv":False},
            "l_forearm_yaw":{"r":0, "inv":False},
            "l_wrist_pitch":{"r":0, "inv":False},
            "l_wrist_roll":{"r":0, "inv":False},
            "l_gripper":{"r":0, "inv":False},
        }

        # Cache Harfang nodes
        self.node_by_joint = {}
        for i in range(self.numJoints):
            jname = physics.getJointInfo(self.reachyId, i)[1].decode()
            if jname in self.reachy_3D_link_to_join:
                node = hl.gVal.scene.GetNode(self.reachy_3D_link_to_join[jname])
                if node != hl.NullNode:
                    self.node_by_joint[jname] = node

        # VR cal UI
        self.reachy_pos = hl.Vec3(0, 1, 0)
        self.calib_timer = 2.0
        self.calib_accum = 0.0
        self.calib_node = hl.gVal.scene.GetNode("calibration_gauge_container")
        self.calib_max_scale = self.calib_node.GetTransform().GetScale().x
        self.calib_node.GetTransform().SetScale(hl.Vec3(-1,1,1))

        # IK targets (Harfang axes)
        self.target_right = hl.Vec3(0.062, 0.625, -0.131)
        self.target_left  = hl.Vec3(0.062, 0.625,  0.131)

        # Wrist/gripper logical state
        self.wrist = {"right": {"pitch": 0.0, "roll": 0.0}, "left": {"pitch": 0.0, "roll": 0.0}}
        self.gripper = {"right": 0.5, "left": 0.5}  # 0=open, 1=closed

        self.keyboard_override = True
        self.flag_head_hidden = False

        # LLM command console
        self.llm = LLMBridge()
        self.command_buffer = ""

    # ------------------ UI ------------------

    def imgui(self):
        if hl.ImGuiBegin("Digital Twin (LLM console)"):
            hl.ImGuiText("Examples:")
            hl.ImGuiText("- move right hand to 0.12 0.63 -0.10")
            hl.ImGuiText("- open left gripper 30%")
            hl.ImGuiText("- rotate right wrist roll 10 deg")
            changed, self.command_buffer = hl.ImGuiInputText("Command", self.command_buffer)
            if hl.ImGuiButton("Send"):
                self.apply_text_command(self.command_buffer)
                self.command_buffer = ""
            if not self.llm.enabled:
                hl.ImGuiTextColored(hl.Color.Yellow, "Tip: set OPENAI_API_KEY to enable LLM (fallback parser active)")
        hl.ImGuiEnd()

    def apply_text_command(self, text):
        actions = self.llm.parse(text)
        for a in actions:
            typ = a.get("type")
            args = a.get("args", {})
            try:
                if   typ == "set_target": self.act_set_target(**args)
                elif typ == "delta_target": self.act_delta_target(**args)
                elif typ == "set_wrist": self.act_set_wrist(**args)
                elif typ == "set_gripper": self.act_set_gripper(**args)
                elif typ == "calibrate": self._start_calibration()
                elif typ == "swap_hands": pass  # stub, hook in VR controller swap if you like
                elif typ == "set_keyboard_override": self.keyboard_override = bool(args.get("enabled", True))
                elif typ == "set_joint_offset":
                    name = args.get("name"); val = args.get("value_deg")
                    if name in self.joint_node_offset and val is not None:
                        self.joint_node_offset[name]["r"] = rad(float(val))
            except Exception as e:
                print("[LLM] action failed:", a, e)

    # ------------------ Actions ------------------

    def act_set_target(self, hand, x, y, z):
        v = hl.Vec3(float(x), float(y), float(z))
        if hand == "right": self.target_right = v
        else:               self.target_left  = v

    def act_delta_target(self, hand, dx, dy, dz):
        v = hl.Vec3(float(dx), float(dy), float(dz))
        if hand == "right": self.target_right = self.target_right + v
        else:               self.target_left  = self.target_left + v

    def act_set_wrist(self, hand, pitch=None, roll=None):
        W = self.wrist["right" if hand == "right" else "left"]
        if pitch is not None: W["pitch"] = clamp(rad(float(pitch)), -0.5, 0.6)
        if roll  is not None: W["roll"]  = clamp(rad(float(roll)),  -0.5, 0.5)

    def act_set_gripper(self, hand, value):
        self.gripper["right" if hand == "right" else "left"] = clamp(float(value), 0.0, 1.0)

    # ------------------ Calibration / head visibility ------------------

    def _start_calibration(self): self.calib_accum = 0.0001

    def _update_calibration(self):
        flag = bool(self.calib_accum) or hl.gVal.keyboard.Down(hl.K_Space)
        if flag:
            dt = hl.TickClock()
            dts = hl.time_to_sec_f(dt)
            f = self.calib_accum / self.calib_timer
            self.calib_node.GetTransform().SetScale(hl.Vec3(clamp(f,0,1)*self.calib_max_scale, 1, 1))
            self.calib_accum += dts
            if self.calib_accum >= self.calib_timer:
                if self.want_vr: self._calibrate_vr_head()
                else:            self._calibrate_camera()
                self.calib_accum = 0.0
        else:
            self.calib_node.GetTransform().SetScale(hl.Vec3(0,1,1))
            self.calib_accum = 0.0

    def _calibrate_vr_head(self):
        actual = hl.GetTranslation(hl.gVal.vr_state.head)
        ideal  = hl.GetT(hl.gVal.scene.GetNode("front_head").GetTransform().GetWorld())
        off = ideal - actual
        body_mtx = hl.gVal.ground_vr_mat + hl.TranslationMat4(off)
        body_pos = hl.GetT(body_mtx)
        hl.SetVRGroundAnchor(body_pos.x, body_pos.y, body_pos.z, angle_y=0)
        self._hide_reachy_head()

    def _calibrate_camera(self):
        ideal = hl.GetT(hl.gVal.scene.GetNode("front_head").GetTransform().GetWorld())
        hl.gVal.camera.GetTransform().SetPos(ideal)
        hl.gVal.camera.GetTransform().SetRot(hl.Vec3(0, pi/2.0, 0))

    def _hide_reachy_head(self):
        self._toggle_children("neck_interface", enable=False)

    def _show_reachy_head(self):
        self._toggle_children("neck_interface", enable=True)

    def _toggle_children(self, root_name, enable=True):
        root = hl.gVal.scene.GetNode(root_name)
        def rec(n):
            for c in hl.gVal.scene.GetNodeChildren(n):
                if c.HasObject():
                    (c.Enable() if enable else c.Disable())
                rec(c)
        rec(root)

    # ------------------ Visual & motors ------------------

    def _apply_visuals_special(self):
        # write wrist angles into visual offsets
        self.joint_node_offset["r_wrist_pitch"]["r"] = self.wrist["right"]["pitch"]
        self.joint_node_offset["r_wrist_roll"]["r"]  = self.wrist["right"]["roll"]
        self.joint_node_offset["l_wrist_pitch"]["r"] = self.wrist["left"]["pitch"]
        self.joint_node_offset["l_wrist_roll"]["r"]  = self.wrist["left"]["roll"]

        # gripper mapping (normalize 0..1 to your asymmetric ranges)
        r_open, r_close = -0.290, 0.5
        l_open, l_close = -0.5,   0.290
        self.joint_node_offset["r_gripper"]["r"] = r_open + (r_close - r_open) * self.gripper["right"]
        self.joint_node_offset["l_gripper"]["r"] = l_open + (l_close - l_open) * self.gripper["left"]

    def _apply_motor_overrides(self):
        # Keep Bullet in sync with visuals for wrist/gripper
        def set_motor(name, angle):
            for i in range(self.numJoints):
                if physics.getJointInfo(self.reachyId, i)[1].decode() == name:
                    physics.setJointMotorControl2(self.reachyId, i, physics.POSITION_CONTROL, targetPosition=angle)
                    return
        set_motor("r_wrist_pitch", self.joint_node_offset["r_wrist_pitch"]["r"])
        set_motor("r_wrist_roll",  self.joint_node_offset["r_wrist_roll"]["r"])
        set_motor("l_wrist_pitch", self.joint_node_offset["l_wrist_pitch"]["r"])
        set_motor("l_wrist_roll",  self.joint_node_offset["l_wrist_roll"]["r"])
        set_motor("r_gripper",     self.joint_node_offset["r_gripper"]["r"])
        set_motor("l_gripper",     self.joint_node_offset["l_gripper"]["r"])

    # ------------------ Main loop ------------------

    def run(self):
        while not hl.UpdateDraw():
            physics.stepSimulation()
            self.imgui()

            # calibration & head visibility
            self._update_calibration()
            dist = hl.Dist(
                self.reachy_pos,
                hl.GetT(hl.GetCameraMat4() if not self.want_vr else hl.gVal.vr_state.head)
            )
            if dist < 0.25 and not self.flag_head_hidden:
                self._hide_reachy_head(); self.flag_head_hidden = True
            elif dist > 0.30 and self.flag_head_hidden:
                self._show_reachy_head(); self.flag_head_hidden = False

            # Update special joint visuals from logical state
            self._apply_visuals_special()

            # Compute IK (Harfang->Bullet axes swap: [x,z,y])
            target_r = [self.target_right.x, self.target_right.z, self.target_right.y]
            target_l = [self.target_left.x,  self.target_left.z,  self.target_left.y]
            jointPoses = physics.calculateInverseKinematics2(
                self.reachyId,
                [self.right_ee, self.left_ee],
                [target_r, target_l]
            )

            # Drive motors from IK using robust mapping
            for jIndex, _ in self.movable:
                k = self.idx_by_joint[jIndex]
                physics.setJointMotorControl2(
                    bodyIndex=self.reachyId, jointIndex=jIndex,
                    controlMode=physics.POSITION_CONTROL,
                    targetPosition=jointPoses[k]
                )

            # Keep wrists/grippers in sync with visuals (LLM/keyboard)
            self._apply_motor_overrides()

            # Update Harfang nodes from Bullet (+ offsets)
            joint_states = physics.getJointStates(self.reachyId, jointIndices=list(range(self.numJoints)))
            for j in range(self.numJoints):
                jname = physics.getJointInfo(self.reachyId, j)[1].decode()
                node = self.node_by_joint.get(jname)
                if not node:
                    continue
                rot = joint_states[j][0]
                off = self.joint_node_offset[jname]["r"]
                inv = self.joint_node_offset[jname]["inv"]
                value = (-rot + off) if inv else (rot + off)
                node.GetTransform().SetRot(hl.Vec3(0, value, 0))

            # Draw target crosses
            hl.DrawCrossV(self.target_right, hl.Color.Purple, size=0.05)
            hl.DrawCrossV(self.target_left,  hl.Color.Green,  size=0.05)

        hl.Uninit()


def main(argv):
    app = ReachySim()
    app.run()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
