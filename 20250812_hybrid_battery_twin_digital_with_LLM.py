import os
import json
import numpy as np
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from OpenAI import OpenAI

#OPENAI_API_KEY=sk-projectXXXX... in .env file (command for creating .env-->nano .env)

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')


# -------------------------
# Simple physics model
# -------------------------
@dataclass
class PhysicsModel:
    k: float = 0.0001  # degradation constant

    def predict(self, temp_c, cycles, charge_time_h, c0=1.0):
        """
        Simple exponential decay: C(t) = C0 * exp(-fd)
        fd = k * T * cycles / charge_time
        """
        fd = self.k * temp_c * cycles / max(charge_time_h, 1e-6)
        return c0 * np.exp(-fd)

# -------------------------
# Simple ML residual model
# -------------------------
@dataclass
class MLModel:
    coef: float = 0.02  # simple linear residual coeff

    def predict_residual(self, temp_c, cycles):
        # Toy example: residual is linear in cycles and temp
        return self.coef * np.sin(cycles / 100.0) * (temp_c - 25) / 100.0

# -------------------------
# Hybrid Digital Twin
# -------------------------
@dataclass
class HybridDigitalTwin:
    physics: PhysicsModel
    ml: MLModel
    c0: float = 1.0  # initial capacity

    def predict(self, temp_c, cycles, charge_time_h):
        phys_pred = self.physics.predict(temp_c, cycles, charge_time_h, self.c0)
        resid = self.ml.predict_residual(temp_c, cycles)
        return max(phys_pred + resid, 0.0)  # ensure non-negative

# -------------------------
# LLM interface
# -------------------------
SYSTEM_PROMPT = """
You are an assistant controlling a Hybrid Digital Twin for Li-ion batteries.
The user gives natural-language commands; you respond ONLY with a JSON array of actions.
Actions you may output:
- {"type":"set_params","args":{"temp_c":float,"cycles":int,"charge_time_h":float}}
- {"type":"predict_capacity","args":{}}
- {"type":"set_initial_capacity","args":{"c0":float}}
- {"type":"show_config","args":{}}
Never output explanations. Only valid JSON.
"""


class BatteryLLM:
    def __init__(self):
        # Enable if API key is present and openai SDK is available
        self.enabled = bool(api_key) and (OpenAI is not None)
        self.client = OpenAI(api_key=api_key) if self.enabled else None

    def parse(self, text):
        text = (text or "").strip()
        if not text:
            return []

        if self.enabled:
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                parsed = json.loads(content)

                # Ensure parsed is always a list of actions
                if isinstance(parsed, dict):
                    return [parsed]
                return parsed
            except Exception as e:
                print("[LLM] Error, falling back:", e)

        # If LLM is disabled or failed, use fallback parser
        return self._fallback(text)

    def _fallback(self, text):
        # very simple pattern match fallback
        t = text.lower()
        actions = []
        if "set" in t and "temp" in t:
            try:
                temp = float(t.split("temp")[1].split()[0])
                actions.append(
                    {
                        "type": "set_params",
                        "args": {"temp_c": temp, "cycles": 0, "charge_time_h": 1.0},
                    }
                )
            except:
                pass
        if "predict" in t:
            actions.append({"type": "predict_capacity", "args": {}})
        return actions

# -------------------------
# Main interactive loop
# -------------------------
class BatterySimApp:
    def __init__(self):
        self.twin = HybridDigitalTwin(PhysicsModel(), MLModel())
        self.temp_c = 25.0
        self.cycles = 0
        self.charge_time_h = 1.0
        self.llm = BatteryLLM()

    def run(self):
        print("Hybrid Digital Twin for Li-ion Batteries (LLM interface)")
        print("Type commands like: 'simulate at 30 C, 500 cycles, 2 h charge', 'predict capacity after 200 cycles'")
        while True:
            try:
                cmd = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd.strip():
                continue
            actions = self.llm.parse(cmd)
            for a in actions:
                self._apply_action(a)

    def _apply_action(self, action):
        typ = action.get("type")
        args = action.get("args", {})
        if typ == "set_params":
            self.temp_c = args.get("temp_c", self.temp_c)
            self.cycles = args.get("cycles", self.cycles)
            self.charge_time_h = args.get("charge_time_h", self.charge_time_h)
            print(f"[Set] temp={self.temp_c}°C, cycles={self.cycles}, charge_time={self.charge_time_h} h")
        elif typ == "predict_capacity":
            cap = self.twin.predict(self.temp_c, self.cycles, self.charge_time_h)
            print(f"[Prediction] Capacity = {cap:.4f} (fraction of initial)")
        elif typ == "set_initial_capacity":
            self.twin.c0 = args.get("c0", self.twin.c0)
            print(f"[Set] Initial capacity = {self.twin.c0}")
        elif typ == "show_config":
            print(f"[Config] temp={self.temp_c}°C, cycles={self.cycles}, charge_time={self.charge_time_h} h, C0={self.twin.c0}")
        else:
            print(f"[Unknown action] {action}")

if __name__ == "__main__":
    BatterySimApp().run()
