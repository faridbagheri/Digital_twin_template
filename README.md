
# 🧪 Digital Twin Templates Collection

This repository contains two example/template projects for exploring **digital twin** concepts:

1. **Hybrid Digital Twin Framework for Li-ion Battery Modeling (with LLM integration)**
2. **Reachy Digital Twin (with VR/Physics Simulation + LLM control)**

These projects are **for learning and experimentation only** — they are not production-ready, but they provide a base for building more advanced digital twin applications.

---

## 📂 Project Structure

```
.
├── 20250812_hybrid_battery_twin_digital_with_LLM.py   # Template for Li-ion battery hybrid digital twin
├── 20250812_reachy_digital_with_LLM.py                # Template for Reachy humanoid digital twin
├── env                                                # Example environment variables for LLM API key
└── README.txt                                         # This file
```

---

## 1️⃣ Hybrid Digital Twin for Li-ion Batteries

**File:** `20250812_hybrid_battery_twin_digital_with_LLM.py`

### Overview
A **hybrid** approach that combines:
- **Physics-based modeling** — a simple exponential decay model for capacity loss.
- **ML-based residual correction** — a toy machine learning residual term.
- **LLM Interface** — powered by `gpt-4o-mini` to parse natural-language commands into structured simulation actions.

### Features
- Set simulation parameters (temperature, cycles, charge time) via **natural language**.
- Predict battery capacity from a **physics + ML hybrid model**.
- Show current configuration.
- Supports fallback text parsing if LLM is unavailable.

### Requirements
- Python 3.9+
- `numpy`
- `python-dotenv`
- `openai`

Install:
```bash
pip install numpy python-dotenv openai
```

### Environment Setup
Copy `.env.example` to `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### Run the Simulation
```bash
python 20250812_hybrid_battery_twin_digital_with_LLM.py
```

Example commands in the REPL:
```
simulate at 30 C, 500 cycles, 2 h charge
predict capacity
show config
```

---

## 2️⃣ Reachy Digital Twin

**File:** `20250812_reachy_digital_with_LLM.py`

### Overview
A **VR-enabled** digital twin template for controlling a simulated Reachy humanoid robot using:
- **PyBullet** for physics simulation and inverse kinematics.
- **Harfang3D** for rendering, VR device handling, and user interface.
- **Keyboard/VR Controller** input mapping to robot joints.
- **LLM Interface** for potential natural-language control extensions.

### Features
- Load Reachy URDF model and control via:
  - VR controllers (Oculus/Vive)
  - Keyboard (fallback mode)
- Manual calibration of VR body alignment with Reachy.
- Real-time joint control with inverse kinematics.
- GUI debug panels for fine-tuning joint offsets.

### Requirements
- Python 3.9+
- `pybullet`
- `HarfangHighLevel` (Harfang3D Python bindings)
- VR hardware (optional)

Install:
```bash
pip install pybullet
# HarfangHighLevel installation depends on your platform; see Harfang3D docs
```

### Run the Simulation
```bash
python 20250812_reachy_digital_with_LLM.py
```

Optional: disable VR mode:
```bash
python 20250812_reachy_digital_with_LLM.py no_vr
```

---

## ⚠️ Disclaimer
- These projects are **templates** — intended for prototyping and demonstration.
- The models used (battery decay, residual ML) are **toy examples** and not calibrated for real-world accuracy.
- The Reachy simulation assumes you have the appropriate model assets and dependencies installed.

---

## 📜 License
MIT License — feel free to modify and build upon these templates.

---

## 🤝 Contributing
Pull requests welcome! If you improve the models, add more realistic physics, or enhance the LLM control logic, feel free to submit a PR.

---

## 🧩 References
- [PyBullet](https://pybullet.org/)
- [Harfang3D](https://www.harfang3d.com/)
- [OpenAI Python SDK](https://platform.openai.com/docs/api-reference)
- [Reachy Robot](https://www.pollen-robotics.com/reachy/)
