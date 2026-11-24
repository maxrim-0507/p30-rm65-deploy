# VLA 部署说明

## 是什么

- 模型：GR00T N1.x (3B) 微调，v36 checkpoint（`use_relative_actions`）。
- 格式：**PyTorch / LeRobot `GrootPolicy` safetensors**（不是 ONNX）。加载：
  ```python
  from lerobot.policies.groot.modeling_groot import GrootPolicy
  policy = GrootPolicy.from_pretrained(ckpt).to("cuda").eval()
  ```
- 推理框架：PyTorch CUDA（Thor Jetson）。实测 ~0.27s/chunk。

## 输入 / 输出

- 输入 obs：front_cam 图 + arm_cam 图 + state16 + 指令（"抓取X色的方块"）。
- 输出：40 步 × 13 维动作（arm 座系绝对 eef9 + grip + base_vel）。
- 封装：`vla/local_gr00t_policy.py::LocalGr00tPolicy`（含 pre/post processor）。

## 关键点

1. **固定 seed**：动作头是流匹配（随机），推理前 `torch.manual_seed(seed + t)`，否则不可复现。
2. **预/后处理**：用训练同款 `make_groot_pre_post_processors_from_pretrained`；`robot_type="rm65"`。
3. **频率**：chunk 40 步≈1.28s，推理 0.27s → 延迟被 chunk 执行掩盖，同步闭环即可，**无需异步推理**。

## 加速（可选）

- 当前 PyTorch 已够（0.27s）。
- 需要更快：导出 **TensorRT**（NVIDIA 官方支持 GR00T），比 ONNX 更贴 Jetson。
- 只有目标平台无 PyTorch/CUDA（纯 ONNX Runtime/边缘盒）时才考虑 ONNX（flow-matching + processor 导出较复杂）。

## 部署清单

- [ ] lerobot(GR00T 分支) + torch 装好，cuda 可用
- [ ] checkpoint 就位（`checkpoints/groot_rel_v36/020000/pretrained_model/`）
- [ ] obs 口径与训练一致（见 architecture.md）
- [ ] seed 固定
- [ ] GPU 显存足够（3B，Thor 128GB 统一内存充足）
