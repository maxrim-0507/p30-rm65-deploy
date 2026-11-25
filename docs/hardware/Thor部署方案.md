# Phase 4：Thor 部署方案

> 日期：2026-10-16　|　前置：Phase 1-3 已完成（500 条数据 → 微调 loss 0.04 → TensorRT engine）
> 云端推理闭环已在 L40S 验证通过（7 engine 全加载，25Hz，MSE 0.0011）

## 现状与两个平台障碍

1. **TensorRT engine 平台绑定**：云端 L40S（x86_64）构建的 engine 不能在 Thor（aarch64）直接跑，需在 Thor 重建。
2. **Thor JetPack 版本**：Thor 实测 `R38.4`，而 Isaac-GR00T 的 `scripts/deployment/thor/install_deps.sh` 硬性要求 **JetPack 7.2 / R39**。

## 云端产出清单（`/root/gpufree-data/outputs/`）

| 产出 | 路径 | 大小 |
|---|---|---|
| PyTorch 权重 | `rm65_vla/checkpoint-2000` | ~6.5GB |
| ONNX | `gr00t_trt_deployment/onnx/` | 6.2GB |
| TensorRT engine（x86_64） | `gr00t_trt_deployment/engines/` | 6.1GB |
| RM65 modality config | `Isaac-GR00T/examples/RM65/rm65_config.py` | — |

---

## 方案 A：刷机 Thor → 完整部署（推荐长期）

### A1. 刷机 Thor 到 JetPack 7.2（R39）

用 NVIDIA SDK Manager 或 Jetson 刷机工具，把 Thor 刷到 JetPack 7.2 / Jetson Linux 39.2。
（这是硬件级操作，需物理接触 Thor 或用 SDK Manager 远程刷机）

### A2. Thor 上装 Isaac-GR00T（aarch64）

```bash
# Thor 上（刷机后，JetPack 7.2）
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
bash scripts/deployment/thor/install_deps.sh   # 装 aarch64 torch/tensorrt/flash-attn
source scripts/activate_jetpack72.sh
```

### A3. 从云端下载 checkpoint + ONNX + processor 到 Thor

```bash
# Thor 上，从云端（120.209.70.195:31050）拉取（约 12GB）
scp -P 31050 -r root@120.209.70.195:/root/gpufree-data/outputs/rm65_vla/checkpoint-2000 ./checkpoint-2000
scp -P 31050 -r root@120.209.70.195:/root/gpufree-data/outputs/gr00t_trt_deployment/onnx ./onnx
```

### A4. Thor 上重建 TensorRT engine（aarch64）

```bash
cd Isaac-GR00T
python scripts/deployment/build_tensorrt_engine.py \
  --mode full_pipeline \
  --onnx-dir ./onnx \
  --engine-dir ./engines \
  --precision bf16   # 或 fp8（Thor Blackwell 支持，导出工具暂只 bf16）
```

### A5. Thor 跑推理验证

```bash
python scripts/deployment/standalone_inference_script.py \
  --model-path ./checkpoint-2000 \
  --dataset-path <数据> \
  --embodiment-tag NEW_EMBODIMENT \
  --inference-mode trt_full_pipeline \
  --trt-engine-path ./engines
```

---

## 方案 B：云端 L40S 推理服务器（ZMQ，跳过刷机，快速联调）

推理在云端 GPU 服务器，Thor 只做控制（读相机/关节 → 发 ZMQ 请求 → 执行 RM65 动作）。这是方案文档推荐的架构（推理与控制解耦），**不需要刷机 Thor**。

### B1. 云端启动 ZMQ 推理服务器

```bash
# 云端 L40S
cd /root/gpufree-data/Isaac-GR00T
HF_TOKEN=<token> HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python gr00t/eval/run_gr00t_server.py \
  --model-path /root/gpufree-data/outputs/rm65_vla/checkpoint-2000 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/RM65/rm65_config.py \
  --host 0.0.0.0 --port 5555
```

（用 PyTorch 推理，不用 engine；如要 TRT 加速，改 standalone_inference_script.py 封装成 server）

### B2. Thor 端写 ZMQ client

参照 `gr00t/policy/server_client.py` / `examples/DROID/server_client.py`：
1. 读腕部相机 RGB + RM65 关节角（`rm_robot` SDK）
2. 打包成推理请求，发 ZMQ（云端 IP:5555）
3. 收动作块（7 维相对末端 + 夹爪），用 `ros2_rm_robot`/`rm_robot` 执行

> Thor 已有 `Robotic_Arm` SDK + `rm_robot` 包，client 只需 `pyzmq` + 相机读取 + 动作执行，不依赖 Isaac-GR00T。

### B3. 真机联调

云端服务器保持运行，Thor 端跑 client 循环，验证「相机→推理→动作→RM65 执行」闭环。

---

## 对比

| | 方案 A（刷机） | 方案 B（云端推理） |
|---|---|---|
| 需要刷机 Thor | ✅ 需要（R38→R39） | ❌ 不需要 |
| 推理位置 | Thor 本地（TRT engine） | 云端 L40S（PyTorch/ZMQ） |
| 延迟 | 最低（本地推理） | 有网络延迟（局域网可控） |
| 部署复杂度 | 高（刷机+装环境+重建 engine） | 低（云端跑 server + Thor 写 client） |
| 适用 | 最终生产部署 | 快速联调验证 |

**建议**：先用方案 B 快速跑通真机闭环（不刷机），验证端到端链路；确认没问题后再做方案 A（刷机 + 本地推理）作为最终生产部署。
