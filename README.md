# Bernini 容器化部署与使用指南

> 本项目是 [ByteDance/Bernini](https://github.com/bytedance/Bernini)（Latent Semantic Planning for Video Diffusion）的容器化部署分支。通过 GitHub Actions 自动构建并推送 Docker 镜像（GHCR + 阿里云 ACR），可直接用于本地 GPU 服务器或阿里云 FC3（函数计算）部署。

## 目录

- [镜像地址](#镜像地址)
- [启动服务](#启动服务)
- [API 使用说明](#api-使用说明)
- [FC3 部署](#fc3-部署)
- [GPU 与显存约束](#gpu-与显存约束)

---

## 镜像地址

| 镜像 | 说明 | 构建文件 |
|---|---|---|
| `ghcr.io/balcklive/bernini:latest` | 完整版镜像（含 flash-attn，约 10GB） | `Dockerfile` |
| `crpi-v5j14rjtcacf9f23.cn-shanghai.personal.cr.aliyuncs.com/aliyun_kaka/test:latest` | 完整版镜像（阿里云 ACR 上海） | `Dockerfile` |
| `crpi-v5j14rjtcacf9f23.cn-shanghai.personal.cr.aliyuncs.com/aliyun_kaka/test:fc3` | FC3 精简版（含 flash-attn） | `Dockerfile.fc3` |

代码推送到 `main` 分支后，GitHub Actions 自动构建并推送上述镜像（同时保留最新 3 个版本，自动清理旧版本）。

---

## 启动服务

容器默认命令是 Gradio Web UI；可通过覆写命令切换为 FastAPI 服务或命令行推理。`--config` 参数指向模型权重目录：本地部署用挂载卷（如 `/models`），FC3 用内置路径 `/mnt/bernini`。

### 1. Gradio Web UI（默认入口）

```bash
docker run --gpus all -p 7860:7860 \
  -v /mnt/test:/models \
  ghcr.io/balcklive/bernini:latest \
  python gradio_demo.py --config /models
```

浏览器访问 `http://<服务器IP>:7860`。任务类型下拉框（t2i / i2i / t2v / v2v / rv2v / r2v 等）自动填充 `guidance_mode`，支持上传媒体文件，结果内联展示。

### 2. FastAPI REST API 服务（推荐用于程序调用）

```bash
docker run --gpus all -p 7860:7860 \
  -v /mnt/test:/models \
  ghcr.io/balcklive/bernini:latest \
  python api_server.py --config /models --port 7860
```

启动后：

- 健康检查：`GET /v1/health`
- Swagger 文档：`http://<服务器IP>:7860/docs`

### 3. 命令行推理

```bash
docker run --gpus all --rm \
  -v /mnt/test:/models \
  ghcr.io/balcklive/bernini:latest \
  python infer_single_gpu.py --config /models \
    --task_type t2v --prompt "一只猫在公园散步"
```

完整 Bernini 的更多用法（case 文件、guidance_mode、prompt enhancer 等）参见上游文档 `docs/bernini.md`。

---

## API 使用说明

| 端点 | 方法 | 说明 |
|---|---|---|
| `/v1/health` | GET | 健康检查 |
| `/v1/tasks` | GET | 任务类型与 guidance mode 列表 |
| `/v1/generate` | POST | 生成（JSON，媒体支持 base64 / URL / 路径） |
| `/v1/generate/upload` | POST | 生成（multipart 文件上传） |
| `/v1/output/{file}` | GET | 下载生成结果 |

### 文生视频

```bash
curl -X POST http://<服务器IP>:7860/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"task_type": "t2v", "prompt": "A cat walking in the park", "num_frames": 33}' \
  | jq .output_url
```

### 视频编辑（multipart 上传文件）

> 注意：Windows/WSL 终端 shell 用 GBK 编码发送中文会乱码，请把中文写入 UTF-8 文件后以文件方式引用。

```bash
printf '去除所有的字幕' > /tmp/prompt.txt

curl -X POST http://<服务器IP>:7860/v1/generate/upload \
  -F "task_type=v2v" \
  -F "prompt=</tmp/prompt.txt" \
  -F "num_frames=17" \
  -F "video=@sample.mp4" \
  | jq .output_url
```

生成成功后，`output_url` 形如 `/v1/output/<文件名>`，拼接服务地址即可下载：

```bash
curl -o result.mp4 http://<服务器IP>:7860/v1/output/<文件名>
```

### 可调参数

`task_type`、`prompt`、`neg_prompt`、`guidance_mode`、`num_frames`、`num_inference_steps`、`max_image_size`、`height`、`width`、`flow_shift`、`seed`、`fps`、`use_pe` 等。请求时省略的字段使用任务类型对应的默认值。

---

## FC3 部署

镜像与 FC3 函数必须位于**同一地域**（本项目均为上海 cn-shanghai）。在 FC3 控制台创建自定义容器函数，使用镜像：

```
crpi-v5j14rjtcacf9f23.cn-shanghai.personal.cr.aliyuncs.com/aliyun_kaka/test:fc3
```

entrypoint 命令（`--port 7860` 必须与 `customContainerConfig.port` 一致）：

```
python api_server.py --config /mnt/bernini --port 7860
```

通过 FC3 HTTP 触发器对外暴露 API，地址形如 `*.fcapp.run`。注意：

- 浏览器直接访问 FC3 默认域名会触发下载（阿里云强制 `Content-Disposition: attachment`），程序/curl 调用不受影响；如需浏览器访问请配置自定义域名（CNAME 绑定）。
- 中文提示词请用 UTF-8 文件方式传入（同上）。

---

## GPU 与显存约束

FC3 当前可用的 GPU 类型（上海地域）：

| GPU 类型 | 架构 | 显存 | flash-attn | 备注 |
|---|---|---|---|---|
| `fc.gpu.tesla.1` | Turing (T4) | 16GB | ❌ 自动降级 SDPA | 配额充足，但仅支持低帧数短视频 |
| `fc.gpu.ampere.1` | Ampere (A10) | 16GB | ✅ | 需确认配额 |
| `fc.gpu.ada.2` | Ada (L20) | 24GB | ✅ | 配额少（按量 1 卡），易满 |

**T4（fc.gpu.tesla.1）实测显存约束：**

| 参数 | 结果 |
|---|---|
| 81 帧 + 默认分辨率 | ❌ OOM（需 ~199GB） |
| 33 帧 | ❌ OOM（需 ~36GB） |
| **17 帧 + 512 分辨率 + SDPA** | ✅ 成功（~15 分钟） |
| t2i 文生图（1 帧） | ✅ 成功（~200 秒） |

需要更高质量视频时，建议换 `fc.gpu.ampere.1` 或 `fc.gpu.ada.2`，或用 ECI 部署到更大显存 GPU。

> T4 上使用完整 Bernini（Qwen planner 路径直接调用 `flash_attn_func`）仍需处理，建议视频编辑用 T4 + SDPA 时选择渲染器模型或降低分辨率/帧数。

---

## 其他说明

- 完整镜像约 10GB（CUDA devel + flash-attn），跨云推送约需 1 小时；未改动时利用 registry 缓存可秒级重建。
- 模型代码**强制要求** flash-attn，两个 Dockerfile 均保留，不能为省体积删除。
- 本部署分支为 API/容器化场景优化，原版论文、模型对比、训练等完整文档见上游 [ByteDance/Bernini](https://github.com/bytedance/Bernini)。

## License

Apache License 2.0。详见 [LICENSE](LICENSE)。
