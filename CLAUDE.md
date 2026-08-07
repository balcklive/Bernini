# CLAUDE.md — Bernini 项目部署与运维备忘

本文档记录 Bernini 项目容器化部署（GitHub Actions → ACR → FC3）过程中遇到的问题与解决方案，以及关键运维知识。

## 目录

- [架构与部署链路](#架构与部署链路)
- [镜像地址](#镜像地址)
- [启动方式](#启动方式)
- [踩坑记录](#踩坑记录)
- [API 服务说明](#api-服务说明)
- [FC3 GPU 与显存约束](#fc3-gpu-与显存约束)

---

## 架构与部署链路

```
开发者 push 代码到 GitHub main
  → GitHub Actions 自动构建 Docker 镜像（provenance=false）
  → 同时推送到 GHCR（ghcr.io）和阿里云 ACR（上海）
  → DevPod/FC3 从 ACR 拉取镜像部署
  → 通过 FC3 HTTP 触发器对外提供 API
```

关键文件：
- `Dockerfile` — 唯一镜像（含 flash-attn，约 10GB），同时推送 GHCR / 阿里云 ACR / 火山 VCR
- `.github/workflows/build-image.yml` — 单 job 三仓库并行推送
- `api_server.py` — FastAPI REST API 服务

> `Dockerfile.fc3` 已废弃（与完整版几乎无差别，仅少了 uv），不再构建推送。

---

## 镜像地址

```bash
# GHCR（公开，社区可用）
ghcr.io/balcklive/bernini:latest

# 阿里云 ACR（上海，DevPod/FC3 用）
crpi-v5j14rjtcacf9f23.cn-shanghai.personal.cr.aliyuncs.com/aliyun_kaka/test:latest

# 火山引擎 VCR（北京）
fm-qc-prj-images-cn-beijing.cr.volces.com/gpu-infer/bernini:latest
```

注意：**FC3 函数与 ACR 必须在同一地域**（本项目均为上海 cn-shanghai）。
原 `:fc3` 标签已停止推送，FC3 函数应改用 `:latest`（镜像内容与 FC3 版基本一致）。

---

## 启动方式

```bash
# Gradio Web UI（默认）
docker run --gpus all -p 7860:7860 \
  -v /mnt/test:/models \
  <镜像地址> \
  python gradio_demo.py --config /models

# FastAPI 服务（推荐用于 API 调用）
docker run --gpus all -p 7860:7860 \
  -v /mnt/test:/models \
  <镜像地址> \
  python api_server.py --config /models --port 7860

# 命令行推理
docker run --gpus all --rm \
  -v /mnt/test:/models \
  <镜像地址> \
  python infer_single_gpu.py --config /models \
    --task_type t2v --prompt "一只猫在公园散步"
```

FC3 部署时的 entrypoint（`--port 7860` 必须与 FC3 的 `customContainerConfig.port` 一致）：
```
python api_server.py --config /mnt/bernini --port 7860
```

---

## 踩坑记录

### 1. tzdata 安装卡住（apt-get 交互提示）

**现象**：Docker 构建卡在 `Configuring tzdata` 等待时区输入。

**解决**：Dockerfile 顶部设置：
```dockerfile
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
```

### 2. flash-attn 编译报 `No module named 'torch'`

**现象**：`pip install flash-attn` 在 "Getting requirements to build wheel" 阶段报找不到 torch。

**根因**：PEP 517 构建隔离创建独立沙盒，看不到已安装的 torch。

**解决**：加 `--no-build-isolation`：
```dockerfile
RUN MAX_JOBS=$(nproc) pip3 install --no-cache-dir \
    --no-build-isolation flash-attn==2.8.3
```

### 3. `pip install -e .` 失败（metadata-generation-failed）

**现象**：`pip install -e . --no-deps` 报 metadata-generation-failed。

**根因 1**：pyproject.toml 缺少 `[build-system]` 段。
**根因 2**：setuptools 找不到包（未配置 package discovery）。
**根因 3**：构建隔离看不到已装依赖。

**解决**（pyproject.toml）：
```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["bernini*"]
```
（Dockerfile 中同样加 `--no-build-isolation`）

### 4. FC3 镜像优化失败 `platform of image is unknown/unknown`

**现象**：FC3 部署时报 `stateReason: invalid image, platform of image is unknown/unknown`，`ImageOptimizingFailed`。

**根因**：新版 Docker Buildx 默认添加 Provenance Attestations，镜像清单里出现 `unknown/unknown` 平台条目，FC3 无法解析。

**解决**：workflow 的 build 步骤加 `provenance: false`：
```yaml
- name: Build and push
  uses: docker/build-push-action@v6
  with:
    ...
    provenance: false   # 关键！
```

### 5. FC3 报 `Image not stored in ACR`（地域不一致）

**现象**：`Image and function must be in the same region`。

**根因**：ACR 在上海，FC3 函数在杭州，地域不匹配。

**解决**：确保 ACR 和 FC3 函数在**同一地域**。

### 6. FC3 GPU 配额超限 `fc.gpu.ada.2 exceeded`

**现象**：`Account's total GPU usage of 'fc.gpu.ada.2' exceeded`。

**根因**：Ada GPU 配额只有 1 卡，被旧失败部署占用未释放。

**解决**：
1. 去 FC3 控制台**彻底删除**旧函数实例释放 GPU
2. 或换 GPU 类型（如 `fc.gpu.tesla.1` 配额充足）

配额查看：`https://quotas.console.aliyun.com/` → 通用配额 → 函数计算
（Tesla 总卡 10，按量 3；Ada 总卡 3，按量 1）

### 7. 删除 flash-attn 后容器启动崩溃

**现象**：容器启动报 `ValueError: 不能在fa2和fa3都不支持的情况下工作！！！！`

**根因**：Bernini 的 `modeling_qwen2_5_vl.py` **强制要求** flash-attn 存在（导入时检查），不可省略。

**结论**：**Dockerfile 必须保留 flash-attn，不能为了省体积删除。**

### 8. T4 GPU 上 `FlashAttention only supports Ampere GPUs or newer`

**现象**：生成时报 `RuntimeError: FlashAttention only supports Ampere GPUs or newer`。

**根因**：T4 是 Turing 架构（SM 7.5），flash-attn 需要 Ampere+（SM 8.0+）。FC3 的 `fc.gpu.tesla.1` 是 T4。

**解决**（`bernini/attention.py`）：后端选择时检测 GPU 计算能力，不支持则降级 SDPA：
```python
def _gpu_supports_flash_attn() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8
```
`_select_backend()` 中仅当 `major >= 8` 时才尝试 FA3/FA2，否则用 `sdpa`。

注意：`modeling_qwen2_5_vl.py` 中还有直接调用 `flash_attn_func` 的地方（完整 Bernini 的 Qwen planner 路径会用到），若使用完整 Bernini 在 T4 上仍需处理。

### 9. 浏览器访问 FC3 默认域名触发下载（Content-Disposition: attachment）

**现象**：用浏览器打开 `*.fcapp.run` 地址，网页被下载而不是渲染。

**根因**：阿里云 FC 对**默认域名**强制添加 `Content-Disposition: attachment` 响应头。

**解决**：
- API 调用（curl/程序）不受影响，直接可用
- 浏览器访问需要：配置自定义域名（CNAME 绑定）或使用触发器测试域名

### 10. curl 发送中文提示词乱码

**现象**：`curl -F "prompt=去除字幕"` 收到的提示词变成 `È¥³ýËùÓÐµÄ×ÖÄ»`。

**根因**：Windows/WSL 终端 shell 用 GBK 编码发送中文。

**解决**：把中文写入 UTF-8 文件，用文件引用：
```bash
printf '去除所有的字幕' > /tmp/prompt.txt
curl -F "prompt=</tmp/prompt.txt" ...
```

### 11. api_server.py 的 `resolve_system_prompt` 报 AttributeError

**现象**：`/v1/generate` 报 `AttributeError: 'Namespace' object has no attribute 'task_type'`。

**根因**：传了不完整的 argparse.Namespace 给 `resolve_system_prompt`。

**解决**：直接用 `get_system_prompt_for_task(task_type)` 替代（从 `bernini.prompt_enhancer` 导入）。

### 12. uvicorn 子进程丢失 PIPELINE

**现象**：所有 `/v1/generate` 请求返回 503。

**根因**：`uvicorn.run("api_server:app")` 用字符串模块路径，uvicorn 会 respawn 子进程重新导入模块，导致 `main()` 加载的 PIPELINE 丢失。

**解决**：传 app 对象而非字符串：
```python
uvicorn.run(app, host=args.host, port=args.port, ...)
```

---

## API 服务说明

启动命令：`python api_server.py --config /mnt/bernini --port 7860`

| 端点 | 方法 | 说明 |
|---|---|---|
| `/v1/health` | GET | 健康检查 |
| `/v1/tasks` | GET | 任务类型列表 |
| `/v1/generate` | POST | 生成（JSON，媒体支持 base64/URL/路径） |
| `/v1/generate/upload` | POST | 生成（multipart 文件上传） |
| `/v1/output/{file}` | GET | 下载生成结果 |
| `/docs` | GET | Swagger 文档 |

调用示例：
```bash
# 文生图
curl -X POST <FC3_URL>/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"task_type": "t2i", "prompt": "一只猫"}' \
  | jq .output_url

# 视频编辑（上传文件，注意中文用文件方式传）
curl -X POST <FC3_URL>/v1/generate/upload \
  -F "task_type=v2v" \
  -F "prompt=</tmp/prompt.txt" \
  -F "num_frames=17" \
  -F "video=@sample.mp4"
```

---

## FC3 GPU 与显存约束

FC3 当前可用的 GPU 类型（上海地域）：

| GPU 类型 | 架构 | 显存 | flash-attn | 配额 |
|---|---|---|---|---|
| `fc.gpu.tesla.1` | Turing (T4) | 16GB | ❌ 不支持（需 SDPA 降级） | 总 10 卡，按量 3 卡 |
| `fc.gpu.ampere.1` | Ampere (A10) | 16GB | ✅ | 需确认配额 |
| `fc.gpu.ada.2` | Ada (L20) | 24GB | ✅ | 总 3 卡，按量 1 卡（易满） |

**T4（fc.gpu.tesla.1）显存约束**（实测）：

| 参数 | 结果 |
|---|---|
| 81 帧 + 默认分辨率 | ❌ OOM（需 ~199GB） |
| 33 帧 | ❌ OOM（需 ~36GB） |
| **17 帧 + 512 分辨率 + SDPA** | ✅ 成功（~15 分钟） |
| t2i 文生图（1 帧） | ✅ 成功（~200 秒） |

需要更高质量视频时，建议换 `fc.gpu.ampere.1` 或 `fc.gpu.ada.2`，或用 ECI 部署到更大显存 GPU。

---

## 其他要点

- **GitHub Actions 密钥**：`GHCR_PAT`（delete:packages 权限）、`ALIYUN_ACR_USERNAME`、`ALIYUN_ACR_PASSWORD`、`VOLCANO_CR_USERNAME`、`VOLCANO_CR_PASSWORD`（火山 VCR 密码在控制台「镜像仓库 → 访问凭证」生成）
- **GHCR 版本清理**：workflow 内置"保留最新 3 个版本"清理步骤，需 GHCR_PAT 才有删除权限（ACR/火山无自动清理，需各自控制台手动清）
- **镜像缓存**：workflow 用 `cache-from: type=registry` 加速增量构建，未改动时秒级完成
- **镜像体积**：完整镜像约 10GB（CUDA devel + flash-attn），跨云推送约需 1 小时
- **ACR 触发器**（镜像构建后通知部署）与**构建规则**（从 GitHub 拉代码构建）是两个功能，注意区分
