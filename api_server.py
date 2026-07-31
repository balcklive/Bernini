# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# =============================================================================
# Bernini REST API Server (FastAPI)
# =============================================================================
#
# Single-GPU:
#     python api_server.py --config ByteDance/Bernini-Diffusers
#
# Renderer-only:
#     python api_server.py --config configs/bernini_renderer_wan22 \
#         --high_noise_ckpt <path> --low_noise_ckpt <path>
#
# Docker:
#     docker run --gpus all -p 8000:8000 ghcr.io/你的用户名/bernini:latest \
#         python api_server.py --config ByteDance/Bernini-Diffusers
#
# 调用示例（文本生成视频）:
#     curl -X POST http://localhost:8000/v1/generate \
#         -H "Content-Type: application/json" \
#         -d '{"task_type": "t2v", "prompt": "A cat walking in the park"}'
#
# 调用示例（视频编辑，上传文件）:
#     curl -X POST http://localhost:8000/v1/generate/upload \
#         -F "task_type=v2v" \
#         -F "prompt=Make it night time" \
#         -F "video=@input.mp4"
# =============================================================================

from __future__ import annotations

import argparse
import base64
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bernini.cli import (
    DEFAULT_NEG_PROMPT,
    GUIDANCE_MODES,
    build_pipeline,
)
from bernini.prompt_enhancer import get_system_prompt_for_task
from bernini.pipeline import BerniniPipeline

logger = logging.getLogger("bernini.api")

# ── Global pipeline state ───────────────────────────────────────────────────
PIPELINE: BerniniPipeline | None = None
DEVICE: torch.device | None = None
OUTPUT_DIR: str | None = None
REWRITER = None
ARGS: argparse.Namespace | None = None

# ── Helpers ─────────────────────────────────────────────────────────────────


def _is_full_bernini_pipeline() -> bool:
    return isinstance(PIPELINE, BerniniPipeline)


def _task_defaults(task_type: str) -> dict:
    """Return task-specific default generation parameters."""
    # Reuse the defaults from gradio_demo if available; otherwise static map.
    try:
        from gradio_demo import (
            BASE_TASK_DEFAULTS,
            RENDERER_TASK_DEFAULTS,
            FULL_BERNINI_TASK_DEFAULTS,
        )

        defaults = dict(BASE_TASK_DEFAULTS)
        table = FULL_BERNINI_TASK_DEFAULTS if _is_full_bernini_pipeline() else RENDERER_TASK_DEFAULTS
        defaults.update(table.get(task_type, {}))
        return defaults
    except ImportError:
        # Static fallback
        return {
            "guidance_mode": "v2v_apg",
            "num_frames": 81,
            "num_inference_steps": 40,
            "max_image_size": 848,
            "height": 480,
            "width": 848,
            "flow_shift": 5.0,
            "seed": 42,
            "fps": 16,
            "omega_vid": 1.25,
            "omega_img": 4.5,
            "omega_txt": 4.0,
            "omega_tgt": 0.5,
            "omega_scale": 0.8,
            "eta": 0.5,
            "momentum": 0.0,
            "planning_step": 25,
            "vit_txt_cfg": 1.2,
            "vit_img_cfg": 1.0,
            "vit_denoising_step": 5,
        }


def _guidance_mode_for_task(task_type: str) -> str:
    """Resolve the default guidance mode for a task type."""
    try:
        from gradio_demo import GUIDANCE_MODE_BY_TASK

        return GUIDANCE_MODE_BY_TASK.get(task_type, "v2v_apg")
    except ImportError:
        return "v2v_apg"


def _is_image_task(task_type: str) -> bool:
    return task_type in ("t2i", "i2i")


def _resolve_media(value: str | None, work_dir: str) -> str | None:
    """Convert a media input to a local file path.

    Accepted formats (in order):
    1. Existing local path — used as-is
    2. Base64 data URI (``data:...;base64,...``) — decoded to temp file
    3. Remote URL — downloaded to temp file
    4. Raw base64 string — decoded to temp file
    """
    if value is None:
        return None

    # 1. Already a local file
    if os.path.isfile(value):
        return value

    # 2. Base64 data URI
    if value.startswith("data:"):
        try:
            mime_part, _, encoded = value.partition(",")
            decoded = base64.b64decode(encoded)
            ext = _ext_from_mime(mime_part) or ".bin"
            path = os.path.join(work_dir, f"media_{uuid.uuid4().hex}{ext}")
            with open(path, "wb") as f:
                f.write(decoded)
            logger.debug("Decoded base64 media -> %s", path)
            return path
        except Exception as e:
            logger.warning("Failed to decode base64 data URI: %s", e)
            return None

    # 3. Raw base64 string (long alphanumeric+/=)
    if re.match(r"^[A-Za-z0-9+/=]{50,}$", value):
        try:
            decoded = base64.b64decode(value)
            path = os.path.join(work_dir, f"media_{uuid.uuid4().hex}.bin")
            with open(path, "wb") as f:
                f.write(decoded)
            logger.debug("Decoded raw base64 -> %s", path)
            return path
        except Exception as e:
            logger.warning("Failed to decode raw base64: %s", e)
            return None

    # 4. Remote URL
    try:
        import requests
        logger.info("Downloading media from %s ...", value)
        resp = requests.get(value, timeout=600, stream=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        ext = _ext_from_mime(ct) or Path(value.split("?")[0]).suffix or ".bin"
        path = os.path.join(work_dir, f"download_{uuid.uuid4().hex}{ext}")
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.debug("Downloaded %s -> %s (%d bytes)", value, path, os.path.getsize(path))
        return path
    except Exception as e:
        logger.warning("Failed to download %s: %s", value, e)
        return None


def _ext_from_mime(mime: str) -> str:
    mime_map = {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }
    for prefix, ext in mime_map.items():
        if prefix in mime:
            return ext
    return ".bin"


# ── Request / Response models ──────────────────────────────────────────────


class GenerateRequest(BaseModel):
    task_type: str = Field("t2v", description="Task type: t2i, i2i, t2v, v2v, mv2v, r2v, rv2v, ads2v")
    prompt: str = Field("", description="Text prompt or editing instruction")
    neg_prompt: str | None = Field(None, description="Negative prompt (omit for default)")
    video: str | None = Field(None, description="Source video: local path, base64, data URI, or URL")
    image: str | None = Field(None, description="Single source image")
    images: list[str] | None = Field(None, description="Reference image(s): list of paths/base64/URLs")
    guidance_mode: str | None = Field(None, description="Override guidance mode (auto-selected from task_type if omitted)")
    num_frames: int | None = Field(None, ge=1, le=360)
    num_inference_steps: int | None = Field(None, ge=1, le=100)
    max_image_size: int | None = Field(None, ge=256, le=1280)
    height: int | None = Field(None, ge=0)
    width: int | None = Field(None, ge=0)
    flow_shift: float | None = Field(None, ge=0.0, le=20.0)
    seed: int | None = Field(None)
    fps: int | None = Field(None, ge=1, le=60)
    omega_vid: float | None = Field(None, ge=0.0, le=20.0)
    omega_img: float | None = Field(None, ge=0.0, le=20.0)
    omega_txt: float | None = Field(None, ge=0.0, le=20.0)
    omega_tgt: float | None = Field(None, ge=0.0, le=20.0)
    omega_scale: float | None = Field(None, ge=0.0, le=5.0)
    eta: float | None = Field(None, ge=0.0, le=5.0)
    momentum: float | None = Field(None, ge=-5.0, le=5.0)
    planning_step: int | None = Field(None, ge=1, le=100)
    vit_txt_cfg: float | None = Field(None, ge=0.0, le=5.0)
    vit_img_cfg: float | None = Field(None, ge=0.0, le=5.0)
    vit_denoising_step: int | None = Field(None, ge=1, le=50)
    use_pe: bool = Field(False, description="Enable GPT prompt enhancement (requires --use_pe at startup)")
    pe_model: str | None = Field(None, description="Prompt enhancer model name")


class GenerateResponse(BaseModel):
    task_id: str
    status: str  # "completed" | "failed"
    output_filename: str | None = None
    output_url: str | None = None
    prompt_used: str | None = None
    message: str | None = None
    elapsed_seconds: float | None = None


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pipeline is already initialized in main()
    if PIPELINE is None:
        logger.warning("Pipeline not initialized — the server will return 503 until it is")
    yield
    # Shutdown: log and exit
    logger.info("API server shutting down")


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Bernini API",
    description="Latent Semantic Planning for Video Diffusion — REST API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ──────────────────────────────────────────────────────────────────


@app.get("/v1/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok" if PIPELINE is not None else "loading",
        "pipeline_type": "Bernini" if _is_full_bernini_pipeline() else "Bernini-R",
        "device": str(DEVICE) if DEVICE else None,
        "output_dir": OUTPUT_DIR,
    }


@app.get("/v1/tasks")
async def list_tasks():
    """List available task types and guidance modes."""
    try:
        from gradio_demo import TASK_TYPE_CHOICES
        tasks = TASK_TYPE_CHOICES
    except ImportError:
        tasks = ["t2i", "i2i", "t2v", "v2v", "mv2v", "r2v", "rv2v", "ads2v"]
    return {"tasks": tasks, "guidance_modes": GUIDANCE_MODES}


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate video or image using the Bernini pipeline.

    All media inputs (video, image, images) accept:
    - Local file path (when server has access)
    - Base64 data URI: ``data:video/mp4;base64,...``
    - Raw base64 string
    - Remote URL (downloaded server-side)

    Returns immediately with the result; generation runs synchronously
    (typical: 30-120 seconds for 81-frame 480p video).
    """
    global PIPELINE, OUTPUT_DIR

    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    task_id = uuid.uuid4().hex[:12]
    t0 = time.time()

    # ── Validate ────────────────────────────────────────────────────────
    task_type = req.task_type
    try:
        from gradio_demo import TASK_TYPE_CHOICES
        if task_type not in TASK_TYPE_CHOICES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_type '{task_type}'. Choose from: {TASK_TYPE_CHOICES}",
            )
    except ImportError:
        pass  # skip validation if we can't import

    if not (req.prompt or "").strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    # ── Resolve media inputs ────────────────────────────────────────────
    work_dir = tempfile.mkdtemp(prefix=f"bernini_{task_id}_")
    video_path = _resolve_media(req.video, work_dir)
    image_path = _resolve_media(req.image, work_dir)
    images_paths = None
    if req.images:
        images_paths = [_resolve_media(img, work_dir) for img in req.images]
        images_paths = [p for p in images_paths if p is not None] or None

    # ── Merge defaults ──────────────────────────────────────────────────
    defaults = _task_defaults(task_type)
    kwargs: dict = {
        "neg_prompt": req.neg_prompt if req.neg_prompt is not None else DEFAULT_NEG_PROMPT,
        "guidance_mode": req.guidance_mode or defaults.get("guidance_mode") or _guidance_mode_for_task(task_type),
        "num_frames": req.num_frames if req.num_frames is not None else defaults.get("num_frames", 81),
        "num_inference_steps": req.num_inference_steps if req.num_inference_steps is not None else defaults.get("num_inference_steps", 40),
        "max_image_size": req.max_image_size if req.max_image_size is not None else defaults.get("max_image_size", 848),
        "height": req.height if req.height is not None else defaults.get("height", 480),
        "width": req.width if req.width is not None else defaults.get("width", 848),
        "flow_shift": req.flow_shift if req.flow_shift is not None else defaults.get("flow_shift", 5.0),
        "seed": req.seed if req.seed is not None else defaults.get("seed", 42),
        "fps": req.fps if req.fps is not None else defaults.get("fps", 16),
        "omega_vid": req.omega_vid if req.omega_vid is not None else defaults.get("omega_vid", 1.25),
        "omega_img": req.omega_img if req.omega_img is not None else defaults.get("omega_img", 4.5),
        "omega_txt": req.omega_txt if req.omega_txt is not None else defaults.get("omega_txt", 4.0),
        "omega_tgt": req.omega_tgt if req.omega_tgt is not None else defaults.get("omega_tgt", 0.5),
        "omega_scale": req.omega_scale if req.omega_scale is not None else defaults.get("omega_scale", 0.8),
        "eta": req.eta if req.eta is not None else defaults.get("eta", 0.5),
        "momentum": req.momentum if req.momentum is not None else defaults.get("momentum", 0.0),
        "planning_step": req.planning_step if req.planning_step is not None else defaults.get("planning_step", 25),
        "vit_txt_cfg": req.vit_txt_cfg if req.vit_txt_cfg is not None else defaults.get("vit_txt_cfg", 1.2),
        "vit_img_cfg": req.vit_img_cfg if req.vit_img_cfg is not None else defaults.get("vit_img_cfg", 1.0),
        "vit_denoising_step": req.vit_denoising_step if req.vit_denoising_step is not None else defaults.get("vit_denoising_step", 5),
    }

    # Force single frame for image tasks
    if _is_image_task(task_type):
        kwargs["num_frames"] = 1

    # ── Output path ─────────────────────────────────────────────────────
    ext = "png" if _is_image_task(task_type) else "mp4"
    output_filename = f"{task_id}_{uuid.uuid4().hex[:8]}.{ext}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # ── Prompt enhancement ──────────────────────────────────────────────
    prompt = req.prompt
    # Auto-select the system prompt from the task type (no CLI override in API).
    system_prompt = get_system_prompt_for_task(task_type)

    if req.use_pe and REWRITER is not None:
        pe_model = req.pe_model or ARGS.pe_model if ARGS else None
        logger.info("Enhancing prompt with PE model=%s ...", pe_model)
        from bernini.prompt_enhancer import PromptEnhancer
        enhancer = REWRITER if pe_model is None else PromptEnhancer(model=pe_model)
        prompt = enhancer(
            task_type, prompt,
            video=video_path, image=image_path, images=images_paths,
        ) or prompt
        kwargs["prompt"] = prompt

    # ── Run pipeline ────────────────────────────────────────────────────
    logger.info(
        "task=%s mode=%s steps=%d frames=%d seed=%s",
        task_type, kwargs["guidance_mode"], kwargs["num_inference_steps"],
        kwargs["num_frames"], kwargs["seed"],
    )

    try:
        if _is_full_bernini_pipeline():
            PIPELINE(
                task_type,
                prompt,
                video=video_path,
                image=image_path,
                images=images_paths,
                output_path=output_path,
                system_prompt=system_prompt,
                **kwargs,
            )
        else:
            PIPELINE(
                prompt,
                video=video_path,
                image=image_path,
                images=images_paths,
                output_path=output_path,
                system_prompt=system_prompt,
                **kwargs,
            )
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        elapsed = time.time() - t0
        return GenerateResponse(
            task_id=task_id,
            status="failed",
            message=f"Generation failed: {e}",
            prompt_used=prompt,
            elapsed_seconds=round(elapsed, 1),
        )

    elapsed = time.time() - t0
    logger.info("Generation complete in %.1fs -> %s", elapsed, output_filename)

    return GenerateResponse(
        task_id=task_id,
        status="completed",
        output_filename=output_filename,
        output_url=f"/v1/output/{output_filename}",
        prompt_used=prompt,
        elapsed_seconds=round(elapsed, 1),
    )


@app.post("/v1/generate/upload")
async def generate_with_upload(
    task_type: str = Form("t2v"),
    prompt: str = Form(""),
    neg_prompt: Optional[str] = Form(None),
    guidance_mode: Optional[str] = Form(None),
    num_frames: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    max_image_size: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    flow_shift: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    fps: Optional[int] = Form(None),
    omega_vid: Optional[float] = Form(None),
    omega_img: Optional[float] = Form(None),
    omega_txt: Optional[float] = Form(None),
    omega_tgt: Optional[float] = Form(None),
    omega_scale: Optional[float] = Form(None),
    eta: Optional[float] = Form(None),
    momentum: Optional[float] = Form(None),
    planning_step: Optional[int] = Form(None),
    vit_txt_cfg: Optional[float] = Form(None),
    vit_img_cfg: Optional[float] = Form(None),
    vit_denoising_step: Optional[int] = Form(None),
    use_pe: bool = Form(False),
    video: Optional[UploadFile] = File(None, description="Source video file"),
    image: Optional[UploadFile] = File(None, description="Source image file"),
    images: Optional[list[UploadFile]] = File(None, description="Reference image files"),
):
    """Generate with file uploads via multipart/form-data.

    Use this endpoint when you want to upload media files directly
    (instead of passing them as base64/URL in JSON).
    """
    # Save uploaded files to a temp directory
    upload_dir = tempfile.mkdtemp(prefix="bernini_upload_")
    req_kwargs = {
        "task_type": task_type,
        "prompt": prompt,
        "neg_prompt": neg_prompt,
        "guidance_mode": guidance_mode,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "max_image_size": max_image_size,
        "height": height,
        "width": width,
        "flow_shift": flow_shift,
        "seed": seed,
        "fps": fps,
        "omega_vid": omega_vid,
        "omega_img": omega_img,
        "omega_txt": omega_txt,
        "omega_tgt": omega_tgt,
        "omega_scale": omega_scale,
        "eta": eta,
        "momentum": momentum,
        "planning_step": planning_step,
        "vit_txt_cfg": vit_txt_cfg,
        "vit_img_cfg": vit_img_cfg,
        "vit_denoising_step": vit_denoising_step,
        "use_pe": use_pe,
    }

    if video:
        ext = Path(video.filename or "video.mp4").suffix
        path = os.path.join(upload_dir, f"upload_video_{uuid.uuid4().hex}{ext}")
        with open(path, "wb") as f:
            f.write(await video.read())
        req_kwargs["video"] = path

    if image:
        ext = Path(image.filename or "image.png").suffix
        path = os.path.join(upload_dir, f"upload_image_{uuid.uuid4().hex}{ext}")
        with open(path, "wb") as f:
            f.write(await image.read())
        req_kwargs["image"] = path

    if images:
        paths = []
        for img in images:
            ext = Path(img.filename or "img.png").suffix
            path = os.path.join(upload_dir, f"upload_img_{uuid.uuid4().hex}{ext}")
            with open(path, "wb") as f:
                f.write(await img.read())
            paths.append(path)
        req_kwargs["images"] = paths

    req = GenerateRequest(**req_kwargs)
    return await generate(req)


@app.get("/v1/output/{filename:path}")
async def get_output(filename: str):
    """Download a generated output file by filename."""
    # Security: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.normpath(os.path.join(OUTPUT_DIR, filename))
    if not path.startswith(os.path.normpath(OUTPUT_DIR)):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Output file not found")
    media_type = "video/mp4" if filename.endswith(".mp4") else "image/png" if filename.endswith(".png") else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bernini REST API Server")
    # Pipeline args (same as infer_single_gpu.py)
    g = parser.add_argument_group("model")
    g.add_argument("--config", default="configs/bernini_renderer_wan22")
    g.add_argument("--high_noise_ckpt", default=None)
    g.add_argument("--low_noise_ckpt", default=None)
    g.add_argument("--use_unipc", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--use_src_tgt_id", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--interpolate_src_id", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--max_trained_src_id", type=int, default=5)
    g.add_argument("--flow_shift", type=float, default=5.0)

    # Prompt enhancer
    g = parser.add_argument_group("prompt enhancer")
    g.add_argument("--use_pe", action="store_true")
    g.add_argument("--pe_model", type=str, default=None)

    # Server args
    g = parser.add_argument_group("server")
    g.add_argument("--host", type=str, default="0.0.0.0")
    g.add_argument("--port", type=int, default=8000)
    g.add_argument("--save_dir", type=str, default=None)
    g.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")

    return parser.parse_args()


def main():
    global PIPELINE, DEVICE, OUTPUT_DIR, REWRITER, ARGS

    args = parse_args()
    ARGS = args

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)

    logger.info("Loading pipeline (this may take a while)...")
    t0 = time.time()
    PIPELINE = build_pipeline(args, DEVICE)
    logger.info(
        "Pipeline loaded in %.1fs: %s",
        time.time() - t0,
        "Bernini" if _is_full_bernini_pipeline() else "Bernini-R",
    )

    OUTPUT_DIR = args.save_dir or os.path.join(os.getcwd(), "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.use_pe:
        from bernini.prompt_enhancer import PromptEnhancer
        REWRITER = PromptEnhancer(model=args.pe_model)
        logger.info("Prompt enhancer initialized (model=%s)", args.pe_model)

    logger.info(
        "Starting API server on http://%s:%s  (docs at http://%s:%s/docs)",
        args.host, args.port, args.host, args.port,
    )
    # Pass the app object directly (not a string import path) so uvicorn
    # does NOT respawn a subprocess that would lose the loaded PIPELINE.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
