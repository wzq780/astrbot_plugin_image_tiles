# -*- coding: utf-8 -*-
"""识图增强：把大图切成 N×N 块再交给模型，等于把可用的细节放大 N² 倍。

为什么需要它
------------
不少多模态接口对图片有硬上限（DeepSeek 视觉大约把每张图压到 1300×1300
像素、最多 1024 token，AstrBot 自身还会先按 image_compress_options.max_size
再压一次）。一张手机原图经过这两道压缩，角落的小字、屏幕上的数字、表情里的
细节基本全糊了。

本插件在 ``on_llm_request`` 阶段（priority=1，早于其它处理器）把待发送的图
按 2×2 / 3×3 切开，逐块压缩到「每块最长边 ≤ tile_max_side、每块 ≤ tile_max_bytes」
后重新塞回 ``req.image_urls``，再用一条说明告诉模型这些块是一张图的局部。

设计要点
--------
* 小图不切：最长边小于 ``min_side_to_tile`` 时保持原样，否则纯属浪费 token。
* 先按 EXIF 摆正：手机竖拍的图方向信息在 EXIF 里，不转正会切歪。
* 每块独立限流：先按最长边缩放，再按字节数降质量，保证不吃到平台的体积上限。
* 按图分组说明：多张图时说明里会写清「前 N 块是图 1、接着 M 块是图 2」。
* 全程兜底：任何一步出错都退回原图，绝不让一次识图失败导致整个请求崩掉。
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

import aiohttp
from PIL import Image, ImageOps

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

# 没有配置时的兜底值，与 _conf_schema.json 保持一致
DEFAULTS = {
    "enabled": True,
    "grid": 2,
    "max_images": 2,
    "tile_max_side": 1200,
    "tile_max_bytes_kb": 420,
    "min_side_to_tile": 700,
    "overlap_percent": 0,
    "jpeg_quality": 88,
    "process_inline_images": False,
    "note_enabled": True,
}

JPEG_QUALITY_FLOOR = 55


def _as_int(value, default: int, low: int, high: int) -> int:
    """把配置读成夹在 [low, high] 之间的整数。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


@register(
    "astrbot_plugin_image_tiles",
    "wzq",
    "把大图切成多块再送给模型，显著改善小字与细节识别",
    "0.2.0",
)
class ImageTilesPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.conf = config or {}
        self._last_run = "还没处理过图片"

    # ------------------------------------------------------------------ 配置
    def _opt(self, key: str):
        try:
            value = self.conf.get(key)
        except Exception:
            value = None
        return DEFAULTS[key] if value is None else value

    def _settings(self) -> dict:
        return {
            "enabled": _as_bool(self._opt("enabled"), True),
            "grid": _as_int(self._opt("grid"), 2, 2, 3),
            "max_images": _as_int(self._opt("max_images"), 2, 1, 6),
            "tile_max_side": _as_int(self._opt("tile_max_side"), 1200, 256, 4096),
            "tile_max_bytes": _as_int(self._opt("tile_max_bytes_kb"), 420, 64, 4096) * 1024,
            "min_side_to_tile": _as_int(self._opt("min_side_to_tile"), 700, 0, 8192),
            "overlap_percent": _as_int(self._opt("overlap_percent"), 0, 0, 10),
            "jpeg_quality": _as_int(self._opt("jpeg_quality"), 88, 50, 95),
            "process_inline_images": _as_bool(self._opt("process_inline_images"), False),
            "note_enabled": _as_bool(self._opt("note_enabled"), True),
        }

    # ------------------------------------------------------------------ 主流程
    @filter.on_llm_request(priority=1)
    async def tile_images(self, event: AstrMessageEvent, req: ProviderRequest):
        cfg = self._settings()
        if not cfg["enabled"]:
            return

        urls = list(getattr(req, "image_urls", None) or [])
        if not urls:
            return

        targets = [u for u in urls if cfg["process_inline_images"] or not str(u).startswith("data:")]
        if not targets:
            return
        targets = targets[: cfg["max_images"]]

        groups: list[int] = []
        tiles: list[str] = []
        for url in targets:
            raw = await self._fetch(url)
            if not raw:
                logger.info("[image_tiles] 取图失败，这张保持原样")
                continue
            made = await asyncio.to_thread(self._split, raw, cfg)
            if not made:
                logger.info("[image_tiles] 图片太小或解不开，保持原样")
                continue
            groups.append(len(made))
            tiles.extend(made)

        if not tiles:
            return

        kept = [u for u in urls if u not in targets]
        req.image_urls = kept + tiles
        self._last_run = (
            f"{len(groups)} 张图切成 {len(tiles)} 块"
            f"（{cfg['grid']}×{cfg['grid']}，每块 ≤{cfg['tile_max_side']}px / "
            f"{cfg['tile_max_bytes'] // 1024}KB）"
        )
        logger.info(f"[image_tiles] {self._last_run}")

        if cfg["note_enabled"]:
            self._append_note(req, cfg, groups)

    def _append_note(self, req: ProviderRequest, cfg: dict, groups: list[int]) -> None:
        """把「这些块是同一张图」的说明塞进本轮上下文。"""
        side = f"{cfg['grid']}×{cfg['grid']}"
        if len(groups) == 1:
            layout = f"用户发来的图已切成 {side} 共 {groups[0]} 块"
        else:
            parts, start = [], 1
            for index, count in enumerate(groups, start=1):
                parts.append(f"图 {index} 是第 {start}~{start + count - 1} 块")
                start += count
            layout = f"用户发来 {len(groups)} 张图，各切成 {side} 块，" + "，".join(parts)
        note = (
            f"[图片追加] {layout}，按左→右、上→下的顺序排在后面，"
            "同一组的块拼起来才是完整的一张图。"
            "看细节（文字、数字、表情、画面角落）时以这些分块为准，"
            "不要因为只看到局部就说图片不完整或说没收到。"
        )
        try:
            parts = getattr(req, "extra_user_content_parts", None)
            if parts is None:
                raise AttributeError("no extra_user_content_parts")
            parts.append({"type": "text", "text": note})
        except Exception:
            req.system_prompt = f"{req.system_prompt or ''}\n{note}"

    # ------------------------------------------------------------------ 取图
    async def _fetch(self, url) -> bytes | None:
        """支持 http(s)、data:、base64://、file:// 与本地路径。"""
        url = str(url or "").strip()
        if not url:
            return None
        try:
            if url.startswith("base64://"):
                return base64.b64decode(url[len("base64://"):])
            if url.startswith("data:"):
                _, _, payload = url.partition(",")
                return base64.b64decode(payload)
            if url.startswith("file://"):
                return Path(url[len("file://"):]).read_bytes()
            if url.startswith(("http://", "https://")):
                timeout = aiohttp.ClientTimeout(total=20)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            logger.warning(f"[image_tiles] 下载失败 {resp.status}: {url[:80]}")
                            return None
                        return await resp.read()
            path = Path(url)
            if path.is_file():
                return path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[image_tiles] 取图失败: {type(exc).__name__}")
        return None

    # ------------------------------------------------------------------ 切图
    @staticmethod
    def _split(raw: bytes, cfg: dict) -> list[str]:
        """返回 data URL 列表；返回空列表表示这张图不值得切。"""
        try:
            image = Image.open(io.BytesIO(raw))
            image.seek(0)
            image = ImageOps.exif_transpose(image)  # 手机竖拍靠 EXIF，不转正会切歪
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGBA")
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                background.alpha_composite(image)
                image = background
            image = image.convert("RGB")
        except Exception:
            return []

        width, height = image.size
        if max(width, height) < cfg["min_side_to_tile"]:
            return []

        grid = cfg["grid"]
        overlap = int(cfg["overlap_percent"] / 100 * min(width, height) / grid)
        out: list[str] = []
        for row in range(grid):
            for col in range(grid):
                left = max(0, round(col * width / grid) - overlap)
                right = min(width, round((col + 1) * width / grid) + overlap)
                top = max(0, round(row * height / grid) - overlap)
                bottom = min(height, round((row + 1) * height / grid) + overlap)
                tile = image.crop((left, top, right, bottom))
                longest = max(tile.size)
                if longest > cfg["tile_max_side"]:
                    scale = cfg["tile_max_side"] / longest
                    tile = tile.resize(
                        (max(1, int(tile.width * scale)), max(1, int(tile.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                data = _encode_jpeg(tile, cfg["tile_max_bytes"], cfg["jpeg_quality"])
                out.append("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
        return out

    # ------------------------------------------------------------------ 指令
    @filter.command("识图切块")
    async def show_status(self, event: AstrMessageEvent):
        """查看切图配置与最近一次处理结果"""
        cfg = self._settings()
        state = "开启" if cfg["enabled"] else "关闭"
        lines = [
            f"识图切块：{state}",
            f"网格 {cfg['grid']}×{cfg['grid']}，单次最多 {cfg['max_images']} 张",
            f"每块 ≤{cfg['tile_max_side']}px / {cfg['tile_max_bytes'] // 1024}KB，"
            f"小于 {cfg['min_side_to_tile']}px 的图不切",
            f"重叠 {cfg['overlap_percent']}%，内联图{'也处理' if cfg['process_inline_images'] else '跳过'}",
            f"最近一次：{self._last_run}",
        ]
        yield event.plain_result("\n".join(lines))


def _encode_jpeg(tile: Image.Image, max_bytes: int, quality: int) -> bytes:
    """在不超过 max_bytes 的前提下尽量保质量。"""
    current = quality
    while True:
        buf = io.BytesIO()
        tile.save(buf, format="JPEG", quality=current, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes or current <= JPEG_QUALITY_FLOOR:
            return data
        current -= 8
