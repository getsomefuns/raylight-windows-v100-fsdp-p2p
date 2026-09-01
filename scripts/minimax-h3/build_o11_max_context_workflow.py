#!/usr/bin/env python3
"""Build and validate the O11 maximum-context Text Encoder workflow.

The distributable workflow contains filenames and a meaningful 7000-character
prompt, but no absolute machine paths.  The local manifest records hashes and
media metadata and belongs under ``.local/artifacts``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMFY_ROOT = REPO_ROOT.parent / "ComfyUI"
DEFAULT_IMAGES = (
    r"MiniMax H3 VS Seedance\kpop\图片1.png",
    r"MiniMax H3 VS Seedance\kpop\Kpop成员.png",
    "frame_00m00s_no_watermark.png",
    "frame_00m04s_no_watermark.png",
    "minimax_h3_ref2va_green_robots.jpg",
    "minimax_h3_i2v_spear_portals.jpg",
    "late_checkout_clean.png",
)
DEFAULT_VIDEO = r"MiniMax H3 VS Seedance\kpop\o11_kpop_reference_15s_24fps.mp4"
DEFAULT_CHECKPOINT = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = r"MMH3\minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = r"MMH3\minimax_h3_audio_vae_fp32.safetensors"


def normalize_prompt(prompt: str) -> str:
    """Normalize only newline spelling; do not hide whitespace padding."""

    return prompt.replace("\r\n", "\n").replace("\r", "\n")


def build_acceptance_prompt() -> str:
    """Return a deterministic, semantically useful 7000-character prompt."""

    reference_notes = [
        "<Picture 1>确定舞台空间、青绿色主灯、紫色星云背景与横向电影构图；保留其舞台纵深、反射地面和大型机械布景，但不要照搬任何文字标识。",
        "<Picture 2>确定五人组合的脸部身份、发型差异、身高比例、服装轮廓和团队站位；任何景别都应保持成员可辨识，不互换五官、发色或饰品。",
        "<Picture 3>提供开场广角镜头的真实光影、镜头高度和表演队形；将人物运动方向、视线关系和舞台左右方位作为连续性基准。",
        "<Picture 4>提供中段近景的皮肤质感、妆容、轮廓光与快速动作下的发丝表现；表情自然有力量，嘴型与音乐节奏协调。",
        "<Picture 5>定义绿色机械守卫的硬表面材质、能量纹路和体量感；它们只作为舞台机械装置参与编舞，不攻击、不遮挡主角。",
        "<Picture 6>定义金色传送门、长矛阵列和空间涟漪的视觉语汇；所有武器元素转化为安全的全息舞美，不刺向人物。",
        "<Picture 7>定义安静室内段落的暖色实景、自然窗光和克制情绪；它用于短暂叙事停顿，并与宏大舞台形成清晰但连贯的反差。",
        "<Video 1>提供十五秒真实舞蹈的动作节奏、运镜惯性、群体同步和服装动态。提取其运动语言而非逐帧复制，保持舞步重心、肢体因果与镜头速度可信。",
    ]
    subjects = [
        "五位表演者从背光剪影走入主舞台",
        "领舞抬手引导其余成员形成对称箭头队形",
        "两侧成员交叉换位而中央人物保持目光锁定镜头",
        "全员完成连续转身、踏步和肩部律动",
        "机械守卫随低频鼓点展开肩甲并点亮能量纹路",
        "金色全息长矛在远景中组成安全的放射状拱门",
        "舞台地面沿脚步扩散细腻光波",
        "组合在副歌处完成一次整齐但有个体差异的定格",
        "镜头进入暖色房间捕捉成员短暂交流",
        "人物重新穿过传送门回到开阔舞台",
        "服装边缘和发丝响应快速转身产生真实惯性",
        "背景机械结构缓慢旋转以强化空间尺度",
    ]
    cameras = [
        "使用低机位二十四毫米广角平稳推进",
        "使用五十毫米中景沿舞台弧线横移",
        "使用八十五毫米近景捕捉眼神与细微表情",
        "从高位俯拍缓慢下降到人物胸口高度",
        "环绕队形半周并保持地平线稳定",
        "先跟随脚步横移再自然抬升到全身构图",
        "以长焦压缩空间呈现前后层次",
        "用克制手持感靠近但避免无意义抖动",
    ]
    lights = [
        "青绿色主光与紫色体积雾形成清晰冷暖分层",
        "金色轮廓光只勾勒人物和机械边缘而不过曝",
        "暖白窗光在室内保持真实肤色和柔和阴影",
        "地面反射保留细节，不形成镜像复制或黑色死区",
        "传送门亮度随音乐渐变，避免突然闪烁和高光剪切",
        "远处星云提供缓慢环境变化，不抢夺主体注意力",
    ]
    sounds = [
        "声音包含清晰鼓点、鞋底触地、衣料摩擦和克制电子低频",
        "机械动作使用精确齿轮声和伺服声，不加入爆炸或枪声",
        "室内停顿保留轻微呼吸、房间空气声和远处城市底噪",
        "副歌加入宽阔合成器和短促人声切片，响度始终稳定",
        "转场使用由前一镜头环境声延续而来的声桥",
    ]
    continuity = [
        "保持服装纹理、首饰位置、发型分缝和人物左右关系连续",
        "每个动作必须由前一姿势自然发力并在下一姿势完成重心转移",
        "切镜遵守视线匹配、运动匹配和一百八十度轴线",
        "避免多余手指、肢体穿插、面部漂移、重复人物和背景跳变",
        "画面具有真实运动模糊，但眼睛、嘴部和手部关键帧保持清楚",
        "景深随焦距合理变化，不把所有层面同时锐化成平面贴图",
    ]

    opening = (
        "制作一支具有完整叙事和专业舞台调度的电影级科幻音乐影像。"
        "成片为十六比九横屏，二十四帧每秒，画面写实、细节稳定、运动连续。"
        "七张图片分别承担场景、身份、光影、机械设计、空间特效和情绪参考；"
        "十五秒视频只承担动作与镜头节奏参考。所有引用都必须被实际理解，不得简单拼贴。"
    )
    paragraphs = [opening, *reference_notes]
    for index in range(30):
        start = index * 0.5
        end = start + 0.5
        paragraphs.append(
            f"镜头段落{index + 1:02d}（{start:04.1f}至{end:04.1f}秒）："
            f"{subjects[index % len(subjects)]}，{cameras[index % len(cameras)]}；"
            f"{lights[index % len(lights)]}。{sounds[index % len(sounds)]}；"
            f"{continuity[index % len(continuity)]}。"
            "主体动作、摄影机运动、舞台机关和音乐重拍需要在同一时间点互相呼应，"
            "构图始终保留清楚的前景、中景和背景，转场前给动作留下完整收势。"
        )
    paragraphs.append(
        "全片最终要求：人物身份从远景到特写始终稳定，面部和手部结构可信，"
        "服装与机械材质具有可追踪的纹理细节；声音没有爆音、断裂或错误口型；"
        "摄影机不穿模、不瞬移，舞台左右关系不翻转。结尾让五人面向镜头完成最后一个动作，"
        "机械守卫在后方收拢灯光，金色传送门逐渐变暗，保留一秒稳定尾帧用于后续衔接。"
    )
    paragraphs.extend(
        [
            "人物一致性补充：逐镜核对五名成员的眉眼比例、鼻梁轮廓、唇形、耳饰、发色、刘海方向和服装层次。侧脸、低头、快速旋转以及被彩色灯光照亮时仍应辨认出同一人物，不能把不同参考图的面部特征混合到一张脸上。",
            "动作物理补充：脚掌接触地面后再传递到膝、髋、肩和手臂，起跳、落地、转身都有清楚的重心轨迹。宽松衣料、金属挂件和长发按照速度产生延迟与回弹，避免漂浮、黏连、穿透或突然静止。",
            "摄影语言补充：广角镜头强调队形和空间，中焦镜头交代成员关系，长焦特写只在情绪节点使用。推进、横移、升降和环绕都由稳定轨道完成；快速动作可以保留适量运动模糊，但不使用无动机缩放、滚转或甩镜。",
            "舞台空间补充：前景可出现虚化灯架和透明全息结构，中景始终属于表演者，背景机械守卫与传送门保持足够距离。所有物体拥有接触阴影、遮挡关系和一致透视，镜头移动时视差必须连续可信。",
            "材质渲染补充：皮肤保留细微纹理和自然高光，布料区分缎面、网纱、皮革与金属装饰，机械表面展示涂层磨砂、边缘反射和细小接缝。不要使用塑料蜡像肤质、过度锐化、油画笔触或随机闪烁纹理。",
            "灯光曝光补充：人脸曝光优先，青绿主光、紫色环境光和金色轮廓光分别承担塑形、空间和高潮提示。黑位保留服装细节，传送门中心不过曝，雾气只显示光束层次，不覆盖眼睛、嘴部和关键手势。",
            "剪辑连续补充：每次切换都选择动作中段或音乐重拍，保证切前与切后人物朝向、速度和肢体位置相互对应。室内段通过声桥与颜色渐变进入，再由人物触碰金色微光返回舞台，避免硬切造成时间或地点不明。",
            "声音设计补充：主音乐保持完整节拍与动态范围，鞋底、衣料、呼吸和机械伺服声只增强画面动作。环境声随空间大小改变混响，室内更近更干，舞台更宽更深；任何声效都不能盖住旋律或产生削波。",
            "画面清洁补充：禁止字幕、水印、品牌文字、随机符号、错位徽标和无法阅读的屏幕内容。背景观众不生成重复脸，机械结构不长出多余肢体，金色长矛保持为远处舞美轮廓，不接触表演者身体。",
            "输出质量补充：保持二十四帧节奏稳定，没有跳帧、倒放、冻结或周期性重复。画面边缘不出现黑条、拉伸和裁切错误；色彩从开场冷色、室内暖色到结尾冷暖融合形成完整弧线，并为下一段保留清楚可延续的姿势。",
        ]
    )
    prompt = normalize_prompt("\n\n".join(paragraphs))
    if len(prompt) < 7000:
        raise AssertionError(f"semantic prompt body is unexpectedly short: {len(prompt)}")
    return prompt[:6999].rstrip("\n ") + "。"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_image(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        return {"width": int(image.width), "height": int(image.height)}


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frames": int(stream["nb_read_frames"]),
        "duration_seconds": float(payload["format"]["duration"]),
    }


def validate_media(
    image_paths: Sequence[Path],
    video_path: Path,
    *,
    image_names: Sequence[str] | None = None,
    video_name: str | None = None,
) -> dict[str, Any]:
    if len(image_paths) != 7:
        raise ValueError(f"O11 requires exactly seven images, found {len(image_paths)}")
    if image_names is not None and len(image_names) != 7:
        raise ValueError("image_names must contain exactly seven relative names")
    relative_images = tuple(image_names or (path.name for path in image_paths))
    rows = []
    for path, relative_name in zip(image_paths, relative_images, strict=True):
        metadata = probe_image(path)
        if min(metadata["width"], metadata["height"]) < 480:
            raise ValueError(f"image short edge must be at least 480 pixels: {path}")
        rows.append({"file": relative_name, "sha256": sha256_file(path), **metadata})
    if len({row["sha256"] for row in rows}) != 7:
        raise ValueError("all seven images must have a unique SHA-256")

    video = probe_video(video_path)
    frame_tolerance = 1.0 / video["fps"]
    if not math.isclose(video["duration_seconds"], 15.0, abs_tol=frame_tolerance):
        raise ValueError(
            "reference video must be 15.0 seconds within one source frame; "
            f"found {video['duration_seconds']:.6f}"
        )
    if min(video["width"], video["height"]) < 480:
        raise ValueError("reference video short edge must be at least 480 pixels")
    return {
        "images": rows,
        "video": {
            "file": video_name or video_path.name,
            "sha256": sha256_file(video_path),
            **video,
        },
    }


def build_api_prompt(
    *,
    checkpoint: str,
    video_vae: str,
    audio_vae: str,
    image_files: Sequence[str],
    video_file: str,
    attention_backend: str,
) -> dict[str, dict[str, Any]]:
    if len(image_files) != 7:
        raise ValueError("the O11 API prompt requires exactly seven images")
    prompt: dict[str, dict[str, Any]] = {
        "0": {
            "class_type": "RayInitializer",
            "inputs": {
                "ray_cluster_address": "local",
                "ray_cluster_namespace": "default",
                "GPU": 2,
                "ulysses_degree": 2,
                "ring_degree": 1,
                "cfg_degree": 1,
                "dp_degree": 1,
                "sync_ulysses": True,
                "clear_vram_after_sampling": False,
                "FSDP": True,
                "FSDP_CPU_OFFLOAD": True,
                "XFuser_attention": "TORCH_EFFICIENT",
                "skip_comm_test": True,
                "use_mmap": True,
            },
        },
        "1": {
            "class_type": "RayMiniMaxH3DualGPUCLIPLoader",
            "inputs": {
                "clip_name": checkpoint,
                "execution_mode": "tp2",
                "text_encoder_attention": attention_backend,
                "ray_actors_init": ["0", 0],
            },
        },
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}},
        "20": {"class_type": "LoadVideo", "inputs": {"file": video_file}},
        "21": {"class_type": "GetVideoComponents", "inputs": {"video": ["20", 0]}},
    }
    for index, image_file in enumerate(image_files):
        prompt[str(10 + index)] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_file},
        }
    source_inputs: dict[str, Any] = {
        "clip": ["1", 0],
        "vae": ["2", 0],
        "audio_vae": ["3", 0],
        "prompt": build_acceptance_prompt(),
        "width": 1056,
        "height": 608,
        "length": 362,
        "ref_image_size": "match",
        "ref_videos.ref_video_0": ["21", 0],
    }
    for index in range(7):
        source_inputs[f"ref_images.ref_image_{index}"] = [str(10 + index), 0]
    prompt["30"] = {
        "class_type": "MiniMaxH3ReferenceToVideo",
        "inputs": source_inputs,
    }
    prompt["31"] = {"class_type": "PreviewAny", "inputs": {"source": ["30", 0]}}
    return prompt


def _ui_node(
    node_id: int,
    node_type: str,
    pos: tuple[int, int],
    size: tuple[int, int],
    *,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    widgets: list[Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": node_id,
        "type": node_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": node_id,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"Node name for S&R": node_type},
        "widgets_values": list(widgets or []),
    }
    if title:
        node["title"] = title
    return node


def build_ui_workflow(
    *,
    checkpoint: str,
    video_vae: str,
    audio_vae: str,
    image_files: Sequence[str],
    video_file: str,
    attention_backend: str,
) -> dict[str, Any]:
    if len(image_files) != 7:
        raise ValueError("the O11 UI workflow requires exactly seven images")

    links: list[list[Any]] = []
    next_link = 1

    def link(origin: int, origin_slot: int, target: int, target_slot: int, kind: str) -> int:
        nonlocal next_link
        link_id = next_link
        next_link += 1
        links.append([link_id, origin, origin_slot, target, target_slot, kind])
        return link_id

    init_to_loader = link(1, 0, 2, 0, "RAY_ACTORS_INIT")
    clip_to_source = link(2, 0, 14, 0, "CLIP")
    video_vae_to_source = link(3, 0, 14, 1, "VAE")
    audio_vae_to_source = link(4, 0, 14, 2, "VAE")
    image_links = [link(5 + index, 0, 14, 3 + index, "IMAGE") for index in range(7)]
    video_to_components = link(12, 0, 13, 0, "VIDEO")
    components_to_source = link(13, 0, 14, 10, "IMAGE")
    conditioning_to_preview = link(14, 0, 15, 0, "*")

    nodes = [
        _ui_node(
            1,
            "RayInitializer",
            (-1440, -100),
            (320, 370),
            inputs=[{"name": "wait_for", "type": "*", "link": None}],
            outputs=[
                {"name": "ray_actors_init", "type": "RAY_ACTORS_INIT", "links": [init_to_loader]}
            ],
            widgets=[
                "local",
                "default",
                2,
                2,
                1,
                1,
                1,
                True,
                False,
                True,
                True,
                "TORCH_EFFICIENT",
                True,
                True,
            ],
        ),
        _ui_node(
            2,
            "RayMiniMaxH3DualGPUCLIPLoader",
            (-1050, -90),
            (610, 140),
            inputs=[
                {"name": "ray_actors_init", "type": "RAY_ACTORS_INIT", "link": init_to_loader}
            ],
            outputs=[
                {"name": "clip", "type": "CLIP", "links": [clip_to_source]},
                {"name": "ray_actors_init", "type": "RAY_ACTORS_INIT", "links": None},
            ],
            widgets=[checkpoint, "tp2", attention_backend],
            title="MiniMax H3 Text Encoder (Dual V100 TP2 / O11)",
        ),
        _ui_node(
            3,
            "VAELoader",
            (-1050, 100),
            (610, 70),
            inputs=[],
            outputs=[{"name": "VAE", "type": "VAE", "links": [video_vae_to_source]}],
            widgets=[video_vae],
            title="Video VAE",
        ),
        _ui_node(
            4,
            "VAELoader",
            (-1050, 210),
            (610, 70),
            inputs=[],
            outputs=[{"name": "VAE", "type": "VAE", "links": [audio_vae_to_source]}],
            widgets=[audio_vae],
            title="Audio VAE",
        ),
    ]
    for index, (image_file, image_link) in enumerate(zip(image_files, image_links, strict=True)):
        nodes.append(
            _ui_node(
                5 + index,
                "LoadImage",
                (-1440 + (index % 4) * 330, 440 + (index // 4) * 390),
                (290, 330),
                inputs=[],
                outputs=[
                    {"name": "IMAGE", "type": "IMAGE", "links": [image_link]},
                    {"name": "MASK", "type": "MASK", "links": None},
                ],
                widgets=[image_file, "image"],
                title=f"Reference Picture {index + 1}",
            )
        )
    nodes.extend(
        [
            _ui_node(
                12,
                "LoadVideo",
                (-1440, 1230),
                (420, 120),
                inputs=[],
                outputs=[{"name": "VIDEO", "type": "VIDEO", "links": [video_to_components]}],
                widgets=[video_file],
                title="15.0s Reference Video",
            ),
            _ui_node(
                13,
                "GetVideoComponents",
                (-950, 1230),
                (260, 130),
                inputs=[{"name": "video", "type": "VIDEO", "link": video_to_components}],
                outputs=[
                    {"name": "images", "type": "IMAGE", "links": [components_to_source]},
                    {"name": "audio", "type": "AUDIO", "links": None},
                    {"name": "fps", "type": "FLOAT", "links": None},
                    {"name": "bit_depth", "type": "INT", "links": None},
                ],
            ),
        ]
    )
    source_inputs = [
        {"name": "clip", "type": "CLIP", "link": clip_to_source},
        {"name": "vae", "type": "VAE", "link": video_vae_to_source},
        {"name": "audio_vae", "type": "VAE", "link": audio_vae_to_source},
    ]
    source_inputs.extend(
        {
            "label": f"ref_image_{index}",
            "name": f"ref_images.ref_image_{index}",
            "shape": 7,
            "type": "IMAGE",
            "link": image_links[index],
        }
        for index in range(7)
    )
    source_inputs.append(
        {
            "label": "ref_video_0",
            "name": "ref_videos.ref_video_0",
            "shape": 7,
            "type": "IMAGE",
            "link": components_to_source,
        }
    )
    source_inputs.extend(
        [
            {"name": "prompt", "type": "STRING", "widget": {"name": "prompt"}, "link": None},
            {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
            {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
            {"name": "length", "type": "INT", "widget": {"name": "length"}, "link": None},
            {
                "name": "ref_image_size",
                "type": "COMBO",
                "widget": {"name": "ref_image_size"},
                "link": None,
            },
        ]
    )
    nodes.extend(
        [
            _ui_node(
                14,
                "MiniMaxH3ReferenceToVideo",
                (-300, -90),
                (520, 790),
                inputs=source_inputs,
                outputs=[
                    {"name": "positive", "type": "CONDITIONING", "links": [conditioning_to_preview]},
                    {"name": "LATENT", "type": "LATENT", "links": None},
                ],
                widgets=[build_acceptance_prompt(), 1056, 608, 362, "match"],
                title="O11 Maximum Context (conditioning only)",
            ),
            _ui_node(
                15,
                "PreviewAny",
                (300, -90),
                (300, 120),
                inputs=[{"name": "source", "type": "*", "link": conditioning_to_preview}],
                outputs=[],
                title="Conditioning Acceptance Output",
            ),
        ]
    )
    return {
        "last_node_id": 15,
        "last_link_id": next_link - 1,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.7, "offset": [1450, 250]}},
        "version": 0.4,
    }


def _write_json(path: Path, payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--image", action="append", dest="images")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--video-vae", default=DEFAULT_VIDEO_VAE)
    parser.add_argument("--audio-vae", default=DEFAULT_AUDIO_VAE)
    parser.add_argument(
        "--attention-backend",
        choices=("AUTO", "XFORMERS_GQA", "TORCH_EFFICIENT"),
        default="AUTO",
    )
    parser.add_argument(
        "--workflow-output",
        type=Path,
        default=REPO_ROOT / "example_workflows" / "MMH3_TextEncoder_O11_MaxContext.json",
    )
    parser.add_argument(
        "--api-output",
        type=Path,
        default=REPO_ROOT / ".local" / "artifacts" / "o11" / "max-context-api.json",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=REPO_ROOT / ".local" / "artifacts" / "o11" / "o11-boundary-input-manifest.local.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    image_files = tuple(args.images or DEFAULT_IMAGES)
    input_root = args.comfy_root / "input"
    image_paths = [input_root / name for name in image_files]
    video_path = input_root / args.video
    media = validate_media(
        image_paths,
        video_path,
        image_names=image_files,
        video_name=args.video,
    )
    prompt_text = build_acceptance_prompt()

    api_prompt = build_api_prompt(
        checkpoint=args.checkpoint,
        video_vae=args.video_vae,
        audio_vae=args.audio_vae,
        image_files=image_files,
        video_file=args.video,
        attention_backend=args.attention_backend,
    )
    ui_workflow = build_ui_workflow(
        checkpoint=args.checkpoint,
        video_vae=args.video_vae,
        audio_vae=args.audio_vae,
        image_files=image_files,
        video_file=args.video,
        attention_backend=args.attention_backend,
    )
    api_sha256 = _write_json(args.api_output, api_prompt)
    workflow_sha256 = _write_json(args.workflow_output, ui_workflow)
    manifest = {
        "schema_version": 1,
        "purpose": "O11 maximum-context Text Encoder acceptance",
        "canvas": {"width": 1056, "height": 608, "length": 362, "fps": 24},
        "prompt": {
            "normalized_characters": len(prompt_text),
            "sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        },
        "media": media,
        "workflow": {"sha256": workflow_sha256},
        "api_prompt": {"sha256": api_sha256},
    }
    _write_json(args.manifest_output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
