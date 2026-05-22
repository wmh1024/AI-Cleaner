from __future__ import annotations

import json
import re
from typing import Literal

NlpStyle = Literal["academic", "general", "long_blog", "novel"]

# ── local classification markers ──────────────────────────────
academic_markers = (
    "摘要", "关键词", "引言", "结论", "参考文献", "本研究",
    "研究表明", "实验结果", "分析", "方法",
)
social_markers = (
    "点赞", "关注", "转发", "收藏", "分享", "评论",
    "博主", "粉丝", "推荐", "热门",
)
novel_markers = (
    r"第.{1,5}章",
    r"卷[一二三四五六七八九十\d]",
    "节选",
    "番外",
    "楔子",
    "尾声",
    "序章",
    "引子",
    "她说道",
    "他低声",
    "他叹了口气",
    "她微笑着",
    "他皱了皱眉",
    "她轻声说",
)


def classify_locally(text: str) -> NlpStyle:
    if sum(text.count(marker) for marker in academic_markers) >= 2:
        return "academic"
    novel_score = sum(len(re.findall(marker, text)) for marker in novel_markers)
    if novel_score >= 3:
        return "novel"
    if len(text) > 1200 or sum(text.count(marker) for marker in social_markers) >= 2:
        return "long_blog"
    return "general"


def normalize_nlp_style(raw: str | None) -> NlpStyle:
    if raw in ("academic", "general", "long_blog", "novel"):
        return raw
    return "general"


def parse_llm_classification(raw: str) -> NlpStyle | None:
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    style = data.get("style") or data.get("nlp_style") or data.get("category")
    if style in {"academic", "general", "long_blog", "novel"}:
        return style
    return None


def apply_nlp(
    text: str,
    style: NlpStyle,
    aggressive: bool = False,
    best_of_n: int = 10,
    seed: int | None = None,
) -> str:
    if style == "academic":
        from academic_cn import humanize_academic

        return humanize_academic(
            text,
            aggressive=aggressive,
            seed=seed,
            best_of_n=best_of_n,
        )
    from humanize_cn import humanize

    if style == "novel":
        result = humanize(
            text,
            scene="novel",
            style="novel",
            aggressive=aggressive,
            seed=seed,
            best_of_n=best_of_n,
            score_mode="fused",
        )
        from style_cn import transform_novel
        result = transform_novel(result)
        return result
    if style == "long_blog":
        return humanize(
            text,
            scene="general",
            aggressive=aggressive,
            seed=seed,
            best_of_n=best_of_n,
            score_mode="fused",
        )
    return humanize(
        text,
        scene="general",
        aggressive=aggressive,
        seed=seed,
        best_of_n=best_of_n,
    )
