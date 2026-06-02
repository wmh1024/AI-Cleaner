from __future__ import annotations

from datetime import datetime, timezone

from backend.app.aigc_detector import (
    detect_aigc_risk,
    detection_scene_for_platform,
    format_aigc_report_for_prompt,
)
from backend.app.diffing import build_diff, length_warnings
from backend.app.nlp.pipeline import choose_nlp_style
from backend.app.nlp.wrapper import classify_locally, parse_llm_classification
from backend.app.prompt_service import (
    build_evaluator_messages,
    build_iteration_suffix,
    clean_rewritten_text,
    extract_rewritten_text,
    get_prompt,
)
from backend.app.schemas import RewriteResponse
from backend.app.security import decrypt_secret, encrypt_secret


def test_encrypt_secret_roundtrip():
    encrypted = encrypt_secret("sk-test")
    assert encrypted and encrypted != "sk-test"
    assert decrypt_secret(encrypted) == "sk-test"


def test_length_warnings():
    assert length_warnings("短文本")
    assert length_warnings("字" * 1300)
    assert length_warnings("字" * 400) == []


def test_diff_marks_replace():
    spans = build_diff("人工智能发展", "人工智能稳步发展")
    assert any(span["kind"] in {"insert", "replace"} for span in spans)


def test_paperyy_extracts_final_article():
    raw = "1. **原文 AI 特征分析：**\n略\n4. **优化后的文章：**\n这里是正文"
    assert extract_rewritten_text("paperyy", raw) == "这里是正文"


def test_clean_rewritten_text_keeps_only_article_body():
    raw = """这里为您提供一个经过全面重写、打破僵化八股句式、过渡自然且符合您所有限制要求的版本。

**重写后的文本：**

让计算机读懂人类语言，离不开数学、计算机科学与语言学的交汇。

特征准备妥当后，各种算法轮番上阵。

---

**修改细节说明（对照您的要求）：**
1. **剔除程序化过渡词**：完全删除了原有的“首先、其次、最后”。
2. **打破僵化句式与学术套话**：替换刻板表达。
"""
    cleaned = clean_rewritten_text(raw)
    assert cleaned == "让计算机读懂人类语言，离不开数学、计算机科学与语言学的交汇。\n\n特征准备妥当后，各种算法轮番上阵。"
    assert "修改细节说明" not in cleaned
    assert "这里为您提供" not in cleaned


def test_clean_rewritten_text_strips_short_bold_marker():
    raw = """**修改后：**

自然语言处理是把计算机科学、数学以及语言学相融合的综合学科。
"""
    assert clean_rewritten_text(raw) == "自然语言处理是把计算机科学、数学以及语言学相融合的综合学科。"


def test_get_prompt_reads_supported_templates():
    assert "论文" in get_prompt("weipu")
    assert "AI 文章润色师" in get_prompt("paperyy")
    assert "资深小说编辑" in get_prompt("novel")


def test_novel_agent_uses_narrative_iteration_guidance():
    suffix = build_iteration_suffix("novel", "对白后动作过于机械。", iteration=2)
    assert "小说" in suffix
    assert "对白" in suffix
    assert "保留学术严谨性" not in suffix


def test_novel_evaluator_prompt_is_not_academic():
    messages = build_evaluator_messages("novel", "原文", "当前", "LR(novel)：80/100")
    combined = "\n".join(message["content"] for message in messages)
    assert "中文小说 AIGC 检测风险评估 agent" in combined
    assert "对白" in combined
    assert "不得改变剧情" in combined
    assert "中文论文 AIGC 检测风险评估 agent" not in combined


def test_nlp_classification_helpers():
    assert parse_llm_classification('{"style":"academic"}') == "academic"
    assert parse_llm_classification('{"style":"novel"}') == "novel"
    assert classify_locally("本文基于已有研究构建理论模型，并结合样本进行实证分析。") == "academic"
    assert classify_locally("第一章 雨夜\n\n“你终于来了。”她低声说。他看着窗外，没有回答。") == "novel"


async def test_nlp_pipeline_can_classify_without_llm():
    text = "本文基于已有研究构建理论模型，并结合样本进行实证分析。"
    assert await choose_nlp_style(text, "auto", None) == "academic"
    assert await choose_nlp_style(text, "manual", "unknown") == "academic"


def test_aigc_detector_formats_prompt_guidance():
    text = "首先，文本分类具有显著意义。其次，它在大数据时代具有重要价值。最后，系统可以实现高效处理。"
    report = detect_aigc_risk(text, scene="academic")
    prompt = format_aigc_report_for_prompt(report)
    assert report.score >= 0
    assert report.rule_score >= 0
    assert "humanize-chinese" in prompt
    assert "融合分数" in prompt
    assert "主要风险点" in prompt
    assert "修订动作" in prompt


def test_detection_scene_for_platform_routes_longform_modes():
    assert detection_scene_for_platform("novel") == "novel"
    assert detection_scene_for_platform("weipu") == "academic"
    assert detection_scene_for_platform(None) == "auto"


def test_rewrite_response_accepts_local_nlp_provider():
    response = RewriteResponse(
        id=1,
        original_text="原文",
        rewritten_text="改写",
        raw_output="改写",
        platform="weipu",
        provider="local",
        model="humanize-chinese",
        iterations=0,
        warnings=[],
        nlp_applied=True,
        nlp_style="general",
        diff=[],
        created_at=datetime.now(timezone.utc),
    )
    assert response.provider == "local"
