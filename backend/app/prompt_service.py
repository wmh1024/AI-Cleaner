from __future__ import annotations

import re

from .constants import PROMPTS_DIR, SUPPORTED_PLATFORMS


def get_prompt(platform: str) -> str:
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}")
    path = PROMPTS_DIR / f"{platform}.md"
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def build_messages(
    platform: str,
    text: str,
    evaluator_feedback: str | None = None,
    iteration: int | None = None,
) -> list[dict[str, str]]:
    prompt = get_prompt(platform)
    suffix = build_iteration_suffix(platform, evaluator_feedback, iteration)
    return [
        {"role": "system", "content": prompt + suffix},
        {"role": "user", "content": f"原文：\n{text}"},
    ]


def build_iteration_suffix(
    platform: str,
    evaluator_feedback: str | None,
    iteration: int | None = None,
) -> str:
    if not evaluator_feedback:
        return ""

    iteration_line = f"这是第 {iteration} 轮修订。\n" if iteration else ""
    if platform == "novel":
        return (
            "\n\n迭代修订任务：\n"
            f"{iteration_line}"
            "上一轮文本已经完成基础小说改写，但仍需要继续降低 AIGC 检测风险。\n"
            "请优先根据下面的评估意见做局部修订，而不是重写成另一段剧情。\n\n"
            "上一轮检测风险评估意见：\n"
            f"{evaluator_feedback}\n\n"
            "本轮修订要求：\n"
            "1. 保持原有情节、人物关系、叙事视角、对话事实和场景顺序不变，不新增设定、伏笔或剧情事件。\n"
            "2. 重点降低小说机器感：减少完美闭环式总结、机械转场、对称排比、抽象情绪堆叠和模板化表情描写。\n"
            "3. 调整叙事节奏：让长短句、对白、动作和留白自然交错；避免每段都按“动作-心理-环境”同一节拍推进。\n"
            "4. 用更具体的动作、物件和感官线索承接原文已有信息，但不要凭空发明新道具、新人物或新冲突。\n"
            "5. 保留原文段落和引号结构，整体字数尽量控制在上一轮文本的 ±10% 内。\n"
            "6. 只输出修订后的小说正文，不输出分析、标题、说明、项目符号、markdown 或 emoji。"
        )

    return (
        "\n\n迭代修订任务：\n"
        f"{iteration_line}"
        "上一轮文本已经完成基础改写，但仍需要继续降低 AIGC 检测风险。\n"
        "请优先根据下面的评估意见修订，而不是重新写一篇新文章。\n\n"
        "上一轮检测风险评估意见：\n"
        f"{evaluator_feedback}\n\n"
        "本轮修订要求：\n"
        "1. 保持原文事实、数据、术语、研究对象和结论不变，不新增未经原文支持的信息。\n"
        "2. 重点降低机器感：减少模板化连接词、过度工整的并列结构、连续抽象名词堆叠和过强总结腔。\n"
        "3. 调整句子节奏：适当拆分过长句，穿插中等长度句；避免每句都呈现同一种逻辑结构。\n"
        "4. 让表达更像人工论文修改：保留学术严谨性，但使用更自然的转述、解释性补充和轻微语序变化。\n"
        "5. 不要为了降重而堆砌口水词，不要明显扩写，整体字数尽量控制在上一轮文本的 ±10% 内。\n"
        "6. 只输出修订后的正文，不输出分析、标题、说明或项目符号。"
    )


def build_evaluator_messages(
    platform: str,
    original_text: str,
    current_text: str,
    detection_prompt: str,
) -> list[dict[str, str]]:
    if platform == "novel":
        return [
            {
                "role": "system",
                "content": (
                    "你是中文小说 AIGC 检测风险评估 agent。"
                    "你的任务不是润色或续写，而是结合本地检测器结果，找出当前小说文本中仍可能被检测为 AI 的叙事特征。"
                    "请只输出简短、可执行的修订意见，禁止输出改写后的正文。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请结合本地 humanize-chinese 检测结果，对下面小说文本做 AIGC 检测风险评估，并给出下一轮修订方向。\n\n"
                    "本地检测结果：\n"
                    f"{detection_prompt}\n\n"
                    "你需要重点检查：\n"
                    "1. 是否存在小说常见 AI 模板，如“眼中闪过一丝…、嘴角微微上扬、心中暗道、仿佛/宛如密集比喻”。\n"
                    "2. 是否存在段落推进过于均匀、每段都完整收束、转场过度解释或结尾上帝视角总结。\n"
                    "3. 对白后是否机械跟随动作/心理/环境描写，人物说话方式是否过于同质化。\n"
                    "4. 情绪是否依赖抽象形容词堆叠，而不是由动作、物件、停顿和感官细节承载。\n"
                    "5. 哪些句子应拆分、压短、留白、改成动作带对白，或删除 AI 腔连接词。\n\n"
                    "输出要求：\n"
                    "- 最多 6 条。\n"
                    "- 每条必须是具体可执行的修改建议。\n"
                    "- 优先引用本地检测结果中的风险词、风险句或统计特征。\n"
                    "- 不得改变剧情、人物关系、叙事视角、对话事实和场景顺序。\n\n"
                    f"原文：\n{original_text}\n\n当前改写：\n{current_text}"
                ),
            },
        ]

    return [
        {
            "role": "system",
            "content": (
                "你是中文论文 AIGC 检测风险评估 agent。"
                "你的任务不是润色文章，而是结合本地检测器结果，找出当前改写文本中仍可能被检测为 AI 的语言特征。"
                "请只输出简短、可执行的修订意见，禁止输出改写后的正文。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请结合本地 humanize-chinese 检测结果，对下面文本做 AIGC 检测风险评估，并给出下一轮修订方向。\n\n"
                "本地检测结果：\n"
                f"{detection_prompt}\n\n"
                "你需要重点检查：\n"
                "1. 是否存在模板化连接词或总结腔，如“首先、其次、最后、综上、显著意义、现实而迫切”等。\n"
                "2. 是否存在过于工整的并列结构、排比式表达或均匀句长。\n"
                "3. 是否存在抽象名词连续堆叠、动词弱化、表达过度凝练的问题。\n"
                "4. 是否缺少人工写作中常见的节奏变化、解释性转折和自然语序。\n"
                "5. 哪些句子应拆分、调序、换成更自然的学术转述。\n\n"
                "输出要求：\n"
                "- 最多 6 条。\n"
                "- 每条必须是具体可执行的修改建议。\n"
                "- 优先引用本地检测结果中的风险词、风险句或统计特征。\n"
                "- 不得改变事实、数据、术语和结论。\n\n"
                f"原文：\n{original_text}\n\n当前改写：\n{current_text}"
            ),
        },
    ]


ARTICLE_START_MARKERS = [
    "重写后：",
    "重写后",
    "重写后的文本：",
    "改写后：",
    "改写后",
    "重写后的文章：",
    "改写后的文本：",
    "改写后的文章：",
    "优化后：",
    "优化后",
    "优化后的文本：",
    "优化后的文章：",
    "修改后：",
    "修改后",
    "修改后的文本：",
    "修改后的文章：",
    "最终文本：",
    "最终版本：",
    "正文：",
]

ARTICLE_END_MARKERS = [
    "修改细节说明",
    "修改说明",
    "改写说明",
    "优化说明",
    "调整说明",
    "对照您的要求",
    "原文 AI 特征分析",
    "核心优化策略",
    "优化亮点说明",
]


def _strip_markdown_heading(line: str) -> str:
    return re.sub(r"^\s{0,3}#{1,6}\s*", "", line).strip()


def _normalize_marker_text(line: str) -> str:
    line = _strip_markdown_heading(line)
    line = line.strip().strip("*-_ `\t")
    line = re.sub(r"^\d+[.、．]\s*", "", line)
    return line.strip().strip("：:").strip()


def _looks_like_meta_intro(text: str) -> bool:
    compact = text.strip()
    if len(compact) > 120:
        return False
    meta_words = ["为您提供", "以下是", "下面是", "版本", "重写", "改写", "优化", "符合", "要求"]
    return any(word in compact for word in meta_words)


def clean_rewritten_text(raw_output: str) -> str:
    """Keep only the final rewritten article body from chatty model output."""
    text = raw_output.strip()
    if not text:
        return ""

    # Prefer explicit article-start markers and discard anything before them.
    best_start: int | None = None
    for marker in ARTICLE_START_MARKERS:
        patterns = [
            rf"(?:\*\*)?{re.escape(marker)}(?:\*\*)?",
            rf"(?:\*\*)?{re.escape(marker.rstrip('：'))}\s*[:：](?:\*\*)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match and (best_start is None or match.start() < best_start):
                best_start = match.end()
    if best_start is not None:
        text = text[best_start:].strip()

    # Remove trailing explanation sections such as "修改细节说明".
    earliest_end: int | None = None
    for marker in ARTICLE_END_MARKERS:
        pattern = rf"(?:^|\n)\s*(?:[-*_]{{3,}}\s*\n)?\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(?:\d+[.、．]\s*)?{re.escape(marker)}"
        match = re.search(pattern, text, flags=re.I)
        if match and (earliest_end is None or match.start() < earliest_end):
            earliest_end = match.start()
    if earliest_end is not None:
        text = text[:earliest_end].strip()

    lines = text.splitlines()

    # Drop leading meta sentence if the model says "这里为您提供..." before body.
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and _looks_like_meta_intro(lines[0]):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    # Drop standalone markdown separators left around the body.
    cleaned_lines = []
    for line in lines:
        if re.fullmatch(r"\s*[-*_]{3,}\s*", line):
            continue
        normalized = _normalize_marker_text(line)
        if normalized in {m.rstrip("：") for m in ARTICLE_START_MARKERS}:
            continue
        cleaned_lines.append(line.rstrip())

    text = "\n".join(cleaned_lines).strip()

    # If a model still leaks a numbered explanation block after the article, cut it.
    leak_patterns = [
        r"\n\s*1[.、．]\s*\*\*?剔除",
        r"\n\s*1[.、．]\s*\*\*?去除",
        r"\n\s*1[.、．]\s*\*\*?修改",
        r"\n\s*1[.、．]\s*\*\*?优化",
    ]
    for pattern in leak_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[: match.start()].strip()
            break

    return text.strip()


def extract_rewritten_text(platform: str, raw_output: str) -> str:
    return clean_rewritten_text(raw_output)
