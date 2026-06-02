from __future__ import annotations

import asyncio
import json
import logging
import time
from uuid import uuid4
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .constants import (
    ANTHROPIC_MESSAGES_PATH,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    OPENAI_CHAT_COMPLETIONS_PATH,
)
from .aigc_detector import detect_aigc_risk, detection_scene_for_platform, format_aigc_report_for_prompt
from .database import configure_database, delete_history, get_history, insert_history, list_history
from .diffing import build_diff, length_warnings
from .nlp.pipeline import choose_nlp_style, rewrite_with_nlp, rewrite_with_nlp_style
from .prompt_service import build_evaluator_messages, build_messages, extract_rewritten_text
from .providers import get_provider
from .schemas import (
    HistoryItem,
    NlpRewriteRequest,
    RewriteRequest,
    RewriteResponse,
    SettingsPayload,
    SettingsTestRequest,
    SettingsTestResponse,
    SettingsView,
)
from .settings_service import load_settings, save_settings, settings_view
from .workflow import history_row_to_response, run_rewrite


logger = logging.getLogger(__name__)


app = FastAPI(title="AI-Cleaner", version="0.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NLP_STREAM_CHUNK_SIZE = 8
NLP_STREAM_DELAY_SECONDS = 0.006


@app.on_event("startup")
async def startup() -> None:
    configure_database()


def preview_request_url(provider: str, base_url: str | None) -> str:
    if provider == "openai":
        return (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/") + OPENAI_CHAT_COMPLETIONS_PATH
    return (base_url or DEFAULT_ANTHROPIC_BASE_URL).rstrip("/") + ANTHROPIC_MESSAGES_PATH


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def text_chunks(text: str, size: int = NLP_STREAM_CHUNK_SIZE) -> list[str]:
    chars = list(text)
    return ["".join(chars[index : index + size]) for index in range(0, len(chars), size)]


async def stream_text(event: str, text: str) -> AsyncIterator[str]:
    for chunk in text_chunks(text):
        yield sse(event, {"delta": chunk})
        await asyncio.sleep(NLP_STREAM_DELAY_SECONDS)


@app.get("/api/settings", response_model=SettingsView)
async def get_settings() -> SettingsView:
    return settings_view()


@app.put("/api/settings", response_model=SettingsView)
async def put_settings(payload: SettingsPayload) -> SettingsView:
    return settings_view(save_settings(payload))


@app.post("/api/settings/test", response_model=SettingsTestResponse)
async def test_settings(payload: SettingsTestRequest) -> SettingsTestResponse:
    settings = load_settings()
    provider_name = payload.provider or settings.provider
    model = payload.model or settings.model_for(provider_name)
    base_url = payload.base_url or settings.base_url_for(provider_name)
    request_url = preview_request_url(provider_name, base_url)
    resolved_api_key = payload.api_key.strip() if payload.api_key and payload.api_key.strip() else None
    provider = get_provider(
        settings,
        provider_name=provider_name,
        model=model,
        base_url=base_url,
        api_key=resolved_api_key or settings.api_key_for(provider_name),
    )
    start = time.perf_counter()
    try:
        text = await provider.complete(
            [
                {"role": "system", "content": "Reply with a short acknowledgement."},
                {"role": "user", "content": "hi"},
            ]
        )
        return SettingsTestResponse(
            ok=True,
            provider=provider_name,  # type: ignore[arg-type]
            request_url=request_url,
            latency_ms=round((time.perf_counter() - start) * 1000),
            response_preview=text[:240],
        )
    except Exception as exc:
        return SettingsTestResponse(
            ok=False,
            provider=provider_name,  # type: ignore[arg-type]
            request_url=request_url,
            latency_ms=round((time.perf_counter() - start) * 1000),
            error=str(exc),
        )


@app.post("/api/rewrite", response_model=RewriteResponse)
async def rewrite(payload: RewriteRequest) -> RewriteResponse:
    return await run_rewrite(payload)


@app.post("/api/nlp", response_model=RewriteResponse)
async def nlp_rewrite(payload: NlpRewriteRequest) -> RewriteResponse:
    warnings = length_warnings(payload.text)
    rewritten, style = await rewrite_with_nlp(
        payload.text,
        payload.nlp_mode,
        payload.nlp_style,
        aggressive=payload.aggressive,
        best_of_n=payload.best_of_n,
        seed=payload.seed,
    )
    diff = build_diff(payload.text, rewritten)
    created_at = datetime.now(timezone.utc).isoformat()
    record_id = insert_history(
        {
            "original_text": payload.text,
            "raw_output": rewritten,
            "rewritten_text": rewritten,
            "platform": payload.platform,
            "provider": "local",
            "model": "humanize-chinese",
            "iterations": 0,
            "warnings": warnings,
            "nlp_applied": True,
            "nlp_style": style,
            "diff": diff,
            "created_at": created_at,
        }
    )
    row = get_history(record_id)
    if row is None:
        raise RuntimeError("History record was not persisted.")
    return history_row_to_response(row)


async def nlp_stream_events(payload: NlpRewriteRequest) -> AsyncIterator[str]:
    stream_id = f"nlp-{uuid4().hex[:12]}"
    start = time.perf_counter()
    logger.info("NLP stream started stream_id=%s text_chars=%s", stream_id, len(payload.text))
    try:
        yield sse("stream_started", {"stream_id": stream_id})
        warnings = length_warnings(payload.text)
        yield sse("node_started", {"node": "validate_length", "warnings": warnings, "stream_id": stream_id})
        yield sse("node_started", {"node": "optional_nlp_classify", "stream_id": stream_id})
        yield sse("node_started", {"node": "optional_nlp_rewrite", "stream_id": stream_id})
        rewritten, style = await rewrite_with_nlp(
            payload.text,
            payload.nlp_mode,
            payload.nlp_style,
            aggressive=payload.aggressive,
            best_of_n=payload.best_of_n,
            seed=payload.seed,
        )

        yield sse("nlp_stream_started", {"style": style})
        async for event in stream_text("nlp_delta", rewritten):
            yield event
        yield sse("nlp_result", {"style": style, "text": rewritten})

        yield sse("node_started", {"node": "build_diff"})
        diff = build_diff(payload.text, rewritten)
        yield sse("diff_ready", {"diff": diff})

        yield sse("node_started", {"node": "persist_record"})
        created_at = datetime.now(timezone.utc).isoformat()
        record_id = insert_history(
            {
                "original_text": payload.text,
                "raw_output": rewritten,
                "rewritten_text": rewritten,
                "platform": payload.platform,
                "provider": "local",
                "model": "humanize-chinese",
                "iterations": 0,
                "warnings": warnings,
                "nlp_applied": True,
                "nlp_style": style,
                "diff": diff,
                "created_at": created_at,
            }
        )
        yield sse(
            "done",
            {
                "id": record_id,
                "text": rewritten,
                "raw_output": rewritten,
                "warnings": warnings,
                "nlp_style": style,
                "created_at": created_at,
                "stream_id": stream_id,
            },
        )
        logger.info("NLP stream finished stream_id=%s elapsed_ms=%s", stream_id, round((time.perf_counter() - start) * 1000))
    except Exception as exc:
        logger.exception("NLP stream failed stream_id=%s elapsed_ms=%s", stream_id, round((time.perf_counter() - start) * 1000))
        yield sse("error", {"error": str(exc), "type": exc.__class__.__name__, "stream_id": stream_id})


@app.post("/api/nlp/stream")
async def nlp_rewrite_stream(payload: NlpRewriteRequest) -> StreamingResponse:
    return StreamingResponse(nlp_stream_events(payload), media_type="text/event-stream")


async def rewrite_stream_events(payload: RewriteRequest) -> AsyncIterator[str]:
    stream_id = f"rewrite-{uuid4().hex[:12]}"
    start = time.perf_counter()
    chunk_count = 0
    try:
        settings = load_settings()
        provider_name = payload.provider or settings.provider
        model = payload.model or settings.model_for(provider_name)
        provider = get_provider(
            settings,
            provider_name=provider_name,
            model=model,
            base_url=payload.base_url,
            api_key=payload.api_key,
        )
        warnings = length_warnings(payload.text)
        logger.info(
            "Rewrite stream started stream_id=%s provider=%s model=%s base_url=%s text_chars=%s",
            stream_id,
            provider_name,
            model,
            provider.config.base_url,
            len(payload.text),
        )
        yield sse("stream_started", {"stream_id": stream_id, "provider": provider_name, "model": model})
        yield sse("node_started", {"node": "validate_length", "warnings": warnings, "stream_id": stream_id})

        yield sse("node_started", {"node": "select_prompt", "stream_id": stream_id})
        messages = build_messages(payload.platform, payload.text)

        yield sse("node_started", {"node": "llm_rewrite", "iteration": 1, "stream_id": stream_id})
        raw_chunks: list[str] = []
        async for delta in provider.stream(messages, request_id=stream_id):
            chunk_count += 1
            raw_chunks.append(delta)
            yield sse("llm_delta", {"delta": delta, "stream_id": stream_id})
        raw_output = "".join(raw_chunks)
        current = extract_rewritten_text(payload.platform, raw_output)
        yield sse("iteration_result", {"iteration": 1, "text": current})

        for iteration in range(2, payload.iterations + 1):
            yield sse("node_started", {"node": "evaluate_iterate", "iteration": iteration})
            detection_report = detect_aigc_risk(
                current,
                scene=detection_scene_for_platform(payload.platform),
            )
            detection_prompt = format_aigc_report_for_prompt(detection_report)
            yield sse(
                "aigc_detection_result",
                {
                    "iteration": iteration,
                    "score": detection_report.score,
                    "level": detection_report.level,
                    "issues": detection_report.issue_summaries,
                    "risky_sentences": detection_report.risky_sentences,
                },
            )
            feedback = await provider.complete(
                build_evaluator_messages(
                    payload.platform,
                    payload.text,
                    current,
                    detection_prompt,
                )
            )
            messages = build_messages(payload.platform, current, feedback, iteration=iteration)
            raw_output = await provider.complete(messages)
            current = extract_rewritten_text(payload.platform, raw_output)
            yield sse("iteration_result", {"iteration": iteration, "text": current})

        enabled = settings.nlp_enabled if payload.nlp_enabled is None else payload.nlp_enabled
        nlp_applied = False
        nlp_style: str | None = None
        if enabled:
            yield sse("node_started", {"node": "optional_nlp_classify"})
            nlp_applied = True
            mode = payload.nlp_mode or settings.nlp_mode
            nlp_style = await choose_nlp_style(
                current,
                mode,
                payload.nlp_style or settings.nlp_style,
                provider,
            )
            yield sse("node_started", {"node": "optional_nlp_rewrite", "style": nlp_style})
            current = await rewrite_with_nlp_style(
                current,
                nlp_style,
                aggressive=payload.nlp_aggressive,
                best_of_n=payload.nlp_best_of_n,
                seed=payload.nlp_seed,
            )
            yield sse("nlp_stream_started", {"style": nlp_style})
            async for event in stream_text("nlp_delta", current):
                yield event
            yield sse("nlp_result", {"style": nlp_style, "text": current})

        yield sse("node_started", {"node": "build_diff"})
        diff = build_diff(payload.text, current)
        yield sse("diff_ready", {"diff": diff})

        yield sse("node_started", {"node": "persist_record"})
        created_at = datetime.now(timezone.utc).isoformat()
        should_persist_text = not (payload.api_key and payload.api_key.strip())
        record_id = insert_history(
            {
                "original_text": payload.text if should_persist_text else "[自定义 API Key 请求：内容未在服务器保存]",
                "raw_output": raw_output if should_persist_text else "",
                "rewritten_text": current if should_persist_text else "[自定义 API Key 请求：内容未在服务器保存]",
                "platform": payload.platform,
                "provider": provider_name,
                "model": model,
                "iterations": payload.iterations,
                "warnings": warnings,
                "nlp_applied": nlp_applied,
                "nlp_style": nlp_style,
                "diff": diff if should_persist_text else [],
                "created_at": created_at,
            }
        )
        yield sse(
            "done",
            {
                "id": record_id,
                "text": current,
                "raw_output": raw_output,
                "warnings": warnings,
                "created_at": created_at,
                "stream_id": stream_id,
            },
        )
        logger.info(
            "Rewrite stream finished stream_id=%s chunks=%s elapsed_ms=%s",
            stream_id,
            chunk_count,
            round((time.perf_counter() - start) * 1000),
        )
    except Exception as exc:
        logger.exception(
            "Rewrite stream failed stream_id=%s chunks=%s elapsed_ms=%s",
            stream_id,
            chunk_count,
            round((time.perf_counter() - start) * 1000),
        )
        yield sse("error", {"error": str(exc), "type": exc.__class__.__name__, "stream_id": stream_id})


@app.post("/api/rewrite/stream")
async def rewrite_stream(payload: RewriteRequest) -> StreamingResponse:
    return StreamingResponse(rewrite_stream_events(payload), media_type="text/event-stream")


@app.get("/api/history", response_model=list[HistoryItem])
async def history() -> list[HistoryItem]:
    items: list[HistoryItem] = []
    for row in list_history():
        items.append(
            HistoryItem(
                id=row["id"],
                platform=row["platform"],
                provider=row["provider"],
                model=row["model"],
                original_preview=row["original_text"][:120],
                rewritten_preview=row["rewritten_text"][:120],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
        )
    return items


@app.get("/api/history/{record_id}", response_model=RewriteResponse)
async def history_detail(record_id: int) -> RewriteResponse:
    row = get_history(record_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return history_row_to_response(row)


@app.delete("/api/history/{record_id}")
async def history_delete(record_id: int) -> dict[str, bool]:
    return {"ok": delete_history(record_id)}
