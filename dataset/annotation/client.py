import asyncio
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import tomli


@dataclass
class Provider:
    name: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    concurrency: int = 1

    def public(self) -> dict:
        return {"provider": self.name, "model": self.model,
                "concurrency_limit": self.concurrency}


def load_provider(name: str, database: Path | None = None) -> Provider:
    database = database or Path.home() / ".cc-switch/cc-switch.db"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE app_type='codex' AND name=?", (name,)
        ).fetchone()
    if row is None:
        raise ValueError(f"CC Switch 中找不到 Codex provider: {name}")
    settings = json.loads(row[0])
    config = tomli.loads(settings["config"])
    model = config['model']
    url = config['model_providers'][config['model_provider']]['base_url']
    secret = settings["auth"]["OPENAI_API_KEY"]
    limits = {"buoke": 6, "ldc": 40}
    if name not in limits:
        raise ValueError("本次仅使用用户指定的 buoke 或 ldc")
    return Provider(name, model, url.rstrip("/"), secret, limits[name])


def response_text(data: dict) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str):
        return direct
    texts = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for chunk in item.get("content", []):
            if chunk.get("type") in ("output_text", "text"):
                texts.append(chunk.get("text", ""))
    if not texts and data.get("choices"):
        texts.append(data["choices"][0]["message"]["content"])
    return "\n".join(texts)


def parse_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("响应应为一个JSON对象")
    return result


class TeacherClient:
    def __init__(self, provider: Provider, concurrency: int = 1):
        if not 1 <= concurrency <= provider.concurrency:
            raise ValueError(f"{provider.name} 并行上限为 {provider.concurrency}")
        self.provider = provider
        self.semaphore = asyncio.Semaphore(concurrency)
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20),
                                     limits=httpx.Limits(max_connections=concurrency,
                                                        max_keepalive_connections=concurrency))

    async def close(self):
        await self.http.aclose()

    async def request(self, prompt: str, images: list[str] | None = None,
                      max_output_tokens: int = 1800) -> dict[str, Any]:
        content = [{"type": "input_text", "text": prompt}]
        for image in images or []:
            content.append({"type": "input_image", "image_url": image})
        payload = {
            "model": self.provider.model, "store": False,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
        }
        attempts = []
        began = time.perf_counter()
        async with self.semaphore:
            for attempt in range(3):
                try:
                    response = await self.http.post(
                        self.provider.base_url + "/responses",
                        headers={"Authorization": "Bearer " + self.provider.api_key}, json=payload,
                    )
                    attempts.append({"attempt": attempt + 1, "status": response.status_code})
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < 2:
                            await asyncio.sleep(2 ** (attempt + 1))
                            continue
                    response.raise_for_status()
                    raw = response.json()
                    text = response_text(raw)
                    if not text:
                        raise ValueError(f"接口没有输出正文，status={raw.get('status')}")
                    return {**self.provider.public(), "text": text,
                            "response_model": raw.get("model"), "response_id": raw.get("id"),
                            "status": raw.get("status"), "usage": raw.get("usage"),
                            "elapsed_seconds": time.perf_counter() - began, "attempts": attempts}
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    attempts.append({"attempt": attempt + 1, "error_type": type(exc).__name__})
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** (attempt + 1))
        raise RuntimeError("请求重试结束，未取得有效响应")
