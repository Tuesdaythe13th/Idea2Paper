import time
from typing import Optional, List

import requests

from idea2paper.config import LLM_API_KEY, OPENAI_API_KEY, EMBEDDING_PROVIDER
from idea2paper.infra.run_context import get_logger


# SiliconFlow embeddings (default)
EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

# OpenAI embeddings
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def _get_embedding_provider():
    """Determine which embedding provider to use."""
    if EMBEDDING_PROVIDER == "openai":
        return "openai"
    if EMBEDDING_PROVIDER == "siliconflow":
        return "siliconflow"
    # Auto mode: prefer OpenAI if available, fallback to SiliconFlow
    if OPENAI_API_KEY:
        return "openai"
    if LLM_API_KEY:
        return "siliconflow"
    return None


def _get_openai_embedding(text: str, logger=None, timeout: int = 120) -> Optional[List[float]]:
    """Get embedding using OpenAI API."""
    if logger is None:
        logger = get_logger()
    start_ts = time.time()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_EMBEDDING_MODEL,
        "input": text
    }

    try:
        resp = requests.post(OPENAI_EMBEDDING_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        emb = data["data"][0]["embedding"]
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "openai",
                    "url": OPENAI_EMBEDDING_URL,
                    "model": OPENAI_EMBEDDING_MODEL,
                    "input_preview": text[:200] if len(text) > 200 else text,
                    "timeout": timeout
                },
                response={
                    "ok": True,
                    "latency_ms": int((time.time() - start_ts) * 1000)
                }
            )
        return emb
    except Exception as e:
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "openai",
                    "url": OPENAI_EMBEDDING_URL,
                    "model": OPENAI_EMBEDDING_MODEL,
                    "input_preview": text[:200] if len(text) > 200 else text,
                    "timeout": timeout
                },
                response={
                    "ok": False,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                    "error": str(e)
                }
            )
        return None


def _get_openai_embeddings_batch(texts: List[str], logger=None, timeout: int = 120) -> Optional[List[List[float]]]:
    """Get embeddings for a batch of texts using OpenAI API."""
    if logger is None:
        logger = get_logger()
    start_ts = time.time()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_EMBEDDING_MODEL,
        "input": texts
    }

    try:
        resp = requests.post(OPENAI_EMBEDDING_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        embs = [item["embedding"] for item in data.get("data", [])]
        if len(embs) != len(texts):
            raise ValueError(f"embedding batch size mismatch: got {len(embs)} expected {len(texts)}")
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "openai",
                    "url": OPENAI_EMBEDDING_URL,
                    "model": OPENAI_EMBEDDING_MODEL,
                    "batch_size": len(texts),
                    "timeout": timeout
                },
                response={
                    "ok": True,
                    "latency_ms": int((time.time() - start_ts) * 1000)
                }
            )
        return embs
    except Exception as e:
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "openai",
                    "url": OPENAI_EMBEDDING_URL,
                    "model": OPENAI_EMBEDDING_MODEL,
                    "batch_size": len(texts),
                    "timeout": timeout
                },
                response={
                    "ok": False,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                    "error": str(e)
                }
            )
        return None


def get_embedding(text: str, logger=None, timeout: int = 120) -> Optional[List[float]]:
    """Get embedding for text using configured embeddings API.

    Returns None on failure (no exception thrown).
    """
    # Route to OpenAI if configured
    provider = _get_embedding_provider()
    if provider == "openai":
        return _get_openai_embedding(text, logger, timeout)

    if logger is None:
        logger = get_logger()
    start_ts = time.time()

    if not LLM_API_KEY:
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "siliconflow",
                    "url": EMBEDDING_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": text,
                    "timeout": timeout,
                    "simulated": True
                },
                response={
                    "ok": False,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                    "error": "SILICONFLOW_API_KEY not configured"
                }
            )
        return None

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    try:
        resp = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        emb = data["data"][0]["embedding"]
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "siliconflow",
                    "url": EMBEDDING_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": text,
                    "timeout": timeout,
                    "simulated": False
                },
                response={
                    "ok": True,
                    "latency_ms": int((time.time() - start_ts) * 1000)
                }
            )
        return emb
    except Exception as e:
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "siliconflow",
                    "url": EMBEDDING_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": text,
                    "timeout": timeout,
                    "simulated": False
                },
                response={
                    "ok": False,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                    "error": str(e)
                }
            )
        return None


def _preview_texts(texts: List[str], max_chars: int = 200) -> List[str]:
    previews = []
    for t in texts:
        if t is None:
            previews.append("")
            continue
        s = str(t)
        if len(s) > max_chars:
            previews.append(s[:max_chars] + "...(truncated)")
        else:
            previews.append(s)
    return previews


def get_embeddings_batch(texts: List[str], logger=None, timeout: int = 120) -> Optional[List[List[float]]]:
    """Get embeddings for a batch of texts. Returns None on failure."""
    # Route to OpenAI if configured
    provider = _get_embedding_provider()
    if provider == "openai":
        return _get_openai_embeddings_batch(texts, logger, timeout)

    if logger is None:
        logger = get_logger()
    start_ts = time.time()

    if not LLM_API_KEY:
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "siliconflow",
                    "url": EMBEDDING_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": _preview_texts(texts),
                    "timeout": timeout,
                    "simulated": True,
                    "batch_size": len(texts),
                },
                response={
                    "ok": False,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                    "error": "SILICONFLOW_API_KEY not configured"
                }
            )
        return None

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }

    try:
        resp = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        embs = [item["embedding"] for item in data.get("data", [])]
        if len(embs) != len(texts):
            raise ValueError(f"embedding batch size mismatch: got {len(embs)} expected {len(texts)}")
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "siliconflow",
                    "url": EMBEDDING_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": _preview_texts(texts),
                    "timeout": timeout,
                    "simulated": False,
                    "batch_size": len(texts),
                },
                response={
                    "ok": True,
                    "latency_ms": int((time.time() - start_ts) * 1000)
                }
            )
        return embs
    except Exception as e:
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "siliconflow",
                    "url": EMBEDDING_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": _preview_texts(texts),
                    "timeout": timeout,
                    "simulated": False,
                    "batch_size": len(texts),
                },
                response={
                    "ok": False,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                    "error": str(e)
                }
            )
        return None
