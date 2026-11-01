from __future__ import annotations

from typing import Any, Dict

import requests


UA_RESEARCH = "Mozilla/5.0 (compatible; AcademicResearchBot/1.0)"
UA_BROWSER = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
MAX_CONTENT_BYTES = 1_000_000


def _decode_text(response: requests.Response) -> str:
    encoding = response.encoding or "utf-8"
    try:
        return response.content[:MAX_CONTENT_BYTES].decode(encoding, errors="replace")
    except LookupError:
        return response.content[:MAX_CONTENT_BYTES].decode("utf-8", errors="replace")


def _single_request(
    url: str, timeout: int, as_bytes: bool, user_agent: str
) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": user_agent},
            timeout=timeout,
            allow_redirects=True,
        )
    except requests.exceptions.Timeout:
        return {
            "status_code": None,
            "content": None,
            "final_url": url,
            "fetch_error": "timeout",
            "ua_used": user_agent,
        }
    except requests.exceptions.SSLError:
        return {
            "status_code": None,
            "content": None,
            "final_url": url,
            "fetch_error": "ssl_error",
            "ua_used": user_agent,
        }
    except requests.exceptions.TooManyRedirects:
        return {
            "status_code": None,
            "content": None,
            "final_url": url,
            "fetch_error": "too_many_redirects",
            "ua_used": user_agent,
        }
    except requests.exceptions.ConnectionError:
        return {
            "status_code": None,
            "content": None,
            "final_url": url,
            "fetch_error": "connection_error",
            "ua_used": user_agent,
        }
    except requests.RequestException as exc:
        return {
            "status_code": None,
            "content": None,
            "final_url": url,
            "fetch_error": f"request_exception:{type(exc).__name__}",
            "ua_used": user_agent,
        }

    if as_bytes:
        content: Any = response.content[:MAX_CONTENT_BYTES]
    else:
        content = _decode_text(response)

    return {
        "status_code": response.status_code,
        "content": content,
        "final_url": response.url,
        "fetch_error": "",
        "ua_used": user_agent,
    }


def fetch_url(
    url: str, timeout: int = 30, as_bytes: bool = False, ua: str = UA_RESEARCH
) -> Dict[str, Any]:
    first = _single_request(url, timeout=timeout, as_bytes=as_bytes, user_agent=ua)
    if first["status_code"] == 403 and ua != UA_BROWSER:
        retry = _single_request(url, timeout=timeout, as_bytes=as_bytes, user_agent=UA_BROWSER)
        if retry["status_code"] is not None or retry["fetch_error"]:
            return retry
    return first


def fetch_robots(host: str, timeout: int = 15) -> Dict[str, Any]:
    https_url = f"https://{host}/robots.txt"
    result = fetch_url(https_url, timeout=timeout, as_bytes=False, ua=UA_RESEARCH)
    result["http_fallback"] = False

    if result.get("fetch_error") == "ssl_error":
        http_url = f"http://{host}/robots.txt"
        result = fetch_url(http_url, timeout=timeout, as_bytes=False, ua=UA_RESEARCH)
        result["http_fallback"] = True
        return result

    return result
