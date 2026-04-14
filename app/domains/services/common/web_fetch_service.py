import html
import json
import re
from dataclasses import dataclass
from typing import Any,Dict,Tuple
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup


USER_AGENT = "Mozilla/5.0 (compatible; KBService/1.0)"
MAX_REDIRECTS = 5


def _strip_tags(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _validate_url(url: str) -> Tuple[bool, str]:
    try:
        p = urlparse((url or "").strip())
        if p.scheme not in ("http", "https"):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


@dataclass
class WebFetchResult:
    url: str
    final_url: str
    status: int
    extractor: str
    title: str
    text: str
    truncated: bool
    length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "finalUrl": self.final_url,
            "status": self.status,
            "extractor": self.extractor,
            "title": self.title,
            "truncated": self.truncated,
            "length": self.length,
            "text": self.text,
        }


class WebFetchService:
    @staticmethod
    async def fetch(url: str,extract_mode: str = "text",max_chars: int = 50000) -> WebFetchResult:
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            raise ValueError(f"URL validation failed: {error_msg}")

        async with httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
            timeout=30.0
        ) as client:
            r = await client.get(url, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()

        ctype = (r.headers.get("content-type", "") or "").lower()
        text = ""
        title = ""
        extractor = "raw"
        body = r.text or ""

        if "application/json" in ctype:
            text = json.dumps(r.json(), indent=2, ensure_ascii=False)
            extractor = "json"
        elif "text/html" in ctype or body[:256].lower().startswith(("<!doctype", "<html")):
            title, text, extractor = WebFetchService._extract_from_html(body, extract_mode)
        else:
            text = body

        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]

        return WebFetchResult(
            url=url,
            final_url=str(r.url),
            status=r.status_code,
            extractor=extractor,
            title=title,
            text=text,
            truncated=truncated,
            length=len(text),
        )

    @staticmethod
    def _extract_from_html(html_content: str,extract_mode: str) -> Tuple[str, str, str]:
        try:
            from readability import Document as ReadabilityDocument
            doc = ReadabilityDocument(html_content)
            title = (doc.title() or "").strip()
            summary = doc.summary() or ""
            extractor = "readability"
        except Exception:
            soup = BeautifulSoup(html_content, "html.parser")
            title = (soup.title.string or "").strip() if soup.title else ""
            summary = str(soup.body or soup)
            extractor = "bs4"

        if extract_mode == "markdown":
            content = WebFetchService._to_markdown(summary)
        else:
            content = _normalize(_strip_tags(summary))
        if title:
            return title, f"# {title}\n\n{content}", extractor
        return title, content, extractor

    @staticmethod
    def _to_markdown(raw_html: str) -> str:
        text = re.sub(
            r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>([\s\S]*?)</a>",
            lambda m: f"[{_strip_tags(m[2])}]({m[1]})",
            raw_html,
            flags=re.I,
        )
        text = re.sub(
            r"<h([1-6])[^>]*>([\s\S]*?)</h\1>",
            lambda m: f"\n{'#' * int(m[1])} {_strip_tags(m[2])}\n",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"<li[^>]*>([\s\S]*?)</li>",
            lambda m: f"\n- {_strip_tags(m[1])}",
            text,
            flags=re.I,
        )
        text = re.sub(r"</(p|div|section|article)>", "\n\n", text, flags=re.I)
        text = re.sub(r"<(br|hr)\s*/?>", "\n", text, flags=re.I)
        return _normalize(_strip_tags(text))
