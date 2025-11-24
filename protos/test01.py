from typing import Any, reveal_type

from daglite import task
from daglite.engine import evaluate


@task
def download(url: str) -> str:
    """Download content from a URL."""
    print(f"[download] fetching {url!r}")
    return f"raw({url})"


@task
def parse(raw: str) -> dict[str, Any]:
    """Parse raw content into structured information."""
    print(f"[parse] parsing {raw!r}")
    return {"content": raw, "length": len(raw)}


@task()
def length(info: dict) -> int:
    """Compute the length of the content."""
    print("[length] computing length")
    return info["length"]


raw = download.bind(url="https://example.com")
info = parse.bind(raw=raw)
length_val = length.bind(info=info)

reveal_type(download)
reveal_type(raw)
reveal_type(info)
reveal_type(length_val)

result = evaluate(length_val)
print("RESULT:", result)
