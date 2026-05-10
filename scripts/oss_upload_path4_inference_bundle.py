"""Upload Path 4 inference bundle + INFER.md + QUESTIONS doc to OSS for paris handoff."""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET_NAME = "ledashi-oss"
PREFIX = "fromsz/handoffs/2026-05-10-sl-path4-inference-bundle/"


def _read_env() -> dict[str, str]:
    out: dict[str, str] = {}
    raw = (ROOT / ".env").read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _md5_pair(p: Path) -> tuple[str, str]:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii"), h.hexdigest().lower()


def _fmt(s: int) -> str:
    if s > 1 << 30:
        return f"{s/(1<<30):6.2f} GiB"
    if s > 1 << 20:
        return f"{s/(1<<20):6.2f} MiB"
    return f"{s/1024:6.2f} KiB"


def main() -> int:
    env = _read_env()
    auth = oss2.Auth(env["OSS_ACCESS_KEY_ID"], env["OSS_ACCESS_KEY_SECRET"])
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME, connect_timeout=30)
    print(f"[oss-upload] endpoint={ENDPOINT} bucket={BUCKET_NAME} prefix={PREFIX}")

    manifest: list[tuple[Path, str]] = []
    sl_path4 = ROOT / "runs" / "sl_path4"
    bundle = sl_path4 / "inference_bundle"

    # 1. Top-level docs
    for top in ("INFER.md", "QUESTIONS_FOR_PARIS.md"):
        p = sl_path4 / top
        if p.exists():
            manifest.append((p, top))

    # 2. Bundle contents
    for fname in ("manifest.json", "feature_cols.json", "isotonic.pkl", "rank_z.py"):
        p = bundle / fname
        if p.exists():
            manifest.append((p, f"inference_bundle/{fname}"))

    for m in sorted((bundle / "models").glob("lgb_model_*.txt")):
        manifest.append((m, f"inference_bundle/models/{m.name}"))

    # 3. Also include the inference script itself for reference
    p_inf = ROOT / "scripts" / "p3" / "path4_inference.py"
    if p_inf.exists():
        manifest.append((p_inf, "inference_bundle/path4_inference.py"))

    total = sum(p.stat().st_size for p, _ in manifest)
    print(f"[oss-upload] manifest: {len(manifest)} files, {_fmt(total)}")

    n_put = n_skip = 0
    for local, sub in manifest:
        key = PREFIX + sub
        size = local.stat().st_size
        b64, hex_ = _md5_pair(local)
        try:
            meta = bucket.get_object_meta(key)
            if meta.etag.strip('"').lower() == hex_:
                print(f"  [skip] {key} ({_fmt(size)})")
                n_skip += 1
                continue
        except oss2.exceptions.NoSuchKey:
            pass
        print(f"  [put]  {key} ({_fmt(size)})")
        bucket.put_object_from_file(key, str(local), headers={"Content-MD5": b64})
        n_put += 1

    print(f"\n[oss-upload] DONE. uploaded={n_put} skipped={n_skip}")
    print(f"[oss-upload] browse: https://oss.console.aliyun.com/bucket/oss-cn-shenzhen/{BUCKET_NAME}/object?path={PREFIX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
