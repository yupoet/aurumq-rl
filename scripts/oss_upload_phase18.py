"""Upload Phase 18 handoff bundle to ledashi-oss/fromsz/."""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET_NAME = "ledashi-oss"
PREFIX = "fromsz/handoffs/2026-05-03-phase18-6h/"


def _read_env() -> dict[str, str]:
    raw = (ROOT / ".env").read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="ignore")
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii")


def _upload_one(bucket: oss2.Bucket, local: Path, remote_key: str) -> None:
    size = local.stat().st_size
    label = f"{size/1e6:7.2f} MiB" if size > 1e6 else f"{size/1e3:7.2f} KiB"
    try:
        meta = bucket.get_object_meta(remote_key)
        local_md5_hex = hashlib.md5(local.read_bytes()).hexdigest().lower()
        if meta.etag.strip('"').lower() == local_md5_hex:
            print(f"  [skip] {remote_key} ({label})")
            return
    except oss2.exceptions.NoSuchKey:
        pass
    print(f"  [put]  {remote_key} ({label})")
    headers = {"Content-MD5": _md5_b64(local)}
    bucket.put_object_from_file(remote_key, str(local), headers=headers)


def main() -> int:
    env = _read_env()
    bucket = oss2.Bucket(
        oss2.Auth(env["OSS_ACCESS_KEY_ID"], env["OSS_ACCESS_KEY_SECRET"]),
        ENDPOINT, BUCKET_NAME, connect_timeout=30,
    )
    print(f"[oss-upload] endpoint={ENDPOINT} bucket={BUCKET_NAME} prefix={PREFIX}")

    manifest: list[tuple[Path, str]] = []
    handoff = ROOT / "handoffs" / "2026-05-03-phase18-6h"
    for fname in ("HANDOFF_2026-05-03_phase18.md", "PHASE18_6H_UNATTENDED_INSTRUCTIONS.md"):
        p = handoff / fname
        if p.exists():
            manifest.append((p, fname))

    p18 = ROOT / "runs" / "phase18_6h"
    for fname in ("decision_log.md", "orchestrator.py", "orchestrator_stdout.log"):
        p = p18 / fname
        if p.exists():
            manifest.append((p, f"reports/{fname}"))

    rep = ROOT / "reports" / "phase18_6h"
    for p in sorted(rep.glob("*.md")):
        manifest.append((p, f"reports/{p.name}"))
    for p in sorted(rep.glob("*.json")):
        manifest.append((p, f"reports/{p.name}"))

    for fname in (
        "phase18_18a_drop_mkt_seed4.train.log",
        "phase18_18a_drop_mkt_seed4.eval.log",
        "phase18_18b_drop_mkt_seed5.train.log",
        "phase18_18b_drop_mkt_seed5.eval.log",
        "phase18_18c_drop_mkt_seed6.train.log",
        "phase18_18c_drop_mkt_seed6.eval.log",
        "phase18_18d_drop_mkt_seed7.train.log",
        "phase18_18d_drop_mkt_seed7.eval.log",
    ):
        p = p18 / fname
        if p.exists():
            manifest.append((p, f"logs/{fname}"))

    p18_models = ROOT / "models" / "phase18"
    for p in sorted(p18_models.glob("*")):
        manifest.append((p, f"models/{p.name}"))

    runs = (
        "phase18_18a_drop_mkt_seed4",
        "phase18_18b_drop_mkt_seed5",
        "phase18_18c_drop_mkt_seed6",
        "phase18_18d_drop_mkt_seed7",
    )
    for run_name in runs:
        run_dir = ROOT / "runs" / run_name
        for fname in ("oos_sweep.md", "oos_sweep.json", "training_summary.json", "metadata.json"):
            p = run_dir / fname
            if p.exists():
                manifest.append((p, f"run_sweeps/{run_name}/{fname}"))

    print(f"[oss-upload] manifest: {len(manifest)} files")
    for local, sub in manifest:
        if not local.exists():
            print(f"  [missing] {local}")
            continue
        _upload_one(bucket, local, PREFIX + sub)

    print("[oss-upload] DONE.")
    print(f"  https://oss.console.aliyun.com/bucket/oss-cn-shenzhen/{BUCKET_NAME}/object?path={PREFIX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
