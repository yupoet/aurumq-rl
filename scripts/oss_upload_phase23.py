"""Upload Phase 23 handoff bundle to ledashi-oss/fromsz/."""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET_NAME = "ledashi-oss"
PREFIX = "fromsz/handoffs/2026-05-06-phase23-episode-targets/"


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
    bucket.put_object_from_file(remote_key, str(local), headers={"Content-MD5": _md5_b64(local)})


def main() -> int:
    env = _read_env()
    bucket = oss2.Bucket(
        oss2.Auth(env["OSS_ACCESS_KEY_ID"], env["OSS_ACCESS_KEY_SECRET"]),
        ENDPOINT, BUCKET_NAME, connect_timeout=30,
    )
    print(f"[oss-upload] endpoint={ENDPOINT} bucket={BUCKET_NAME} prefix={PREFIX}")

    manifest: list[tuple[Path, str]] = []
    handoff = ROOT / "handoffs" / "2026-05-06-phase23-episode-targets"
    for p in sorted(handoff.glob("*.md")):
        manifest.append((p, p.name))

    # Phase 23A primary candidate
    run_name = "phase23a_episode_seed42"
    run_dir = ROOT / "runs" / run_name
    for step in (224928, ):   # best by T-1 hit
        ckpt = run_dir / "checkpoints" / f"ppo_{step}_steps.zip"
        if ckpt.exists():
            manifest.append((ckpt, f"models/{run_name}_best_step{step}.zip"))
    final_zip = run_dir / "ppo_final.zip"
    if final_zip.exists():
        manifest.append((final_zip, f"models/{run_name}_final.zip"))
    for fname in (
        "episode_eval.md", "episode_eval.json", "episode_picks.jsonl",
        "training_summary.json", "metadata.json", "train.log",
    ):
        p = run_dir / fname
        if p.exists():
            manifest.append((p, f"run_sweeps/{run_name}/{fname}"))

    # Episode inspection reports (all-panel + factor T-k breakdown)
    insp_dir = ROOT / "runs" / "_episode_inspect"
    if insp_dir.exists():
        for p in sorted(insp_dir.glob("*")):
            manifest.append((p, f"inspect/{p.name}"))

    # Cross-phase episode_eval comparison (Phase 16a / 22A / 22B / 22C / 23A)
    cross_runs = [
        "phase16a_fixed_drop_mkt_300k",
        "phase22a_main_wave_v1_seed42",
        "phase22b_main_wave_v1_seed1",
        "phase22c_topk3_seed42",
    ]
    for cr in cross_runs:
        rd = ROOT / "runs" / cr
        for fname in ("episode_eval.md", "episode_eval.json", "episode_picks.jsonl"):
            p = rd / fname
            if p.exists():
                manifest.append((p, f"comparison/{cr}_{fname}"))

    # Spec + plan
    for relpath in (
        "docs/superpowers/specs/2026-05-06-phase23-episode-targets-design.md",
    ):
        p = ROOT / relpath
        if p.exists():
            manifest.append((p, f"docs/{p.name}"))

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
