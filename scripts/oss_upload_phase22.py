"""Upload Phase 22 handoff bundle to ledashi-oss/fromsz/."""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET_NAME = "ledashi-oss"
PREFIX = "fromsz/handoffs/2026-05-06-phase22-main-wave/"


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
    handoff = ROOT / "handoffs" / "2026-05-06-phase22-main-wave"
    for p in sorted(handoff.glob("*.md")):
        manifest.append((p, p.name))

    # Phase 22 run bundles (A, B, C — best ckpts + reports + picks + inspect)
    runs_specs: list[tuple[str, int | None]] = [
        ("phase22a_main_wave_v1_seed42", 299904),     # 22A best step
        ("phase22b_main_wave_v1_seed1", None),        # 22B did not converge
        ("phase22c_topk3_seed42", 174944),            # 22C top-5 eval best step
        ("phase22c_topk3_seed42", 199936),            # 22C top-3 eval best step
    ]
    seen_dirs: set[str] = set()
    seen_ckpts: set[tuple[str, int]] = set()
    for run_name, ckpt_step in runs_specs:
        run_dir = ROOT / "runs" / run_name
        if ckpt_step is not None:
            key = (run_name, ckpt_step)
            ckpt = run_dir / "checkpoints" / f"ppo_{ckpt_step}_steps.zip"
            if key not in seen_ckpts and ckpt.exists():
                manifest.append((ckpt, f"models/{run_name}_best_step{ckpt_step}.zip"))
                seen_ckpts.add(key)
        if run_name in seen_dirs:
            continue
        seen_dirs.add(run_name)
        final_zip = run_dir / "ppo_final.zip"
        if final_zip.exists():
            manifest.append((final_zip, f"models/{run_name}_final.zip"))
        for fname in (
            "main_wave_eval.md", "main_wave_eval.json", "main_wave_picks.jsonl",
            "training_summary.json", "metadata.json", "train.log", "inspect_top3.md",
        ):
            p = run_dir / fname
            if p.exists():
                manifest.append((p, f"run_sweeps/{run_name}/{fname}"))

    # Phase 16a baseline eval re-run (key comparison)
    p16 = ROOT / "runs" / "phase16a_fixed_drop_mkt_300k"
    for fname in ("main_wave_eval.md", "main_wave_eval.json", "main_wave_picks.jsonl"):
        p = p16 / fname
        if p.exists():
            manifest.append((p, f"baseline/phase16a_{fname}"))

    # Pipeline log
    log = ROOT / "runs" / "_phase22_overnight.out"
    if log.exists():
        manifest.append((log, "reports/overnight_pipeline.log"))

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
