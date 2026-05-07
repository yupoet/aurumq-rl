"""Upload Phase 24/25 handoff bundle to ledashi-oss/fromsz/."""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET_NAME = "ledashi-oss"
PREFIX = "fromsz/handoffs/2026-05-07-phase24-25-tech-factors/"


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
    handoff = ROOT / "handoffs" / "2026-05-07-phase24-25-tech-factors"
    for p in sorted(handoff.glob("*.md")):
        manifest.append((p, p.name))

    # Factor spec for upstream — KEY DELIVERABLE
    spec = ROOT / "docs/factor_spec_for_upstream/TECH_FACTOR_SPEC.md"
    if spec.exists():
        manifest.append((spec, f"factor_spec_for_upstream/{spec.name}"))

    # Phase 24A artifacts (failed run, kept for forensics)
    run_name = "phase24a_tech_seed42"
    run_dir = ROOT / "runs" / run_name
    final_zip = run_dir / "ppo_final.zip"
    if final_zip.exists():
        manifest.append((final_zip, f"models/{run_name}_final.zip"))
    for fname in (
        "episode_eval.md", "episode_eval.json", "episode_picks.jsonl",
        "training_summary.json", "metadata.json", "train.log",
        "t1_diagnostic.md",
    ):
        p = run_dir / fname
        if p.exists():
            manifest.append((p, f"run_sweeps/{run_name}/{fname}"))

    # Phase 25A partial (60% killed; 7 ckpts evaluated)
    run_name_25a = "phase25a_weighted_seed42"
    run_dir_25a = ROOT / "runs" / run_name_25a
    for fname in (
        "episode_eval.md", "episode_eval.json", "episode_picks.jsonl",
        "metadata.json", "train.log",
    ):
        p = run_dir_25a / fname
        if p.exists():
            manifest.append((p, f"run_sweeps/{run_name_25a}/{fname}"))

    # Importance + weights pipeline outputs
    aux_files = [
        "runs/phase23a_episode_seed42/factor_importance_ig_only.json",
        "runs/_episode_inspect/factor_t_minus_k_ALL355.md",
        "runs/_episode_inspect/factor_t_minus_k_ALL355.json",
        "runs/_episode_inspect/factor_t_minus_k_alpha_gtja.md",
        "runs/_episode_inspect/tech_factor_t_minus_k.json",
        "runs/phase25_factor_weights.json",
        "runs/phase25_factor_weights_base_only.json",
    ]
    for relpath in aux_files:
        p = ROOT / relpath
        if p.exists():
            manifest.append((p, f"importance_pipeline/{p.name}"))

    # Pipeline log
    overnight_log = ROOT / "runs" / "_phase25_overnight.log"
    if overnight_log.exists():
        manifest.append((overnight_log, "reports/phase25_overnight.log"))

    # Source code (the factor module that should be REMOVED post-upstream-fix)
    code_files = [
        "src/aurumq_rl/technical_factors.py",
        "src/aurumq_rl/main_wave_episodes.py",
        "src/aurumq_rl/main_wave_target_labels.py",
        "scripts/compute_factor_weights.py",
        "scripts/eval_factor_importance.py",
        "scripts/_inspect_factor_at_t_minus_k.py",
        "scripts/_inspect_tech_factor_t_minus_k.py",
        "scripts/_diagnose_t1_hits.py",
    ]
    for relpath in code_files:
        p = ROOT / relpath
        if p.exists():
            manifest.append((p, f"source/{p.name}"))

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
