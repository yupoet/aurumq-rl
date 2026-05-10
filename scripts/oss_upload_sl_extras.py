"""Upload SL extras: SHAP audit + Path 5 regime stacking + paris reply.

Companion handoff to:
  oss://ledashi-oss/fromsz/handoffs/2026-05-10-sl-path4-inference-bundle/  (production)
  oss://ledashi-oss/fromsz/handoffs/2026-05-10-sl-paths-4-2-final-results/ (full results)
"""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET_NAME = "ledashi-oss"
PREFIX = "fromsz/handoffs/2026-05-10-sl-extras/"


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

    # SHAP audit + reply
    for f in (
        ROOT / "runs" / "sl_path4" / "feature_importance_audit.md",
        ROOT / "runs" / "sl_path4" / "feature_importance.json",
        ROOT / "runs" / "sl_path4" / "feature_audit_top30.json",
        ROOT / "runs" / "sl_path4" / "feature_audit_drop_candidates.json",
        ROOT / "runs" / "sl_path4" / "REPLY_TO_PARIS.md",
    ):
        if f.exists():
            manifest.append((f, f"shap_audit/{f.name}"))

    # Path 5 regime stacking
    p5 = ROOT / "runs" / "sl_regime_stack"
    for fname in ("RESULTS.md", "ensemble.json", "predictions.parquet",
                  "meta_lgb_model.txt", "meta_isotonic.pkl"):
        f = p5 / fname
        if f.exists():
            manifest.append((f, f"regime_stacking/{fname}"))

    # Path 7 conformal + sizing
    p7 = ROOT / "runs" / "sl_conformal"
    for fname in ("RESULTS.md", "ensemble.json"):
        f = p7 / fname
        if f.exists():
            manifest.append((f, f"conformal_sizing/{fname}"))

    # Path 6 Bayesian opt (when complete)
    p6 = ROOT / "runs" / "sl_path6"
    if (p6 / "trials.json").exists():
        manifest.append((p6 / "trials.json", "bayesian_opt/trials.json"))
    if (p6 / "ensemble.json").exists():
        manifest.append((p6 / "ensemble.json", "bayesian_opt/ensemble.json"))
    if (p6 / "RESULTS.md").exists():
        manifest.append((p6 / "RESULTS.md", "bayesian_opt/RESULTS.md"))
    for run_dir in sorted(p6.glob("*_seed*/")):
        for fname in ("results.json", "lgb_model.txt"):
            f = run_dir / fname
            if f.exists():
                manifest.append((f, f"bayesian_opt/runs/{run_dir.name}/{fname}"))

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
