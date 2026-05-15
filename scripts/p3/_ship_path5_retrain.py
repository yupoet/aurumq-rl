"""Ship path5 + path5_long retrain bundles to fromsz/handoffs/2026-05-15-path5-meta-retrain/."""
import os
from pathlib import Path
import time

for v in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy'):
    os.environ.pop(v, None)
os.environ['NO_PROXY'] = '*'

import oss2

ak = sk = None
for line in Path('D:/dev/aurumq-handoffs/.secrets/credentials.env').read_text(encoding='utf-8').splitlines():
    if '=' in line and not line.strip().startswith('#'):
        k, _, v = line.partition('='); k, v = k.strip(), v.strip()
        if k == 'OSS_ACCESS_KEY_ID': ak = v
        elif k == 'OSS_ACCESS_KEY_SECRET': sk = v

# Upload to ledashi-oss (Shenzhen); CRR auto-mirrors to ledashi-oss-sgp.
bucket_sz = oss2.Bucket(oss2.Auth(ak, sk), 'oss-cn-shenzhen.aliyuncs.com', 'ledashi-oss', connect_timeout=30)

DST_PREFIX = "fromsz/handoffs/2026-05-15-path5-meta-retrain/"

LOCAL_BUNDLES = [
    ("path5", Path("D:/dev/aurumq-rl/runs/sl_path5_retrain_v2_bundle")),
    ("path5_long", Path("D:/dev/aurumq-rl/runs/sl_path5_long_retrain_v2_bundle")),
]
README = Path("D:/dev/aurumq-handoffs/outbox/2026-05-15-ledashi-path5-meta-retrain/README.md")

def upload(bucket, prefix, local_path, dst_key):
    sz = local_path.stat().st_size
    t = time.time()
    bucket.put_object_from_file(dst_key, str(local_path))
    print(f"  [{bucket.bucket_name}] {dst_key}  ({sz/1e3:.1f} KB, {time.time()-t:.1f}s)")

print(f"[ship] uploading to {DST_PREFIX}")
for name, local_dir in LOCAL_BUNDLES:
    for f in sorted(local_dir.iterdir()):
        if f.is_file():
            upload(bucket_sz, DST_PREFIX, f, f"{DST_PREFIX}{name}/{f.name}")

# README at root of bundle
upload(bucket_sz, DST_PREFIX, README, f"{DST_PREFIX}README.md")

print("[done] retrain bundles uploaded to ledashi-oss/fromsz/; CRR will mirror to sgp")
