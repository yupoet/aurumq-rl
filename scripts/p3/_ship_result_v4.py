"""Ship RESULT v4 bundle to fromsz/handoffs/2026-05-15-ledashi-result-v4/."""
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

bucket = oss2.Bucket(oss2.Auth(ak, sk), 'oss-cn-shenzhen.aliyuncs.com', 'ledashi-oss', connect_timeout=30)
DST_PREFIX = "fromsz/handoffs/2026-05-15-ledashi-result-v4/"
SRC = Path("D:/dev/aurumq-handoffs/outbox/2026-05-15-ledashi-result-v4")

print(f"[ship] uploading to {DST_PREFIX}")
for f in sorted(SRC.iterdir()):
    if not f.is_file(): continue
    sz = f.stat().st_size
    t = time.time()
    bucket.put_object_from_file(f"{DST_PREFIX}{f.name}", str(f))
    print(f"  {f.name}  ({sz/1e3:.1f} KB, {time.time()-t:.1f}s)")
print("[done] RESULT v4 shipped (Shenzhen, CRR will mirror SGP)")
