#!/usr/bin/env python3
"""Build Claret's data files from the USPTO Trademark ID Manual export.

Get the export at https://idm-tmng.uspto.gov/id-master-list-public.html -> Download
(Excel). The "Excel" file is an HTML table; save it as idmanual.xls.

  pip install sentence-transformers numpy
  python3 tools/build_data.py idmanual.xls data/

Writes:
  data/entries.json   [[term id, class, description, note], ...]  (active entries, classes 1-45)
  data/vectors.bin    one sign bit per dimension of each entry's all-MiniLM-L6-v2
                      embedding: 48 bytes per entry, same order, most significant bit first
  data/version.txt    ID Manual version and date
"""
import html, json, re, sys, datetime
import numpy as np
from sentence_transformers import SentenceTransformer

src, out = sys.argv[1], sys.argv[2].rstrip("/")
t = open(src, encoding="utf-8").read()
ents, seen, ver = [], set(), ""
for tr in re.findall(r"<tr>(.*?)</tr>", t, re.S):
    c = [html.unescape(re.sub("<[^>]+>", "", x)).strip() for x in re.findall(r"<td>(.*?)</td>", tr, re.S)]
    if len(c) < 7:
        continue
    tid, cls, desc, status, note = c[0], c[1], c[2], c[3], c[6]
    if status not in ("A", "M") or not cls.isdigit() or int(cls) == 0 or not desc or tid in seen:
        continue  # deleted entries, certification classes A/B, and class 000 (not selectable when filing)
    seen.add(tid)
    ents.append([tid, int(cls), desc, note[:400]])
    ver = max(ver, c[8].strip('"')) if len(c) > 8 else ver
ents.sort(key=lambda e: (e[1], int(e[0].split("-")[1]) if e[0].split("-")[1].isdigit() else 0))


def clean(d):
    d = re.sub(r"\[[^\]]*\]", " ", d)  # [NOTE: ...] and [classification ...] brackets
    d = re.sub(r"[{}]", " ", d)        # keep the words inside {indicate ...}
    return re.sub(r"\s+", " ", d).strip()


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
vec = model.encode([clean(e[2]) for e in ents], batch_size=256, normalize_embeddings=True,
                   show_progress_bar=True, convert_to_numpy=True)
np.packbits(vec > 0, axis=1).tofile(f"{out}/vectors.bin")
json.dump(ents, open(f"{out}/entries.json", "w", encoding="utf-8"), separators=(",", ":"), ensure_ascii=False)
open(f"{out}/version.txt", "w").write(f"{len(ents)} entries, built {datetime.date.today().isoformat()}\n")
print(len(ents), "entries")
