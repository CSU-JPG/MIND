"""Write a MIND result JSON filtered to one perspective. Usage:
    _split_persp.py <result.json> <1st_data|3rd_data> <out.json>
Used by scores.bat to print per-perspective tables via _scores_table.py.
"""
import json
import sys

src, persp, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(src, encoding="utf-8"))
d["data"] = [e for e in d.get("data", []) if e.get("perspective") == persp]
json.dump(d, open(out, "w"))
print(f"{persp}: {len(d['data'])} samples")
