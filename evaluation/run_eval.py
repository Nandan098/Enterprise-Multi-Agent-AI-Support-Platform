from __future__ import annotations

import json
from pathlib import Path

import requests

BASE="http://localhost:8000"
CASES=json.loads((Path(__file__).parent/"golden_questions.json").read_text())

def main():
    results=[]
    for case in CASES:
        r=requests.post(f"{BASE}/evaluate",json={"question":case["question"],"expected_keywords":case["expected_keywords"]},timeout=180)
        r.raise_for_status(); d=r.json()
        results.append({"route_ok":d["route"]==case["expected_route"],"validation":d["validation"],"keyword_score":d["keyword_score"],"latency_ms":d["latency_ms"]})
    route_rate=sum(x["route_ok"] for x in results)/len(results)
    val_rate=sum(x["validation"]=="PASS" for x in results)/len(results)
    kw=[x["keyword_score"] for x in results if x["keyword_score"] is not None]
    avg_kw=sum(kw)/len(kw) if kw else 0
    avg_latency=sum(x["latency_ms"] for x in results)/len(results)
    print("\n=== ACE PROJECT 1 EVALUATION ===")
    print(f"Cases:               {len(results)}")
    print(f"Route accuracy:      {route_rate:.1%}")
    print(f"Validation pass:     {val_rate:.1%}")
    print(f"Avg keyword score:   {avg_kw:.1%}")
    print(f"Avg latency:         {avg_latency:.0f} ms")
    return 0

if __name__=="__main__": raise SystemExit(main())
