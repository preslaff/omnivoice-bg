# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Post-pass over the parliament crawl: find sessions with an implausible
text/audio ratio and repair the fixable class.

Two failure classes (both observed in the 2026-07 crawl):
  LOW  chars/sec (< --min-cps): the stenographic transcript was not yet
       published when the session was crawled (audio banked, text ~empty).
       FIXABLE without re-downloading video: re-fetch the API JSON, re-clean,
       and if the transcript has matured, rewrite transcript/turns + manifest.
  HIGH chars/sec (> --max-cps): audio much shorter than the transcript
       implies (server-side truncated/malformed video). NOT fixable from the
       API; reported, and with --reset-high the session's .ok + manifest row
       + audio are removed so the next crawl run re-processes it fully.

Default is a DRY RUN (report only). Use --apply to fetch + repair.

    ./venv/bin/python refresh_bg_parliament.py --out data/bg_parliament          # report
    ./venv/bin/python refresh_bg_parliament.py --out data/bg_parliament --apply  # repair LOW
"""
import argparse
import json
import os
import pathlib
import time

import bg_sten_clean
from prep_bg_parliament import api_get, make_session


def cps(row):
    return len(row.get("text") or "") / max(row.get("duration") or 1e-9, 1e-9)


def rewrite_manifest(path: pathlib.Path, rows):
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/bg_parliament")
    ap.add_argument("--min-cps", type=float, default=1.0)
    ap.add_argument("--max-cps", type=float, default=40.0)
    ap.add_argument("--apply", action="store_true",
                    help="re-fetch transcripts for LOW sessions and repair")
    ap.add_argument("--reset-high", action="store_true",
                    help="delete .ok/audio/manifest row for HIGH sessions so "
                         "the next crawl re-processes them (re-downloads video)")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    manifest_path = out / "manifest.jsonl"
    rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
    low = [r for r in rows if cps(r) < args.min_cps]
    high = [r for r in rows if cps(r) > args.max_cps]
    print(f"manifest: {len(rows)} sessions, "
          f"{sum(r['duration'] for r in rows)/3600:.1f} h")
    for tag, group in (("LOW", low), ("HIGH", high)):
        for r in group:
            print(f"  [{tag}] {r['id']} {r.get('session_date')} "
                  f"dur={r['duration']:.0f}s text={len(r.get('text') or '')}c "
                  f"({cps(r):.2f} c/s)")
    if not args.apply and not args.reset_high:
        print(f"\ndry run: {len(low)} repairable (--apply), "
              f"{len(high)} resettable (--reset-high)")
        return

    changed = False
    if args.apply and low:
        session = make_session(retries=5, backoff=2.0)
        for r in low:
            pl_id = r["pl_sten_id"]
            try:
                rec = api_get(session, pl_id)
                full_text, turns, meta = bg_sten_clean.clean_body(
                    rec.get("Pl_Sten_body") or "")
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {r['id']}: fetch/clean failed: {e!r}")
                continue
            new_cps = len(full_text) / max(r["duration"], 1e-9)
            if new_cps < args.min_cps:
                print(f"  [still-empty] {r['id']}: {len(full_text)}c "
                      f"({new_cps:.2f} c/s) — transcript not published yet")
            else:
                sid = r["id"]
                (out / "transcripts" / f"{sid}.txt").write_text(
                    full_text, encoding="utf-8")
                with open(out / "turns" / f"{sid}.jsonl", "w",
                          encoding="utf-8") as f:
                    for t in turns:
                        f.write(json.dumps(t, ensure_ascii=False) + "\n")
                r["text"] = full_text
                r["headers"] = meta["headers"]
                changed = True
                print(f"  [repaired] {sid}: {len(full_text)}c "
                      f"({new_cps:.2f} c/s), {len(turns)} turns")
            time.sleep(args.sleep)

    if args.reset_high and high:
        keep_ids = {r["id"] for r in high}
        for r in high:
            sid = r["id"]
            (out / "state" / f"{sid}.ok").unlink(missing_ok=True)
            audio = pathlib.Path(r["audio_path"])
            audio.unlink(missing_ok=True)
            print(f"  [reset] {sid}: .ok + audio removed — next crawl run "
                  f"re-processes it")
        rows = [r for r in rows if r["id"] not in keep_ids]
        changed = True

    if changed:
        rewrite_manifest(manifest_path, rows)
        print(f"manifest rewritten: {len(rows)} rows")


if __name__ == "__main__":
    main()
