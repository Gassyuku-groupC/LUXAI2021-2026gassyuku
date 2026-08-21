#!/usr/bin/env python3
"""Batch crawl public Kaggle episode JSON files by episode id range."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EPISODE_URL = "https://www.kaggleusercontent.com/episodes/{episode_id}.json"


def existing_episode_ids(output_dir: Path) -> List[int]:
    ids = []
    for path in output_dir.glob("*.json"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(ids)


def episode_range(args: argparse.Namespace) -> range:
    if args.from_existing:
        ids = existing_episode_ids(args.output_dir)
        if not ids:
            raise ValueError(f"No numeric JSON filenames found in {args.output_dir}")
        if args.scan_existing_window:
            start = min(ids) - args.before
            end = max(ids) + args.after
        else:
            start = max(ids) + 1
            end = max(ids) + args.after
    else:
        if args.start is None or args.end is None:
            raise ValueError("Provide --start/--end or use --from-existing.")
        start = args.start
        end = args.end
    if end < start:
        raise ValueError(f"Invalid range: {start}..{end}")
    return range(max(start, 1), end + 1)


def fetch_episode(episode_id: int, timeout: float) -> Tuple[Optional[bytes], str]:
    req = Request(
        EPISODE_URL.format(episode_id=episode_id),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.read(), "downloaded"
    except HTTPError as exc:
        return None, f"http_{exc.code}"
    except URLError as exc:
        reason = exc.reason
        winerror = getattr(reason, "winerror", None)
        errno = getattr(reason, "errno", None)
        if winerror is not None:
            return None, f"url_error_win_{winerror}"
        if errno is not None:
            return None, f"url_error_errno_{errno}"
        return None, f"url_error:{reason.__class__.__name__}"
    except TimeoutError:
        return None, "timeout"


def parse_episode(data: bytes) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        replay = json.loads(data.decode("utf-8"))
    except Exception as exc:
        return None, f"bad_json:{exc}"
    if not isinstance(replay, dict):
        return None, "not_object"
    if replay.get("name") != "lux_ai_2021":
        return None, f"wrong_game:{replay.get('name')}"
    if not replay.get("steps"):
        return None, "missing_steps"
    return replay, "ok"


def team_match(
    replay: Dict[str, Any],
    include_teams: List[str],
    require_all_teams: bool,
) -> bool:
    if not include_teams:
        return True
    names = [str(name) for name in replay.get("info", {}).get("TeamNames") or []]
    if require_all_teams:
        return all(team in names for team in include_teams)
    return any(team in names for team in include_teams)


def row_for_existing(path: Path) -> Dict[str, Any]:
    return {
        "episode_id": path.stem,
        "status": "existing",
        "path": str(path),
        "bytes": path.stat().st_size,
        "teams": "",
        "width": "",
        "height": "",
        "turns": "",
        "reason": "",
    }


def crawl_one(
    episode_id: int,
    output_dir: Path,
    include_teams: List[str],
    require_all_teams: bool,
    overwrite: bool,
    timeout: float,
) -> Dict[str, Any]:
    out_path = output_dir / f"{episode_id}.json"
    if out_path.exists() and not overwrite:
        return row_for_existing(out_path)

    data, fetch_status = fetch_episode(episode_id, timeout)
    if data is None:
        return {
            "episode_id": episode_id,
            "status": "skipped",
            "path": "",
            "bytes": 0,
            "teams": "",
            "width": "",
            "height": "",
            "turns": "",
            "reason": fetch_status,
        }

    replay, parse_status = parse_episode(data)
    if replay is None:
        return {
            "episode_id": episode_id,
            "status": "skipped",
            "path": "",
            "bytes": len(data),
            "teams": "",
            "width": "",
            "height": "",
            "turns": "",
            "reason": parse_status,
        }

    names = [str(name) for name in replay.get("info", {}).get("TeamNames") or []]
    if not team_match(replay, include_teams, require_all_teams):
        return {
            "episode_id": episode_id,
            "status": "filtered",
            "path": "",
            "bytes": len(data),
            "teams": " | ".join(names),
            "width": replay["steps"][0][0]["observation"].get("width", ""),
            "height": replay["steps"][0][0]["observation"].get("height", ""),
            "turns": len(replay.get("steps") or []),
            "reason": "team_filter",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return {
        "episode_id": episode_id,
        "status": "saved",
        "path": str(out_path),
        "bytes": len(data),
        "teams": " | ".join(names),
        "width": replay["steps"][0][0]["observation"].get("width", ""),
        "height": replay["steps"][0][0]["observation"].get("height", ""),
        "turns": len(replay.get("steps") or []),
        "reason": "",
    }


def write_index(index_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    exists = index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8-sig") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=[
                "episode_id",
                "status",
                "path",
                "bytes",
                "teams",
                "width",
                "height",
                "turns",
                "reason",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl public Kaggle Lux AI 2021 episode JSONs by numeric episode id. "
            "Use a small worker count to avoid hammering Kaggle."
        )
    )
    parser.add_argument("--start", type=int, help="First episode id to try.")
    parser.add_argument("--end", type=int, help="Last episode id to try.")
    parser.add_argument(
        "--from-existing",
        action="store_true",
        help="Start after the largest numeric JSON filename in --output-dir.",
    )
    parser.add_argument(
        "--scan-existing-window",
        action="store_true",
        help="With --from-existing, scan min(existing)-before through max(existing)+after instead.",
    )
    parser.add_argument("--before", type=int, default=0)
    parser.add_argument("--after", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/raw/data"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("dataset/raw/data_crawl_index.csv"),
    )
    parser.add_argument(
        "--include-team",
        action="append",
        default=[],
        help="Save only replays containing this exact TeamNames value. Repeatable.",
    )
    parser.add_argument(
        "--include-teams-file",
        type=Path,
        help="Text file with one TeamNames value per line.",
    )
    parser.add_argument(
        "--require-all-teams",
        action="store_true",
        help="Require every --include-team to appear in the replay.",
    )
    parser.add_argument("--max-downloads", type=int, help="Stop after saving this many new files.")
    parser.add_argument(
        "--stop-after-empty-batches",
        type=int,
        default=20,
        help="Stop after this many consecutive batches save no new files. Default: 20.",
    )
    parser.add_argument(
        "--index-status",
        choices=["all", "saved", "saved-existing"],
        default="all",
        help="Rows to append to the crawl CSV. Use saved-existing for an Excel-friendly compact index.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.include_teams_file:
        teams_from_file = [
            line.strip().lstrip("\ufeff")
            for line in args.include_teams_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        args.include_team = [*args.include_team, *teams_from_file]

    ids = list(episode_range(args))
    saved = 0
    attempted = 0
    empty_batches = 0
    print(f"trying episode ids: {ids[0]}..{ids[-1]} ({len(ids)} ids)")

    for offset in range(0, len(ids), args.batch_size):
        batch = ids[offset: offset + args.batch_size]
        rows = []
        with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
            futures = {
                pool.submit(
                    crawl_one,
                    episode_id,
                    args.output_dir,
                    args.include_team,
                    args.require_all_teams,
                    args.overwrite,
                    args.timeout,
                ): episode_id
                for episode_id in batch
            }
            for future in as_completed(futures):
                row = future.result()
                attempted += 1
                rows.append(row)
                if row["status"] == "saved":
                    saved += 1
                    print(
                        f"saved {row['episode_id']}: {row['teams']} "
                        f"{row['width']}x{row['height']}"
                    )
                if args.max_downloads is not None and saved >= args.max_downloads:
                    break
        saved_in_batch = sum(row["status"] == "saved" for row in rows)
        if args.index_status == "saved":
            rows_to_write = [row for row in rows if row["status"] == "saved"]
        elif args.index_status == "saved-existing":
            rows_to_write = [row for row in rows if row["status"] in {"saved", "existing"}]
        else:
            rows_to_write = rows
        if rows_to_write:
            write_index(args.index, rows_to_write)
        if saved_in_batch:
            empty_batches = 0
        else:
            empty_batches += 1
        print(
            f"progress: attempted={attempted} saved={saved} "
            f"batch_saved={saved_in_batch} empty_batches={empty_batches} index={args.index}"
        )
        if args.max_downloads is not None and saved >= args.max_downloads:
            break
        if args.stop_after_empty_batches and empty_batches >= args.stop_after_empty_batches:
            print(f"stopped: no new saved replays in {empty_batches} consecutive batches")
            break
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
