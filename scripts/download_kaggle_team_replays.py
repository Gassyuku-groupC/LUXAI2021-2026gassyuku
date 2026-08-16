#!/usr/bin/env python3
"""Download Lux AI 2021 replays through Kaggle team/submission/episode APIs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen


EPISODE_URL = "https://www.kaggleusercontent.com/episodes/{episode_id}.json"


class RateLimitError(RuntimeError):
    pass


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def kaggle_exe(project_root: Path) -> str:
    local = project_root / ".venv" / "Scripts" / "kaggle.exe"
    if local.exists():
        return str(local)
    found = shutil.which("kaggle")
    if found:
        return found
    raise RuntimeError("kaggle CLI was not found. Use .venv\\Scripts\\kaggle.exe or install kaggle.")


def run_kaggle_csv(kaggle: str, args: List[str]) -> List[Dict[str, str]]:
    result = subprocess.run(
        [kaggle, *args, "--format", "csv", "-q"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    text = result.stdout.strip()
    if not text or "No " in text:
        return []
    return list(csv.DictReader(io.StringIO(text)))


def read_leaderboard(path: Path, top: int) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as in_file:
        rows = list(csv.DictReader(in_file))
    selected = []
    for row in rows:
        team_name = row.get("team_name") or row.get("TeamName") or ""
        team_id = row.get("team_id") or row.get("TeamId") or ""
        if not team_name or not team_id or team_name.startswith("[Deleted]"):
            continue
        selected.append({"team_id": team_id, "team_name": team_name})
        if len(selected) >= top:
            break
    return selected


def append_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def get_team_submissions(
    kaggle: str,
    team: Dict[str, str],
    limit: int,
) -> List[Dict[str, str]]:
    rows = run_kaggle_csv(kaggle, ["competitions", "team-submissions", team["team_id"]])
    submissions = []
    for row in rows[:limit]:
        submission_id = row.get("id") or row.get("Id") or row.get("ref") or row.get("Ref")
        if not submission_id:
            continue
        submissions.append(
            {
                "team_id": team["team_id"],
                "team_name": team["team_name"],
                "submission_id": submission_id,
                "date_submitted": row.get("dateSubmitted") or row.get("date") or "",
                "public_score": row.get("publicScore") or "",
            }
        )
    return submissions


def get_submission_episodes(
    kaggle: str,
    submission: Dict[str, str],
    limit: int,
) -> List[Dict[str, str]]:
    rows = run_kaggle_csv(kaggle, ["competitions", "episodes", submission["submission_id"]])
    episodes = []
    for row in rows[:limit]:
        episode_id = row.get("id") or row.get("Id")
        if not episode_id:
            continue
        episodes.append(
            {
                **submission,
                "episode_id": episode_id,
                "create_time": row.get("createTime") or "",
                "end_time": row.get("endTime") or "",
                "state": row.get("state") or "",
                "type": row.get("type") or "",
            }
        )
    return episodes


def download_direct(episode_id: str, timeout: float) -> bytes:
    req = Request(
        EPISODE_URL.format(episode_id=episode_id),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def download_with_kaggle_cli(
    kaggle: str,
    episode_id: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
    fail_fast_429: bool,
) -> bytes:
    last_error = ""
    for attempt in range(retries + 1):
        with tempfile.TemporaryDirectory(prefix="lux_kaggle_replay_") as tmp:
            tmp_path = Path(tmp)
            result = subprocess.run(
                [kaggle, "competitions", "replay", str(episode_id), "-p", str(tmp_path), "-q"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            if result.returncode == 0:
                candidates = sorted(tmp_path.glob("*.json"))
                if not candidates:
                    raise RuntimeError("kaggle replay produced no JSON file")
                return candidates[0].read_bytes()

            last_error = result.stderr.strip() or result.stdout.strip()
            if "429" in last_error and fail_fast_429:
                raise RateLimitError(last_error)
            if "429" not in last_error or attempt >= retries:
                break
            sleep_seconds = retry_sleep * (2 ** attempt)
            print(f"rate limited on episode {episode_id}; sleeping {sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(last_error)


def download_replay_bytes(
    kaggle: str,
    episode_id: str,
    timeout: float,
    method: str,
    retries: int,
    retry_sleep: float,
    fail_fast_429: bool,
) -> Tuple[bytes, str]:
    errors = []
    methods = ["direct", "kaggle"] if method == "auto" else [method]
    for item in methods:
        try:
            if item == "direct":
                return download_direct(episode_id, timeout), item
            if item == "kaggle":
                return download_with_kaggle_cli(
                    kaggle,
                    episode_id,
                    timeout,
                    retries,
                    retry_sleep,
                    fail_fast_429,
                ), item
            raise ValueError(f"unknown download method: {item}")
        except RateLimitError:
            raise
        except Exception as exc:
            message = str(exc).replace("\r", " ").replace("\n", " ").strip()
            errors.append(f"{item}:{exc.__class__.__name__}:{message[:300]}")
    raise RuntimeError(";".join(errors))


def replay_matches_lux(data: bytes) -> Tuple[bool, Dict[str, str]]:
    replay = json.loads(data.decode("utf-8"))
    if replay.get("name") != "lux_ai_2021":
        return False, {}
    first_obs = replay["steps"][0][0]["observation"]
    return True, {
        "teams": " | ".join(str(name) for name in replay.get("info", {}).get("TeamNames") or []),
        "width": str(first_obs.get("width", "")),
        "height": str(first_obs.get("height", "")),
        "turns": str(len(replay.get("steps") or [])),
        "rewards": " | ".join(str(reward) for reward in replay.get("rewards") or []),
    }


def download_one(
    row: Dict[str, str],
    output_dir: Path,
    timeout: float,
    overwrite: bool,
    kaggle: str,
    method: str,
    retries: int,
    retry_sleep: float,
    fail_fast_429: bool,
) -> Dict[str, str]:
    episode_id = row["episode_id"]
    out_path = output_dir / f"{episode_id}.json"
    if out_path.exists() and not overwrite:
        return {**row, "status": "existing", "path": str(out_path), "bytes": str(out_path.stat().st_size)}
    try:
        data, used_method = download_replay_bytes(
            kaggle,
            episode_id,
            timeout,
            method,
            retries,
            retry_sleep,
            fail_fast_429,
        )
        ok, meta = replay_matches_lux(data)
        if not ok:
            return {**row, "status": "wrong_game", "method": used_method, "path": "", "bytes": str(len(data))}
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return {
            **row,
            **meta,
            "status": "saved",
            "method": used_method,
            "path": str(out_path),
            "bytes": str(len(data)),
        }
    except RateLimitError as exc:
        return {
            **row,
            "status": "rate_limited",
            "method": method,
            "error": str(exc)[:500],
            "path": "",
            "bytes": "",
        }
    except Exception as exc:
        return {
            **row,
            "status": f"error:{exc.__class__.__name__}",
            "method": method,
            "error": str(exc)[:500],
            "path": "",
            "bytes": "",
        }


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(
        description="Use Kaggle official team/submission/episode APIs to download public replays."
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path("dataset/processed/leaderboard_teams.csv"),
    )
    parser.add_argument("--top-teams", type=int, default=30)
    parser.add_argument("--team-id", action="append", default=[])
    parser.add_argument("--submission-limit-per-team", type=int, default=3)
    parser.add_argument("--episode-limit-per-submission", type=int, default=50)
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="Skip this many episode rows before applying --max-downloads.",
    )
    parser.add_argument("--max-downloads", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=30.0)
    parser.add_argument(
        "--fail-fast-429",
        action="store_true",
        help="Mark a replay as rate_limited immediately instead of sleeping/retrying.",
    )
    parser.add_argument(
        "--download-sleep",
        type=float,
        default=0.0,
        help="Sleep after each replay download result. Useful for Kaggle rate limits.",
    )
    parser.add_argument(
        "--download-method",
        choices=["auto", "direct", "kaggle"],
        default="auto",
        help="Use direct kaggleusercontent download, official kaggle replay CLI, or try both.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/raw/data"))
    parser.add_argument("--submissions-output", type=Path, default=Path("dataset/raw/team_submissions.csv"))
    parser.add_argument("--episodes-output", type=Path, default=Path("dataset/raw/team_episodes.csv"))
    parser.add_argument(
        "--episodes-input",
        type=Path,
        help="Skip team/submission API calls and download replay ids from an existing episodes CSV.",
    )
    parser.add_argument("--downloads-output", type=Path, default=Path("dataset/raw/team_replay_downloads.csv"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    kaggle = kaggle_exe(root)
    if args.episodes_input:
        with args.episodes_input.open(encoding="utf-8-sig", newline="") as in_file:
            loaded_episodes = list(csv.DictReader(in_file))
        episodes = []
        seen_episode_ids = set()
        for episode in loaded_episodes:
            episode_id = episode.get("episode_id")
            if not episode_id or episode_id in seen_episode_ids:
                continue
            seen_episode_ids.add(episode_id)
            episodes.append(episode)
        submissions = []
        print(
            f"loaded episodes: {len(loaded_episodes)} rows, "
            f"{len(episodes)} unique from {args.episodes_input}"
        )
    else:
        teams = read_leaderboard(args.leaderboard, args.top_teams)
        for team_id in args.team_id:
            teams.append({"team_id": str(team_id), "team_name": f"team_{team_id}"})

        submissions = []
        for team in teams:
            try:
                team_submissions = get_team_submissions(kaggle, team, args.submission_limit_per_team)
                submissions.extend(team_submissions)
                print(f"team {team['team_name']} ({team['team_id']}): submissions={len(team_submissions)}")
            except Exception as exc:
                print(f"team {team['team_name']} ({team['team_id']}): {exc}")
        append_csv(
            args.submissions_output,
            submissions,
            ["team_id", "team_name", "submission_id", "date_submitted", "public_score"],
        )

        episodes = []
        seen_episodes = set()
        for submission in submissions:
            try:
                submission_episodes = get_submission_episodes(kaggle, submission, args.episode_limit_per_submission)
                for episode in submission_episodes:
                    if episode["episode_id"] in seen_episodes:
                        continue
                    seen_episodes.add(episode["episode_id"])
                    episodes.append(episode)
                print(f"submission {submission['submission_id']}: episodes={len(submission_episodes)}")
            except Exception as exc:
                print(f"submission {submission['submission_id']}: {exc}")
        append_csv(
            args.episodes_output,
            episodes,
            [
                "team_id",
                "team_name",
                "submission_id",
                "date_submitted",
                "public_score",
                "episode_id",
                "create_time",
                "end_time",
                "state",
                "type",
            ],
        )

    to_download = episodes[args.episode_offset: args.episode_offset + args.max_downloads]
    download_rows = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = [
            pool.submit(
                download_one,
                row,
                args.output_dir,
                args.timeout,
                args.overwrite,
                kaggle,
                args.download_method,
                args.retries,
                args.retry_sleep,
                args.fail_fast_429,
            )
            for row in to_download
        ]
        for future in as_completed(futures):
            row = future.result()
            download_rows.append(row)
            if row["status"] in {"saved", "existing"}:
                print(f"{row['status']} {row['episode_id']}: {row.get('teams', '')}")
            if args.download_sleep:
                time.sleep(args.download_sleep)
    append_csv(
        args.downloads_output,
        download_rows,
        [
            "team_id",
            "team_name",
            "submission_id",
            "date_submitted",
            "public_score",
            "episode_id",
            "create_time",
            "end_time",
            "state",
            "type",
            "teams",
            "width",
            "height",
            "turns",
            "rewards",
            "status",
            "method",
            "error",
            "path",
            "bytes",
        ],
    )
    saved = sum(row["status"] == "saved" for row in download_rows)
    existing = sum(row["status"] == "existing" for row in download_rows)
    print(f"done: submissions={len(submissions)} episodes={len(episodes)} saved={saved} existing={existing}")


if __name__ == "__main__":
    main()
