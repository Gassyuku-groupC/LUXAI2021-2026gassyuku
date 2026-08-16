#!/usr/bin/env python3
"""Parse Kaggle leaderboard raw CSVs into a normalized team table."""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


COLUMN_ALIASES = {
    "rank": ("rank", "Rank", "PublicRank", "PrivateRank"),
    "team_id": ("team_id", "TeamId", "teamId", "Team ID"),
    "team_name": ("team_name", "TeamName", "teamName", "Team Name", "Team"),
    "submission_id": ("submission_id", "SubmissionId", "submissionId", "Submission ID"),
    "score": ("score", "Score", "PublicScore", "PrivateScore"),
    "submission_date": ("submission_date", "SubmissionDate", "submissionDate", "Submission Date"),
}


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def candidate_csvs(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))
    return []


def extract_zip_csvs(input_path: Path, extract_dir: Path) -> List[Path]:
    zips = []
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        zips.append(input_path)
    elif input_path.is_dir():
        zips.extend(sorted(input_path.glob("*.zip")))

    extracted = []
    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".csv"):
                    archive.extract(name, extract_dir)
                    extracted.append(extract_dir / name)
    return extracted


def first_present(row: Dict[str, str], aliases: Iterable[str]) -> str:
    for alias in aliases:
        if alias in row:
            return row.get(alias, "")
    return ""


def parse_rank(raw: str, fallback: int) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return fallback


def normalize_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as in_file:
        reader = csv.DictReader(in_file)
        rows = []
        for index, row in enumerate(reader, start=1):
            normalized = {
                "rank": parse_rank(first_present(row, COLUMN_ALIASES["rank"]), index),
                "team_id": first_present(row, COLUMN_ALIASES["team_id"]),
                "team_name": first_present(row, COLUMN_ALIASES["team_name"]),
                "submission_id": first_present(row, COLUMN_ALIASES["submission_id"]),
                "score": first_present(row, COLUMN_ALIASES["score"]),
                "submission_date": first_present(row, COLUMN_ALIASES["submission_date"]),
                "source": str(csv_path),
            }
            rows.append(normalized)
    rows.sort(key=lambda item: item["rank"])
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=[
                "rank",
                "team_id",
                "team_name",
                "submission_id",
                "score",
                "submission_date",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_team_list(path: Path, rows: List[Dict[str, str]], top: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    for row in rows:
        team = row["team_name"].strip()
        if not team or team.startswith("[Deleted]"):
            continue
        selected.append(team)
        if len(selected) >= top:
            break
    path.write_text("\n".join(selected) + "\n", encoding="utf-8-sig")


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Normalize Kaggle leaderboard CSV data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/leaderboard"),
        help="Leaderboard CSV, zip, or directory containing Kaggle leaderboard downloads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/processed/leaderboard_teams.csv"),
    )
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--teams-output",
        type=Path,
        default=Path("dataset/processed/leaderboard_top_teams.txt"),
        help="Write the top team names for replay crawling.",
    )
    args = parser.parse_args()

    extracted_dir = args.input / "_extracted" if args.input.is_dir() else args.input.parent / "_extracted"
    csv_paths = candidate_csvs(args.input) + extract_zip_csvs(args.input, extracted_dir)
    if not csv_paths:
        raise ValueError(f"No CSV leaderboard files found in {args.input}")

    rows = []
    for csv_path in csv_paths:
        rows.extend(normalize_rows(csv_path))
    rows.sort(key=lambda item: item["rank"])
    write_csv(args.output, rows)
    write_team_list(args.teams_output, rows, args.top)

    print(f"rows: {len(rows)}")
    print(f"output: {args.output}")
    print(f"teams: {args.teams_output}")
    print("top teams:")
    for row in rows[: args.top]:
        submission = row["submission_id"] or "-"
        score = row["score"] or "-"
        print(f"{row['rank']:>4}  {row['team_name']}  score={score} submission={submission}")


if __name__ == "__main__":
    main()
