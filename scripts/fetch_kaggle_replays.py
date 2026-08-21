import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


EPISODE_URL = "https://www.kaggleusercontent.com/episodes/{episode_id}.json"


def parse_replay_ref(text, default_submission_id=None):
    text = text.strip()
    if not text:
        return None

    parsed = urlparse(text)
    query = parse_qs(parsed.query)
    episode_id = first(query.get("episodeId"))
    submission_id = first(query.get("submissionId")) or default_submission_id

    if episode_id is None:
        match = re.search(r"episodes[-_/]episode[-_/](\d+)", text)
        if match:
            episode_id = match.group(1)

    if episode_id is None:
        match = re.search(r"(?:episodeId=|episodes/|episode-)(\d+)", text)
        if match:
            episode_id = match.group(1)

    if episode_id is None and re.fullmatch(r"\d+", text):
        episode_id = text

    if episode_id is None:
        raise ValueError(f"Could not find an episode id in: {text}")
    if submission_id is None:
        raise ValueError(
            "Could not find a submissionId. Add it to the URL or pass "
            "--submission-id."
        )
    return submission_id, episode_id, text


def first(values):
    if not values:
        return None
    return values[0]


def download_direct(episode_id):
    url = EPISODE_URL.format(episode_id=episode_id)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response:
        return response.read()


def download_with_kaggle_cli(episode_id):
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("kaggle CLI was not found on PATH")

    with tempfile.TemporaryDirectory(prefix="lux_replay_") as tmp:
        tmp_path = Path(tmp)
        result = subprocess.run(
            [
                kaggle,
                "competitions",
                "replay",
                str(episode_id),
                "-p",
                str(tmp_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        candidates = sorted(tmp_path.glob("*.json"))
        if not candidates:
            raise RuntimeError("kaggle CLI did not produce a JSON replay file")
        return candidates[0].read_bytes()


def download_replay(episode_id, method):
    errors = []
    methods = ["direct", "kaggle"] if method == "auto" else [method]
    for item in methods:
        try:
            if item == "direct":
                return download_direct(episode_id), item
            if item == "kaggle":
                return download_with_kaggle_cli(episode_id), item
            raise ValueError(f"Unknown download method: {item}")
        except Exception as exc:
            errors.append(f"{item}: {exc}")
    raise RuntimeError("; ".join(errors))


def validate_json(data, episode_id):
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Downloaded episode {episode_id} is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise ValueError(f"Downloaded episode {episode_id} is not a JSON object")
    return parsed


def append_index(index_path, rows):
    exists = index_path.exists()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "submission_id",
                "episode_id",
                "path",
                "bytes",
                "method",
                "source",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def collect_refs(args):
    refs = list(args.refs)
    if args.input_file:
        for line in Path(args.input_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                refs.append(line)
    if not refs:
        raise ValueError("Provide replay URLs/episode ids or --input-file.")
    return refs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Download Kaggle Lux replay JSON files into "
            "dataset/raw/<submissionId>/<episodeId>.json."
        )
    )
    parser.add_argument("refs", nargs="*", help="Kaggle replay URLs or episode ids")
    parser.add_argument("--input-file", help="Text file with one URL/episode id per line")
    parser.add_argument(
        "--submission-id",
        help="Submission id to use when a ref is only an episode id.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/raw",
        help="Output root. Default: dataset/raw",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "direct", "kaggle"],
        default="auto",
        help="Download method. Default tries direct URL, then kaggle CLI.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    rows = []
    for ref in collect_refs(args):
        submission_id, episode_id, source = parse_replay_ref(ref, args.submission_id)
        out_dir = output_root / str(submission_id)
        out_path = out_dir / f"{episode_id}.json"

        if out_path.exists() and not args.overwrite:
            data = out_path.read_bytes()
            validate_json(data, episode_id)
            print(f"exists: {out_path}")
            rows.append(
                {
                    "submission_id": submission_id,
                    "episode_id": episode_id,
                    "path": str(out_path),
                    "bytes": len(data),
                    "method": "existing",
                    "source": source,
                }
            )
            continue

        data, method = download_replay(episode_id, args.method)
        validate_json(data, episode_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        print(f"downloaded: {out_path} ({len(data)} bytes via {method})")
        rows.append(
            {
                "submission_id": submission_id,
                "episode_id": episode_id,
                "path": str(out_path),
                "bytes": len(data),
                "method": method,
                "source": source,
            }
        )

    append_index(output_root / "replays_index.csv", rows)
    print(f"index: {output_root / 'replays_index.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
