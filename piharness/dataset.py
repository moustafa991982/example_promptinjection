"""
Loads the Giskard-AI/prompt-injections corpus.

The upstream repo ships two files that must be joined on ``index``:
  prompt_injections.csv  -> prompt, name, group, source, language
  giskard_meta_data.csv  -> expected_strings + matcher flags, group_mapping,
                            description, deviation_description

``group`` is the coarse taxonomy (Jailbreak / Hijacking attacks) and
``group_mapping`` is the fine-grained attack family (DAN, DUDE, Hate Speech...).
We keep both.
"""

from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

UPSTREAM = "https://github.com/Giskard-AI/prompt-injections.git"
PROMPTS_FILE = "prompt_injections.csv"
META_FILE = "giskard_meta_data.csv"


@dataclass
class TestCase:
    case_id: str
    prompt: str
    name: str
    group: str
    group_mapping: str
    source: str
    language: str
    expected_strings: List[str]
    all_expected_strings_must_be_found: bool
    exact_matching: bool
    word_matching: bool
    case_sensitive: bool
    punctuation_sensitive: bool
    description: str = ""
    deviation_description: str = ""

    def meta_dict(self) -> dict:
        return {
            "expected_strings": self.expected_strings,
            "all_expected_strings_must_be_found": self.all_expected_strings_must_be_found,
            "exact_matching": self.exact_matching,
            "word_matching": self.word_matching,
            "case_sensitive": self.case_sensitive,
            "punctuation_sensitive": self.punctuation_sensitive,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _parse_expected(raw) -> List[str]:
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [str(raw)]
    if isinstance(parsed, str):
        return [parsed]
    return [str(x) for x in parsed]


def ensure_dataset(dataset_dir: Path, auto_clone: bool = True, pin: Optional[str] = None) -> Path:
    """Return a directory containing the two CSVs, cloning upstream if needed."""
    dataset_dir = Path(dataset_dir)
    if (dataset_dir / PROMPTS_FILE).exists() and (dataset_dir / META_FILE).exists():
        return dataset_dir

    if not auto_clone:
        raise FileNotFoundError(
            f"{PROMPTS_FILE} / {META_FILE} not found under {dataset_dir}. "
            f"Clone it manually:\n  git clone {UPSTREAM} {dataset_dir}"
        )

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] cloning {UPSTREAM} -> {dataset_dir}", file=sys.stderr)
    cmd = ["git", "clone", "--depth", "1", UPSTREAM, str(dataset_dir)]
    subprocess.run(cmd, check=True)
    if pin:
        subprocess.run(["git", "-C", str(dataset_dir), "fetch", "--depth", "1", "origin", pin], check=True)
        subprocess.run(["git", "-C", str(dataset_dir), "checkout", pin], check=True)
    return dataset_dir


def dataset_revision(dataset_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(dataset_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def load_cases(
    dataset_dir: Path,
    groups: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    extra_file: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[TestCase]:
    dataset_dir = Path(dataset_dir)

    with open(dataset_dir / PROMPTS_FILE, newline="", encoding="utf-8") as fh:
        prompts = {row["index"]: row for row in csv.DictReader(fh)}
    with open(dataset_dir / META_FILE, newline="", encoding="utf-8") as fh:
        meta = {row["index"]: row for row in csv.DictReader(fh)}

    missing = set(prompts) ^ set(meta)
    if missing:
        raise ValueError(
            f"prompt/meta index mismatch on {sorted(missing)[:10]} — the two CSVs "
            "must be a one-to-one mapping (see upstream validate.py)"
        )

    cases: List[TestCase] = []
    for idx in sorted(prompts, key=lambda x: int(x)):
        p, m = prompts[idx], meta[idx]
        cases.append(
            TestCase(
                case_id=f"giskard-{int(idx):03d}",
                prompt=p["prompt"],
                name=p.get("name", ""),
                group=p.get("group", ""),
                group_mapping=m.get("group_mapping", ""),
                source=p.get("source", ""),
                language=p.get("language", "en"),
                expected_strings=_parse_expected(m["expected_strings"]),
                all_expected_strings_must_be_found=_to_bool(m.get("all_expected_strings_must_be_found", True)),
                exact_matching=_to_bool(m.get("exact_matching", False)),
                word_matching=_to_bool(m.get("word_matching", False)),
                case_sensitive=_to_bool(m.get("case_sensitive", True)),
                punctuation_sensitive=_to_bool(m.get("punctuation_sensitive", True)),
                description=m.get("description", ""),
                deviation_description=m.get("deviation_description", ""),
            )
        )

    if extra_file:
        cases.extend(load_extra_cases(Path(extra_file)))

    if groups:
        wanted = {g.lower() for g in groups}
        cases = [c for c in cases if c.group_mapping.lower() in wanted or c.group.lower() in wanted]
    if languages:
        wanted = {l.lower() for l in languages}
        cases = [c for c in cases if c.language.lower() in wanted]
    if limit:
        cases = cases[:limit]

    return cases


def load_extra_cases(path: Path) -> List[TestCase]:
    """Load your own injections from a JSON file.

    Each entry needs at minimum ``prompt`` and ``expected_strings``. Everything
    else falls back to permissive defaults. Use this to bolt domain-specific
    attacks onto the public corpus — e.g. probes that try to make a vehicle
    support assistant emit diagnostic session unlock hints or leak a system
    prompt containing fleet identifiers.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    out = []
    for i, item in enumerate(raw):
        out.append(
            TestCase(
                case_id=item.get("case_id", f"custom-{i:03d}"),
                prompt=item["prompt"],
                name=item.get("name", f"custom-{i}"),
                group=item.get("group", "Custom"),
                group_mapping=item.get("group_mapping", item.get("group", "Custom")),
                source=item.get("source", str(path)),
                language=item.get("language", "en"),
                expected_strings=_parse_expected(item["expected_strings"]),
                all_expected_strings_must_be_found=_to_bool(item.get("all_expected_strings_must_be_found", False)),
                exact_matching=_to_bool(item.get("exact_matching", False)),
                word_matching=_to_bool(item.get("word_matching", False)),
                case_sensitive=_to_bool(item.get("case_sensitive", False)),
                punctuation_sensitive=_to_bool(item.get("punctuation_sensitive", True)),
                description=item.get("description", ""),
                deviation_description=item.get("deviation_description", ""),
            )
        )
    return out
