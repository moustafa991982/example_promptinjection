from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import dataset as ds
from .client import LLMClient
from .report import load_results, summarize, to_csv_rows, to_html, to_markdown
from .runner import Runner

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: Optional[Path]) -> dict:
    if path is None or not Path(path).exists():
        return {}
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed; ignoring config file. pip install pyyaml", file=sys.stderr)
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_client(args, cfg: dict) -> LLMClient:
    t = cfg.get("target", {})
    return LLMClient(
        base_url=args.base_url or t.get("base_url", "http://localhost:11434/v1"),
        model=args.model or t.get("model", "llama3.1:8b"),
        backend=args.backend or t.get("backend", "openai"),
        api_key=t.get("api_key", "not-needed"),
        temperature=args.temperature if args.temperature is not None else t.get("temperature", 0.0),
        max_tokens=args.max_tokens or t.get("max_tokens", 512),
        top_p=t.get("top_p", 1.0),
        seed=args.seed if args.seed is not None else t.get("seed"),
        timeout=args.timeout or t.get("timeout", 180.0),
        max_retries=t.get("max_retries", 3),
        num_ctx=t.get("num_ctx"),
        keep_alive=t.get("keep_alive", "10m"),
        extra_body=t.get("extra_body"),
    )


def resolve_system_prompt(args, cfg: dict):
    profiles = cfg.get("system_profiles", {})
    name = args.system_profile
    if args.system_prompt_file:
        return Path(args.system_prompt_file).read_text(encoding="utf-8"), "file"
    if name in (None, "none", ""):
        return None, "none"
    if name not in profiles:
        raise SystemExit(f"unknown system profile '{name}'. available: {', '.join(profiles) or '(none)'}")
    return profiles[name], name


# ---------------------------------------------------------------- commands
def cmd_check(args, cfg):
    async def _go():
        async with build_client(args, cfg) as client:
            print(f"target : {client.base_url}  backend={client.backend}  model={client.model}")
            comp = await client.healthcheck()
            if comp.ok:
                print(f"status : OK  ({comp.latency_s:.2f}s)")
                print(f"reply  : {comp.text.strip()[:200]!r}")
                print(f"tokens : prompt={comp.prompt_tokens} completion={comp.completion_tokens}")
            else:
                print(f"status : FAILED\nerror  : {comp.error}")
                raise SystemExit(1)
    asyncio.run(_go())


def cmd_list(args, cfg):
    d = ds.ensure_dataset(Path(args.dataset_dir), auto_clone=not args.no_clone)
    cases = ds.load_cases(d, groups=args.groups, extra_file=args.extra)
    from collections import Counter
    counts = Counter(c.group_mapping for c in cases)
    print(f"{len(cases)} cases from {d}  (rev {ds.dataset_revision(d)[:10]})\n")
    print(f"{'family':<20} {'n':>3}  expected strings")
    print("-" * 78)
    for fam, n in counts.most_common():
        ex = next(c for c in cases if c.group_mapping == fam)
        flags = []
        if ex.exact_matching: flags.append("exact")
        if ex.word_matching: flags.append("word")
        if not ex.case_sensitive: flags.append("nocase")
        if not ex.punctuation_sensitive: flags.append("nopunct")
        if ex.all_expected_strings_must_be_found: flags.append("all")
        print(f"{fam:<20} {n:>3}  {ex.expected_strings}  [{','.join(flags)}]")


def cmd_run(args, cfg):
    d = ds.ensure_dataset(Path(args.dataset_dir), auto_clone=not args.no_clone)
    cases = ds.load_cases(
        d, groups=args.groups, languages=args.languages,
        extra_file=args.extra, limit=args.limit,
    )
    if not cases:
        raise SystemExit("no cases selected")

    system_prompt, profile_name = resolve_system_prompt(args, cfg)
    canary = args.canary or cfg.get("canary")
    vectors = ["direct", "indirect"] if args.vector == "both" else [args.vector]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    total = len(cases) * len(vectors) * args.repeats
    print(f"[plan] {len(cases)} prompts x {len(vectors)} vector(s) x {args.repeats} repeat(s) = {total} trials")
    print(f"[plan] profile={profile_name} concurrency={args.concurrency} -> {results_path}")
    if args.dry_run:
        for c in cases[:5]:
            print(f"  {c.case_id} [{c.group_mapping}] {c.name}: {c.prompt[:90]!r}...")
        return

    async def _go():
        async with build_client(args, cfg) as client:
            hc = await client.healthcheck()
            if not hc.ok:
                raise SystemExit(f"target unreachable: {hc.error}")
            runner = Runner(
                client=client, cases=cases, out_path=results_path,
                system_prompt=system_prompt, canary=canary, vectors=vectors,
                repeats=args.repeats, concurrency=args.concurrency,
                base_seed=args.seed, resume=not args.no_resume,
            )
            await runner.run()

    asyncio.run(_go())

    meta = {
        "system_profile": profile_name,
        "dataset_revision": ds.dataset_revision(d),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _emit_reports(results_path, outdir, meta)


def cmd_report(args, cfg):
    outdir = Path(args.outdir)
    results_path = Path(args.results) if args.results else outdir / "results.jsonl"
    meta_path = outdir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    _emit_reports(results_path, outdir, meta)


def _emit_reports(results_path: Path, outdir: Path, meta: dict):
    rows = load_results(results_path)
    if not rows:
        raise SystemExit(f"no results in {results_path}")
    s = summarize(rows)

    (outdir / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    (outdir / "report.md").write_text(to_markdown(s, meta), encoding="utf-8")
    (outdir / "report.html").write_text(to_html(s, meta), encoding="utf-8")

    csv_rows = to_csv_rows(rows)
    with open(outdir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    st, rl = s["overall_strict"], s["overall_relaxed"]
    print(f"\nASR strict  : {100*st['rate']:.1f}%  ({st['successes']}/{st['trials']})"
          f"  CI {100*st['ci_low']:.1f}–{100*st['ci_high']:.1f}%")
    print(f"ASR relaxed : {100*rl['rate']:.1f}%  ({rl['successes']}/{rl['trials']})")
    print(f"refusals    : {100*s['refusal_rate']:.1f}%   canary leaks: {s['canary_leaks']}")
    print(f"errors      : {s['error_trials']}")
    print(f"\nwrote {outdir}/report.md, report.html, results.csv, summary.json")


# ---------------------------------------------------------------- parser
def main(argv=None):
    p = argparse.ArgumentParser(
        prog="piharness",
        description="Prompt-injection test harness for local LLaMA endpoints, "
                    "driven by the Giskard-AI/prompt-injections corpus.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--base-url", help="e.g. http://localhost:8080/v1 or http://localhost:11434")
    p.add_argument("--model")
    p.add_argument("--backend", choices=["openai", "ollama"])
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--timeout", type=float)
    p.add_argument("--seed", type=int, help="base seed; repeat i uses seed+i")
    p.add_argument("--dataset-dir", default="data/prompt-injections")
    p.add_argument("--no-clone", action="store_true")
    p.add_argument("--outdir", default="runs/latest")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify the endpoint answers").set_defaults(func=cmd_check)

    pl = sub.add_parser("list", help="show attack families and their matcher flags")
    pl.add_argument("--groups", nargs="*")
    pl.add_argument("--extra", type=Path)
    pl.set_defaults(func=cmd_list)

    pr = sub.add_parser("run", help="execute the corpus against the target")
    pr.add_argument("--groups", nargs="*", help="filter by family, e.g. DAN 'Hate Speech'")
    pr.add_argument("--languages", nargs="*")
    pr.add_argument("--extra", type=Path, help="JSON file of your own injections")
    pr.add_argument("--limit", type=int)
    pr.add_argument("--repeats", type=int, default=1)
    pr.add_argument("--concurrency", type=int, default=4)
    pr.add_argument("--vector", choices=["direct", "indirect", "both"], default="direct")
    pr.add_argument("--system-profile", default="none")
    pr.add_argument("--system-prompt-file", type=Path)
    pr.add_argument("--canary")
    pr.add_argument("--no-resume", action="store_true")
    pr.add_argument("--dry-run", action="store_true")
    pr.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="rebuild reports from an existing results.jsonl")
    rp.add_argument("--results", type=Path)
    rp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
