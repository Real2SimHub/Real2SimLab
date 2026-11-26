"""``r2s report RUN_DIR``: Markdown and CSV tables and a static HTML report that opens offline.

The HTML report opens with the scorecard: per benchmark, each method's headline metric (the
benchmark's own, combined the benchmark's way) with its reference points and the share of
deliveries the frozen re-run reproduced. Per task: the recorded video with each twin's projected
trajectory drawn on it, the scores, plots of the reference and each twin's trajectories, the
official report of native scorers, and each agent's redacted trajectory. Everything (videos
included) is copied into ``report/``, so the directory can be zipped and shared.
"""

from __future__ import annotations

import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from real2simlab import __version__
from real2simlab._fs import read_regular
from real2simlab.agents.base import read_trajectory
from real2simlab.agents.record import AgentRecord, recorded_file
from real2simlab.errors import R2SError, SetupError
from real2simlab.formats.base import get_format
from real2simlab.formats.tracks import Tracks
from real2simlab.orchestrate.layout import (
    RunLayout,
    incomplete,
    read_experiment,
    read_result,
    read_results,
    slug,
)
from real2simlab.report.svg import Series, line_plot
from real2simlab.scoring.model import TaskResult, method_label
from real2simlab.task import Task, load_task

LOCAL_MEDIA = "local-media"
"""Where a run keeps media of recordings that may not be redistributed (beside report/)."""
COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b")
STYLE = """
:root { --fg: #1d2330; --muted: #5d6575; --line: #d9dde5; --bg: #ffffff; --panel: #f6f7f9;
        --good: #1a7f37; --bad: #b42318; --warn: #9a6700; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e8ec; --muted: #9aa3b2; --line: #343a46; --bg: #14171c; --panel: #1c2027;
          --good: #3fb950; --bad: #f85149; --warn: #d29922; } }
body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       color: var(--fg); background: var(--bg); margin: 0 auto; max-width: 1180px; padding: 16px; }
h1 { font-size: 22px; } h2 { font-size: 18px; margin-top: 28px; } h3 { font-size: 15px; }
table { border-collapse: collapse; margin: 8px 0 16px; display: block; overflow-x: auto; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 10px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { background: var(--panel); font-weight: 600; }
.muted { color: var(--muted); } .good { color: var(--good); } .bad { color: var(--bad); }
.warn { color: var(--warn); font-weight: 600; }
.flag { display: inline-block; padding: 0 6px; border-radius: 8px; font-size: 12px;
        border: 1px solid currentColor; margin-left: 4px; }
.grid { stroke: var(--line); stroke-width: 0.6; }
.plot-frame { fill: none; stroke: var(--line); }
.plot text { fill: var(--fg); font-size: 10px; } .plot .plot-title { font-weight: 600; font-size: 11px; }
.plots { display: flex; flex-wrap: wrap; gap: 8px; }
video { max-width: 100%; border: 1px solid var(--line); }
.videos { display: flex; flex-wrap: wrap; gap: 12px; } .videos figure { margin: 0; max-width: 340px; }
figcaption { color: var(--muted); font-size: 12px; }
details { background: var(--panel); border-radius: 6px; padding: 6px 10px; margin: 6px 0; }
pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; max-height: 420px; overflow: auto; }
a { color: #1f6feb; }
"""


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if abs(value) >= 1000 or (abs(value) < 1e-3 and value != 0):
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _estimate(data: dict[str, Any] | None) -> str:
    if not data or data.get("mean") is None:
        return "—"
    text = f"{data['mean']:.3f}"
    if (
        data.get("ci_low") is not None
        and data.get("ci_high") is not None
        and data["ci_low"] != data["ci_high"]
    ):
        text += f' <span class="muted">[{data["ci_low"]:.3f}, {data["ci_high"]:.3f}]</span>'
    return text


def _rerun_cell(agreed: int, compared: int, total: int, rates: list[float]) -> str:
    """``agreed/compared`` re-runs, with the results that had nothing to compare."""
    if compared:
        cell = f"{agreed}/{compared}"
        return cell + (f" ({total - compared} not compared)" if total > compared else "")
    if rates:
        return f"{sum(rates) / len(rates):.0%}"
    return "—"


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>{body}"
        f'<p class="muted">Generated by Real2SimLab {__version__}.</p></body></html>'
    )


class ReportBuilder:
    def __init__(self, run_dir: Path) -> None:
        self.layout = RunLayout(Path(run_dir))
        self.out = self.layout.report_dir
        self.results = read_results(self.layout)
        self.experiment = read_experiment(self.layout)
        # The results the experiment counted (``results.json`` rows): result files of an
        # earlier configuration, or of methods and tasks no longer in it, are left out.
        rows = self.results.get("rows")
        counted = {
            (row.get("method"), row.get("task"), row.get("trial"), row.get("seed"))
            for row in rows or []
        }
        self.task_results: list[TaskResult] = [
            result
            for result in (read_result(path) for path in self.layout.result_files())
            if rows is None or (result.method, result.task_id, result.trial, result.seed) in counted
        ]
        self.warnings: list[str] = [
            f"These results are not one complete set: {problem}. Run the same r2s eval or "
            "r2s run command again with this run directory to finish them."
            for problem in incomplete(self.layout.root, results=self.results)
        ]
        scored_hashes: dict[str, set[str]] = defaultdict(set)
        for result in self.task_results:
            scored_hashes[result.task_id].add(result.task_hash)
        self.tasks: dict[str, Task] = {}
        self.task_warnings: dict[str, str] = {}
        for task_id, path in self.experiment.get("tasks", {}).items():
            root = Path(path)
            root = root if root.is_absolute() else self.layout.root / root
            try:
                task = load_task(root)
            except R2SError:
                continue
            if scored_hashes.get(task_id) and task.content_hash() not in scored_hashes[task_id]:
                # The task at that path is another version than the one scored: its recording
                # and reference would be shown next to scores they did not produce.
                warning = (
                    f"The task {task_id} at {root} has changed since it was scored (its content "
                    "hash differs): its scores are those of the earlier version, and its "
                    "recording and reference are not shown. Evaluate it again to score this "
                    "version."
                )
                self.warnings.append(warning)
                self.task_warnings[task_id] = warning
                continue
            self.tasks[task_id] = task
        pending = sum(1 for result in self.task_results if result.infra_error is not None)
        if pending:
            # Shown on the HTML index (which lists no results), in report.md and by `r2s report`:
            # it says where each cause is.
            self.warnings.append(
                f"{pending} result(s) could not be scored because the infrastructure failed "
                "(a sandbox that did not start, a failed official scorer): they are left out "
                "of every number, and each one's cause is on its task's page (and in the "
                "results of report.md and results.csv). Once it is fixed, run the same command "
                "again to score them."
            )
        self.notes: list[str] = []
        self._outputs: dict[Path, Tracks | None] = {}
        self.changed_outputs: dict[str, list[str]] = defaultdict(list)

    # --- tables ---------------------------------------------------------------------------------

    def _scorecard(self) -> tuple[list[str], list[list[str]], list[bool], str]:
        """The scorecard: one column per benchmark (its headline metric), one row per method,
        then one row per reference point. Returns header, rows, which rows are references, and
        the note explaining it."""
        methods = self.results.get("methods", [])
        # One column per benchmark and headline metric (custom tasks sharing a benchmark name
        # may report different metrics).
        entries: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for method in methods:
            for entry in method.get("benchmarks", []):
                entries[entry["benchmark"], entry["metric"]][method["method"]] = entry
        benchmarks = sorted(entries)
        header = ["method"]
        for benchmark in benchmarks:
            first = next(iter(entries[benchmark].values()))
            arrow = "↑" if first.get("higher_is_better", True) else "↓"
            unit = f", {first['unit']}" if first.get("unit") else ""
            header.append(f"{benchmark[0]} ({first['metric']}{unit}) {arrow}")
        header += ["delivered", "re-run reproduced"]
        rows: list[list[str]] = []
        is_reference: list[bool] = []
        for method in methods:
            cells = [self._labels()[method["method"]]]
            agreed = compared = total = 0
            rates: list[float] = []
            for benchmark in benchmarks:
                entry = entries[benchmark].get(method["method"])
                if entry is None:
                    cells.append("")
                    continue
                scored = f" ({entry['scored']}/{entry['results']})"
                pending = entry.get("infra_errors") or 0
                if pending:
                    scored = (
                        f" ({entry['scored']}/{entry['results']}; {pending} not scored: "
                        "infrastructure)"
                    )
                cells.append(
                    _md(entry["value"], 3) + (scored if entry["scored"] < entry["results"] else "")
                )
                total += entry["results"]
                if entry.get("rerun_compared") is not None:
                    compared += entry["rerun_compared"]
                    agreed += entry.get("rerun_agreed") or 0
                else:  # results.json of an earlier version: only the rate is known
                    rate = entry.get("session_consistent_rate")
                    rate = entry.get("reproducible_rate") if rate is None else rate
                    if rate is not None:
                        rates.append(rate)
            cells.append(f"{_delivered_rate(method)}")
            cells.append(_rerun_cell(agreed, compared, total, rates))
            rows.append(cells)
            is_reference.append(False)
        names: list[str] = []
        for benchmark in benchmarks:
            for entry in entries[benchmark].values():
                names += [n for n in entry.get("references", {}) if n not in names]
        for name in names:
            cells = [name]
            for benchmark in benchmarks:
                value = next(
                    (
                        e["references"][name]
                        for e in entries[benchmark].values()
                        if name in e.get("references", {})
                    ),
                    None,
                )
                cells.append("" if value is None else f"{value:.3g}")
            rows.append([*cells, "", ""])
            is_reference.append(True)
        note = (
            "Each column is one benchmark's own headline metric, combined over its tasks the "
            f"benchmark's way, with {self.results.get('confidence', 0.95):.0%} bootstrap intervals "
            "over tasks; (scored / results) when some could not be scored. Re-run reproduced: "
            "of the deliveries whose frozen, offline re-run could be compared, those that gave "
            "the output the agent itself produced (or, without one, the same output twice); "
            "results with nothing to compare (a run that failed) are counted apart. Reference "
            "rows: the same metric for a static scene, another recording of the event, or the "
            "benchmark's reference solution."
        )
        return header, rows, is_reference, note

    def _labels(self) -> dict[str, str]:
        """Each method's name as the tables show it (``method_label``)."""
        return {
            m["method"]: method_label(m["method"], bool(m.get("privileged")), m.get("network"))
            for m in self.results.get("methods", [])
        }

    def write_tables(self) -> None:
        rows = self.results.get("rows", [])
        methods = {m["method"]: m for m in self.results.get("methods", [])}
        with (self.out / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            if rows:
                # Added at the end (existing columns keep their places): whether the method
                # read hidden data, and how its build was networked (``open``: unrestricted).
                writer = csv.DictWriter(stream, fieldnames=[*rows[0], "privileged", "network"])
                writer.writeheader()
                for row in rows:
                    method = methods.get(row.get("method"), {})
                    writer.writerow(
                        {
                            **row,
                            "privileged": bool(method.get("privileged")),
                            "network": method.get("network") or "",
                        }
                    )
        header, card, references, note = self._scorecard()
        lines = [
            f"# {self.results.get('experiment', 'experiment')}",
            "",
            *(f"> **Warning.** {warning}\n" for warning in self.warnings),
            "## Scorecard",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
            *(
                "| " + " | ".join((f"*{c}*" if ref and c else c) for c in row) + " |"
                for row, ref in zip(card, references, strict=True)
            ),
            "",
            note,
            "",
            "## Methods",
            "",
            "| method | kind | tasks | delivered | reproduction skill |",
            "| --- | --- | --- | --- | --- |",
        ]
        for method in self.results.get("methods", []):
            kind = method["method_kind"] + (" (privileged)" if method.get("privileged") else "")
            if method.get("network") == "open":
                kind += " (open network)"
            lines.append(
                f"| {method['method']} | {kind} | {method['tasks']} | "
                f"{_delivered_rate(method)} | {_md(method['reproduction_skill'])} |"
            )
        lines += [
            "",
            "## Results",
            "",
            "| method | task | trial | seed | headline | reproduction error | reproduction skill "
            "| re-run |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        labels = self._labels()
        for row in rows:
            headline = (
                f"{row['headline_metric']} {_mdv(row['headline'])}"
                if row.get("headline_metric")
                else "—"
            )
            if row.get("infra_error"):
                headline = f"not scored (infrastructure: {row['infra_error']})"
            rerun = row.get("session_consistent")
            rerun = row.get("reproducible") if rerun is None else rerun
            lines.append(
                f"| {labels.get(row['method'], row['method'])} | {row['task']} | {row['trial']} | "
                f"{row['seed']} | "
                f"{headline} | {_mdv(row['reproduction_error'])}"
                f"{_unit(row.get('unit')) if row['reproduction_error'] is not None else ''} | "
                f"{_mdv(row['reproduction_skill'])} | {_mdv(rerun)} |"
            )
        skipped = self.results.get("skipped", [])
        if skipped:
            lines += ["", "Skipped:", *[f"- {item}" for item in skipped]]
        (self.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- html ---------------------------------------------------------------------------------

    def write_index(self) -> None:
        methods = self.results.get("methods", [])
        body = [f"<h1>{html.escape(self.results.get('experiment', 'Experiment'))}</h1>"]
        body += [
            f'<p class="bad"><b>Warning.</b> {html.escape(warning)}</p>'
            for warning in self.warnings
        ]
        body.append(
            f'<p class="muted">{len(self.results.get("tasks", []))} tasks, '
            f"{len(methods)} methods; Real2SimLab {html.escape(str(self.results.get('lab_version')))}. "
            '<a href="report.md">Markdown</a> · <a href="results.csv">CSV</a></p>'
        )
        open_network = [m["method"] for m in methods if m.get("network") == "open"]
        if open_network:
            body.append(
                f'<p class="warn">⚠ Open network: {", ".join(map(html.escape, open_network))} built '
                "twins with unrestricted network access, not the egress allowlist.</p>"
            )
        header, card, references, note = self._scorecard()
        if len(header) > 3:
            body.append(
                '<h2>Scorecard</h2><table class="scorecard"><tr>'
                + "".join(f"<th>{html.escape(h)}</th>" for h in header)
                + "</tr>"
            )
            for row, ref in zip(card, references, strict=True):
                style = ' class="muted"' if ref else ""
                body.append(
                    f"<tr{style}>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
                )
            body.append("</table>")
            body.append(f'<p class="muted">{html.escape(note)}</p>')
        body.append(
            "<h2>Methods</h2><table><tr><th>method</th><th>kind</th><th>tasks</th>"
            "<th>delivered</th><th>reproduction skill</th></tr>"
        )
        for method in methods:
            flags = ""
            if method.get("privileged"):
                flags += '<span class="flag muted">privileged</span>'
            if method.get("network") == "open":
                flags += '<span class="flag warn">open network</span>'
            body.append(
                f"<tr><td>{html.escape(method['method'])}{flags}</td>"
                f"<td>{html.escape(method['method_kind'])}</td><td>{method['tasks']}</td>"
                f"<td>{_delivered_rate(method)}</td>"
                f"<td>{_estimate(method['reproduction_skill'])}</td></tr>"
            )
        body.append("</table>")
        body.append(
            '<p class="muted">Reproduction skill (tasks scored by the Lab\'s metrics): '
            "1 - error / error of a static scene, clipped to [0, 1].</p>"
        )
        by_task: dict[str, dict[str, list[TaskResult]]] = defaultdict(lambda: defaultdict(list))
        for result in self.task_results:
            by_task[result.task_id][result.method].append(result)
        names = [m["method"] for m in methods]
        labels = self._labels()
        body.append(
            "<h2>Tasks</h2><table><tr><th>task</th><th>headline</th>"
            + "".join(f"<th>{html.escape(labels[n])}</th>" for n in names)
            + "</tr>"
        )
        for task_id in sorted(by_task):
            cells = []
            metric = ""
            for name in names:
                items = by_task[task_id].get(name, [])
                values = [
                    r.headline.value
                    for r in items
                    if r.headline is not None and r.headline.value is not None
                ]
                metric = metric or next((r.headline.metric for r in items if r.headline), "")
                failed = any(not r.delivered for r in items)
                text = f"{np.mean(values):.3g}" if values else "—"
                if failed:
                    text += ' <span class="bad">✗</span>'
                cells.append(f"<td>{text}</td>")
            body.append(
                f'<tr><td><a href="tasks/{slug(task_id)}.html">{html.escape(task_id)}</a></td>'
                f"<td>{html.escape(metric)}</td>{''.join(cells)}</tr>"
            )
        body.append('</table><p class="muted">✗: nothing was delivered.</p>')
        skipped = self.results.get("skipped", [])
        if skipped:
            body.append(
                "<h2>Skipped</h2><ul>"
                + "".join(f"<li>{html.escape(s)}</li>" for s in skipped)
                + "</ul>"
            )
        if self.notes:
            body.append(
                "<h2>Notes</h2><ul>"
                + "".join(f"<li>{html.escape(n)}</li>" for n in self.notes)
                + "</ul>"
            )
        (self.out / "index.html").write_text(
            _page(self.results.get("experiment", "report"), "".join(body)), encoding="utf-8"
        )

    def write_task_pages(self) -> None:
        by_task: dict[str, list[TaskResult]] = defaultdict(list)
        for result in self.task_results:
            by_task[result.task_id].append(result)
        (self.out / "tasks").mkdir(parents=True, exist_ok=True)
        for task_id, results in sorted(by_task.items()):
            page = self._task_page(
                task_id, sorted(results, key=lambda r: (r.method, r.trial, r.seed))
            )
            (self.out / "tasks" / f"{slug(task_id)}.html").write_text(page, encoding="utf-8")

    def _task_page(self, task_id: str, results: list[TaskResult]) -> str:
        task = self.tasks.get(task_id)
        # A recording that may not be redistributed (and the overlays drawn on it) stays out
        # of report/, in <run>/local-media/: the pages still play it on this machine, and
        # sharing report/ does not share it.
        shareable = task is not None and task.config.redistributable
        if shareable:
            media, href = self.out / "media" / slug(task_id), f"../media/{slug(task_id)}"
        else:
            media = self.layout.root / LOCAL_MEDIA / slug(task_id)
            href = f"../../{LOCAL_MEDIA}/{slug(task_id)}"
        media.mkdir(parents=True, exist_ok=True)
        body = [f'<p><a href="../index.html">← summary</a></p><h1>{html.escape(task_id)}</h1>']
        if task_id in self.task_warnings:
            body.append(
                f'<p class="bad"><b>Warning.</b> {html.escape(self.task_warnings[task_id])}</p>'
            )
        if task is not None:
            config = task.config
            body.append(
                f"<p>{html.escape(config.title)}. {html.escape(config.description or '')}</p>"
                f'<p class="muted">Source: {html.escape(config.source.dataset)} {html.escape(config.source.version)}; '
                f"license {html.escape(config.license)}; primary metric {html.escape(str(config.scoring.primary))}.</p>"
            )
        observed_video = self._observed_video(task, media)
        fmt = get_format(task.config.delivery.output_format) if task is not None else None
        videos = []
        if observed_video is not None:
            videos.append(
                f'<figure><video controls loop muted src="{href}/{observed_video.name}"></video>'
                "<figcaption>observed</figcaption></figure>"
            )
        for index, result in enumerate(results):
            twin = self._twin_tracks(result, fmt)
            if twin is not None and observed_video is not None:
                overlay = self._overlay(
                    observed_video,
                    twin,
                    media / f"overlay_{result.method}_{result.trial}_{result.seed}.mp4",
                    COLORS[index % len(COLORS)],
                )
                if overlay is not None:
                    videos.append(
                        f'<figure><video controls loop muted src="{href}/{overlay.name}"></video>'
                        f"<figcaption>{html.escape(self._labels().get(result.method, result.method))}"
                        ": its frozen re-run's trajectory, "
                        "projected with its own camera, drawn on the recording</figcaption></figure>"
                    )
        for result in results:  # read (and checked) once: see _twin_tracks
            self._twin_tracks(result, fmt)
        for warning in self.changed_outputs.get(task_id, []):
            body.append(f'<p class="bad"><b>Warning.</b> {html.escape(warning)}</p>')
        if videos:
            body.append('<h2>Reproduction</h2><div class="videos">' + "".join(videos) + "</div>")
            if not shareable:
                body.append(
                    '<p class="muted">This recording may not be redistributed (license: '
                    f"{html.escape(task.config.license if task else 'unknown')}). Its video and "
                    f"the overlays drawn on it are kept in {LOCAL_MEDIA}/ next to report/, not in "
                    "it: they play here, and sharing report/ does not share them.</p>"
                )
        body.append(
            "<table><tr><th>method</th><th>trial</th><th>seed</th><th>delivered</th>"
            "<th>headline</th><th>reproduction error</th><th>skill</th><th>reproducible</th>"
            "<th>matches the session</th></tr>"
        )
        labels = self._labels()
        for result in results:
            summary = result.summary
            failure = (
                f' <span class="bad" title="{html.escape(result.delivery_failure or "")}">✗</span>'
                if not result.delivered
                else ""
            )
            headline = (
                f"{result.headline.metric} {_fmt(result.headline.value)}"
                if result.headline
                else "—"
            )
            if result.infra_error is not None:
                headline = f"not scored (infrastructure: {result.infra_error})"
            body.append(
                f"<tr><td>{html.escape(labels.get(result.method, result.method))}{failure}</td><td>{result.trial}</td><td>{result.seed}</td>"
                f"<td>{_fmt(result.delivered)}</td><td>{html.escape(headline)}</td>"
                f"<td>{_fmt(summary.reproduction_error)}{html.escape(_unit(result.metric_unit))}</td>"
                f"<td>{_fmt(summary.reproduction_skill)}</td><td>{_fmt(result.reproduction.reproducible)}</td>"
                f"<td>{_fmt(result.reproduction.session_consistent)}</td></tr>"
            )
        body.append("</table>")
        for result in results:
            if not result.delivered:
                body.append(
                    f"<p><b>{html.escape(result.method)}</b> delivered no twin: "
                    f"{html.escape(result.delivery_failure or 'unknown reason')}</p>"
                )
            if result.native is not None:
                text = json.dumps(result.native, indent=1, ensure_ascii=False)
                body.append(
                    f"<details><summary>{html.escape(result.method)}: the official scorer's report"
                    f"</summary><pre>{html.escape(text[:20000])}</pre></details>"
                )
        if task is not None and fmt is not None and fmt.provides_tracks:
            directory = task.reference_dir()
            if directory.is_dir():
                try:
                    truth = fmt.to_tracks(directory)
                except (OSError, ValueError, R2SError):
                    truth = None
                if truth is not None:
                    body.append("<h2>Trajectories</h2>" + self._plots(truth, results, fmt))
        body.append(self._agents_section(task_id, results))
        return _page(task_id, "".join(body))

    def _plots(self, truth: Tracks, results: list[TaskResult], fmt: Any) -> str:
        plots = []
        for name in truth.objects:
            for axis, label in ((0, "x"), (2, "z")):
                series = [
                    Series("reference", truth.times(), truth.positions(name)[:, axis], "#555555")
                ]
                for index, result in enumerate(results):
                    twin = self._twin_tracks(result, fmt)
                    if twin is not None and name in twin.objects:
                        series.append(
                            Series(
                                result.method,
                                twin.times(),
                                twin.positions(name)[:, axis],
                                COLORS[index % len(COLORS)],
                            )
                        )
                plots.append(
                    line_plot(
                        series, title=f"{name}.{label}", x_label="time (s)", y_label=f"{label} (m)"
                    )
                )
        return '<div class="plots">' + "".join(plots) + "</div>"

    def _agents_section(self, task_id: str, results: list[TaskResult]) -> str:
        parts: list[str] = []
        for result in results:
            build = self.layout.build_dir(task_id, result.method, result.trial)
            record, damage = _agent_record(build / "work" / "agent" / "agent_record.json")
            # The trajectory its record names (mini-swe-agent's is one JSON document), which it
            # names once the Lab removed the injected credentials from the agent's files: not
            # yet (the build stopped before), or they are still in it.
            trajectory = recorded_file(build / "work", record.get("trajectory"))
            if trajectory is None and not damage:
                continue
            events = read_trajectory(trajectory) if trajectory is not None else []
            items = []
            for event in events[:500]:
                kind = str(event.get("type") or event.get("role") or "event")
                text = json.dumps(event, ensure_ascii=False)
                items.append(
                    f"<details><summary>{html.escape(kind)}</summary><pre>{html.escape(text[:6000])}</pre></details>"
                )
            usage = record.get("usage") or {}
            parts.append(
                f"<h3>{html.escape(result.method)}, trial {result.trial}: agent trajectory (redacted)</h3>"
                f'<p class="muted">{html.escape(str(record.get("agent")))} {html.escape(str(record.get("agent_version")))}, '
                f"model {html.escape(str(record.get('model')))}, effort {html.escape(str(record.get('reasoning_effort')))}, "
                f"{_fmt(record.get('duration_s'), 0)} s, exit {html.escape(str(record.get('exit_status')))}, "
                f"tokens in/out {usage.get('input_tokens', '—')}/{usage.get('output_tokens', '—')}, "
                f"cost {_cost(record.get('cost_usd'))}, "
                f"network {html.escape(str((record.get('network') or {}).get('mode')))}, "
                f"{len(events)} events, {record.get('redactions', 0)} redactions.</p>"
                + (f'<p class="warn">{html.escape(damage)}</p>' if damage else "")
                + "".join(items)
            )
        return ("<h2>Agents</h2>" + "".join(parts)) if parts else ""

    # --- media -------------------------------------------------------------------------------

    def _observed_video(self, task: Task | None, media: Path) -> Path | None:
        if task is None or not task.has_capture():
            return None
        capture = task.capture()
        view = next(iter(capture.views.values()))
        source = task.visible_dir / view.rgb.path
        if not source.is_file():
            return None
        target = media / "observed.mp4"
        shutil.copyfile(source, target)
        return target

    def _twin_tracks(self, result: TaskResult, fmt: Any) -> Tracks | None:
        """The trajectory of the frozen re-run that was scored, or ``None``: also when its
        output has changed since (it would be drawn next to a score it did not produce), or
        when nothing shows that it has not (no run record, or a record of a successful run
        without the fingerprint that is always recorded for its output)."""
        if fmt is None or not fmt.provides_tracks or not result.delivered:
            return None
        run_dir = self.layout.run_dir(
            result.task_id, result.method, result.trial, result.seed, "run"
        )
        if run_dir in self._outputs:
            return self._outputs[run_dir]
        self._outputs[run_dir] = None
        output = run_dir / "output"
        if not output.is_dir():
            return None
        from real2simlab.runner.run import RunStatus, read_record
        from real2simlab.twin.fingerprint import fingerprint

        try:
            record = read_record(run_dir)
            expected = record.output_fingerprint if record is not None else None
            current = fingerprint(output)
        except Exception:  # an unreadable record or output: nothing to draw
            return None
        if expected is not None and current != expected:
            self.changed_outputs[result.task_id].append(
                f"The output of {result.method} (trial {result.trial}, seed {result.seed}) has "
                "changed since it was scored: its trajectory is not drawn, since it is not the "
                "one that produced the score. Run the evaluation again to score what is there."
            )
            return None
        if expected is None and (record is None or record.status is RunStatus.SUCCEEDED):
            self.changed_outputs[result.task_id].append(
                f"The output of {result.method} (trial {result.trial}, seed {result.seed}) "
                "cannot be checked against its score (its run record is missing, or has no "
                "output_fingerprint): its trajectory is not drawn. Run the evaluation again to "
                "score what is there."
            )
            return None
        try:
            tracks = fmt.to_tracks(output)
        except Exception:  # the program's output: whatever it holds, the page is still written
            return None
        self._outputs[run_dir] = tracks
        return tracks

    def _overlay(self, video: Path, twin: Tracks, target: Path, color: str) -> Path | None:
        from real2simlab.media import MediaError, read_frames, write_video

        if twin.camera is None or twin.camera.width <= 0 or twin.camera.height <= 0:
            return None
        try:
            frames = read_frames(video).copy()
        except (MediaError, R2SError) as error:
            self.notes.append(f"no overlay video: {error}")
            return None
        rgb = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
        height, width = frames.shape[1:3]
        scale = np.array([width / twin.camera.width, height / twin.camera.height])
        yy, xx = np.mgrid[0:height, 0:width]
        for name in twin.objects:
            uv = twin.camera.project(twin.positions(name)) * scale
            for index in range(min(len(frames), len(uv))):
                u, v = uv[index]
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                ring = np.abs(np.hypot(xx - u, yy - v) - 5) < 1.3
                frames[index][ring] = rgb
        try:
            return write_video(frames, target, fps=twin.fps)
        except (MediaError, R2SError) as error:
            self.notes.append(f"no overlay video: {error}")
            return None

    def build(self) -> Path:
        # Everything is written afresh: a recording whose task may no longer be redistributed,
        # an overlay of an earlier result, or the page of a task whose results are no longer
        # counted (an earlier configuration's scores) must not stay from an earlier build.
        from real2simlab._fs import force_remove

        for written in (self.out / "tasks", self.out / "media", self.layout.root / LOCAL_MEDIA):
            problem = force_remove(written)
            if problem is not None:
                raise SetupError(problem, fix=f"remove {written} yourself, then build the report")
        self.out.mkdir(parents=True, exist_ok=True)
        self.write_tables()
        self.write_task_pages()
        self.write_index()
        return self.out / "index.html"


def _delivered_rate(method: dict[str, Any]) -> str:
    """A method's delivered rate, or "—" when the Lab could score none of its results."""
    if method.get("results", 0) - method.get("infra_errors", 0) <= 0:
        return "—"
    return f"{method['delivered_rate']:.0%}"


def _md(data: dict[str, Any] | None, digits: int = 3) -> str:
    if not data or data.get("mean") is None:
        return "—"

    def num(value: float) -> str:
        return f"{value:.{digits}g}" if abs(value) >= 10 else f"{value:.{digits}f}"

    if data.get("ci_low") is None or data.get("ci_low") == data.get("ci_high"):
        return num(data["mean"])
    return f"{num(data['mean'])} [{num(data['ci_low'])}, {num(data['ci_high'])}]"


def _unit(unit: str | None) -> str:
    """A unit to print after a value: none for dimensionless metrics (unit ``1``)."""
    return f" {unit}" if unit and unit != "1" else ""


def _mdv(value: Any) -> str:
    return "—" if value is None else (f"{value:.4g}" if isinstance(value, float) else str(value))


def _cost(value: Any) -> str:
    """What an agent run cost, as its record says (``cost_usd``): unknown when it says none."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"${value:g}"
    return "unknown"


def _agent_record(path: Path) -> tuple[dict[str, Any], str | None]:
    """The agent record at ``path`` as JSON (``{}`` when there is none, or when it is not a
    regular file: it is in the agent's log directory, so a link is never followed), and why it
    is not shown when it is damaged."""
    try:
        data = read_regular(path)
        if data is None:
            return {}, None
        return AgentRecord.model_validate_json(data).to_json(), None
    except (OSError, ValueError) as error:
        first = str(error).splitlines()[0]
        return {}, f"Its agent record {path} is damaged ({first}): the details are not shown."


def write_report(run_dir: Path) -> Path:
    """Write ``<run_dir>/report/`` and return the path of ``index.html``."""
    return ReportBuilder(run_dir).build()
