"""Aggregating task results across tasks, seeds and trials, with bootstrap confidence intervals.

The bootstrap is two-level: resample tasks with replacement, then resample each sampled task's
results (its trials and seeds) with replacement, and take the mean. Percentile intervals at
``confidence`` from ``resamples`` draws of a generator seeded with ``seed``, so the same
results always give the same intervals.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from real2simlab import __version__
from real2simlab.scoring.model import (
    BenchmarkScore,
    Estimate,
    Headline,
    MethodSummary,
    Results,
    TaskResult,
)
from real2simlab.scoring.reproduction import PROTOCOL_VERSION

RESAMPLES = 2000
CONFIDENCE = 0.95
T = TypeVar("T")


def bootstrap(
    groups: Sequence[Sequence[float]],
    *,
    resamples: int = RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 0,
) -> Estimate:
    """Mean over groups (each group: one task's values) with a two-level bootstrap CI."""
    groups = [list(group) for group in groups if len(group)]
    if not groups:
        return Estimate(mean=None, n=0)
    task_means = np.array([float(np.mean(np.asarray(g, dtype=np.float64))) for g in groups])
    mean = float(task_means.mean())
    count = sum(len(group) for group in groups)
    if len(groups) == 1 and len(groups[0]) == 1:
        return Estimate(mean=mean, ci_low=mean, ci_high=mean, n=1)
    rng = np.random.default_rng(seed)
    arrays: list[NDArray[np.float64]] = [np.asarray(group, dtype=np.float64) for group in groups]
    draws = np.empty(resamples)
    for index in range(resamples):
        means: list[float] = []
        for pick in rng.integers(0, len(arrays), size=len(arrays)):
            sample = arrays[int(pick)]
            means.append(float(sample[rng.integers(0, len(sample), size=len(sample))].mean()))
        draws[index] = float(np.mean(np.asarray(means)))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return Estimate(mean=mean, ci_low=float(low), ci_high=float(high), n=count)


def bootstrap_statistic(
    groups: Sequence[Sequence[T]],
    statistic: Callable[[list[T]], float | None],
    *,
    resamples: int = RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 0,
) -> Estimate:
    """``statistic`` of every item, with the same two-level bootstrap as :func:`bootstrap`
    (resample groups, then items within each), for benchmarks that do not simply average."""
    groups = [list(group) for group in groups if len(group)]
    items = [item for group in groups for item in group]
    value = statistic(items) if items else None
    if value is None:
        return Estimate(mean=None, n=len(items))
    if len(items) == 1:
        return Estimate(mean=value, ci_low=value, ci_high=value, n=1)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(resamples):
        sample: list[T] = []
        for pick in rng.integers(0, len(groups), size=len(groups)):
            group = groups[int(pick)]
            sample.extend(group[int(i)] for i in rng.integers(0, len(group), size=len(group)))
        drawn = statistic(sample)
        if drawn is not None:
            draws.append(drawn)
    if not draws:
        return Estimate(mean=value, n=len(items))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(np.asarray(draws), [alpha, 1 - alpha])
    return Estimate(mean=value, ci_low=float(low), ci_high=float(high), n=len(items))


def _groups(
    results: Sequence[TaskResult], value: Callable[[TaskResult], float | None]
) -> list[list[float]]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for result in results:
        item = value(result)
        if item is not None:
            by_task[result.task_id].append(item)
    return [by_task[task] for task in sorted(by_task)]


def _rerun_agreed(result: TaskResult) -> bool | None:
    """Whether the frozen re-run agreed with the agent's session output or, without one, with
    a second re-run; ``None`` when there was nothing to compare."""
    consistent = result.reproduction.session_consistent
    return result.reproduction.reproducible if consistent is None else consistent


def _rate(values: Sequence[bool | None]) -> float | None:
    known = [value for value in values if value is not None]
    return float(np.mean(known)) if known else None


def _official_aggregate(
    scorer: str, results: Sequence[TaskResult]
) -> Callable[[list[Mapping[str, Any]]], dict[str, float | None] | None] | None:
    """The scorer's own way of combining tasks, if it has one."""
    from real2simlab import plugins
    from real2simlab.errors import R2SError

    try:
        factory = plugins.load("scorer", scorer)
    except (R2SError, ImportError):
        return None
    combine = getattr(factory, "aggregate", None)
    if combine is None or not results or combine([r.native or {} for r in results[:1]]) is None:
        return None
    return combine


def scored_results(results: Sequence[TaskResult]) -> list[TaskResult]:
    """The results that are scores: all but those the Lab could not score (``infra_error``)."""
    return [result for result in results if result.infra_error is None]


def benchmark_scores(
    results: Sequence[TaskResult], *, resamples: int = RESAMPLES, seed: int = 0
) -> tuple[BenchmarkScore, ...]:
    """The scorecard of one method: per benchmark and headline metric, the metric over the
    tasks that report it, combined the benchmark's way (the mean, unless its scorer defines
    another aggregate). Tasks of one benchmark normally share one headline metric; tasks that
    report different ones (custom tasks under one name) get one entry per metric. Results the
    Lab could not score (``infra_error``) are counted in ``results`` and ``infra_errors`` and
    left out of everything else."""
    entries: defaultdict[tuple[str, str, str], list[TaskResult]] = defaultdict(list)
    pending: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    headlines: dict[tuple[str, str, str], Headline] = {}
    for result in results:
        if result.headline is not None:
            key = (result.benchmark, result.headline.metric, result.headline.scorer or "")
            headlines.setdefault(key, result.headline)
            if result.infra_error is None:
                entries[key].append(result)
            else:
                pending[key] += 1
                entries.setdefault(key, [])
    scores: list[BenchmarkScore] = []
    for key in sorted(entries):
        benchmark, same = key[0], entries[key]
        if not same:  # every result of this benchmark is one the Lab could not score
            headline = headlines[key]
            scores.append(
                BenchmarkScore(
                    benchmark=benchmark,
                    metric=headline.metric,
                    unit=headline.unit,
                    higher_is_better=headline.higher_is_better,
                    range=headline.range,
                    value=Estimate(mean=None, n=0),
                    scored=0,
                    results=pending[key],
                    infra_errors=pending[key],
                    delivered=0,
                )
            )
            continue
        first = same[0].headline
        assert first is not None
        combine = _official_aggregate(first.scorer, same) if first.scorer else None
        if combine is not None:
            by_task: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for r in same:
                by_task[r.task_id].append(r.native or {})
            groups = [by_task[task] for task in sorted(by_task)]
            value = bootstrap_statistic(
                groups,
                lambda reports, combine=combine: (combine(reports) or {}).get("value"),
                resamples=resamples,
                seed=seed,
            )
            combined = combine([report for group in groups for report in group]) or {}
            references = {
                name: float(v) for name, v in combined.items() if name != "value" and v is not None
            }
        else:
            value = bootstrap(
                _groups(same, lambda r: r.headline.value if r.headline else None),
                resamples=resamples,
                seed=seed,
            )
            names = sorted({n for r in same if r.headline for n in r.headline.references})
            references = {
                name: float(
                    np.mean(
                        [
                            r.headline.references[name]
                            for r in same
                            if r.headline is not None and name in r.headline.references
                        ]
                    )
                )
                for name in names
            }
        compared = [agreed for agreed in map(_rerun_agreed, same) if agreed is not None]
        scores.append(
            BenchmarkScore(
                benchmark=benchmark,
                metric=first.metric,
                unit=first.unit,
                higher_is_better=first.higher_is_better,
                range=first.range,
                value=value,
                scored=sum(1 for r in same if r.headline and r.headline.value is not None),
                results=len(same) + pending[key],
                references=references,
                reproducible_rate=_rate([r.reproduction.reproducible for r in same]),
                session_consistent_rate=_rate([r.reproduction.session_consistent for r in same]),
                rerun_compared=len(compared),
                rerun_agreed=sum(compared),
                infra_errors=pending[key],
                delivered=sum(1 for r in same if r.delivered),
            )
        )
    return tuple(scores)


def summarize_method(
    method: str, results: Sequence[TaskResult], *, resamples: int = RESAMPLES, seed: int = 0
) -> MethodSummary:
    first = results[0]
    scored = scored_results(results)
    return MethodSummary(
        method=method,
        method_kind=first.method_kind,
        privileged=any(r.privileged for r in results),
        tasks=len({r.task_id for r in results}),
        results=len(results),
        reproduction_skill=bootstrap(
            _groups(scored, lambda r: r.summary.reproduction_skill), resamples=resamples, seed=seed
        ),
        delivered_rate=float(np.mean([r.delivered for r in scored])) if scored else 0.0,
        benchmarks=benchmark_scores(results, resamples=resamples, seed=seed),
        infra_errors=len(results) - len(scored),
    )


def row(result: TaskResult) -> dict[str, Any]:
    """One table row for a task result."""
    summary = result.summary
    headline = result.headline
    return {
        "method": result.method,
        "benchmark": result.benchmark,
        "task": result.task_id,
        "trial": result.trial,
        "seed": result.seed,
        "delivered": result.delivered,
        "headline_metric": headline.metric if headline else None,
        "headline": headline.value if headline else None,
        "reproduction_error": summary.reproduction_error,
        "reproduction_skill": summary.reproduction_skill,
        "reproducible": result.reproduction.reproducible,
        "session_consistent": result.reproduction.session_consistent,
        "metric": result.primary_metric,
        "unit": result.metric_unit,
        "failure": result.delivery_failure,
        "infra_error": result.infra_error,
    }


def aggregate(
    experiment: str,
    results: Sequence[TaskResult],
    *,
    resamples: int = RESAMPLES,
    seed: int = 0,
    network: dict[str, str] | None = None,
) -> Results:
    """The experiment's aggregate: one summary per method, plus one row per result."""
    by_method: defaultdict[str, list[TaskResult]] = defaultdict(list)
    for result in results:
        by_method[result.method].append(result)
    summaries: list[MethodSummary] = []
    for method in by_method:
        summary = summarize_method(method, by_method[method], resamples=resamples, seed=seed)
        if network and method in network:
            summary = summary.model_copy(update={"network": network[method]})
        summaries.append(summary)
    return Results(
        experiment=experiment,
        created_at=datetime.now(UTC),
        lab_version=__version__,
        protocol_version=PROTOCOL_VERSION,
        confidence=CONFIDENCE,
        bootstrap_resamples=resamples,
        methods=tuple(summaries),
        tasks=tuple(sorted({r.task_id for r in results})),
        rows=tuple(
            row(r) for r in sorted(results, key=lambda r: (r.method, r.task_id, r.trial, r.seed))
        ),
    )
