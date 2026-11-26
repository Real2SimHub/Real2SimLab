"""``r2s task``, ``r2s bench``, ``r2s capture``: authoring and inspecting tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from real2simlab.cli.common import TASKS_INTERRUPTED, add_parser, emit, note_other_tasks, table


def register_task(subparsers: Any) -> None:
    parser = add_parser(
        subparsers,
        "task",
        "create, validate and inspect task directories (r2s-task/1.0)",
        "Create, validate and inspect task directories (r2s-task/1.0).\n\n"
        "A task directory holds task.toml, visible/ (what a method sees) and hidden/ (the\n"
        "reference and what scoring needs). See the tutorial 'Write tasks for your benchmark'.",
    )
    sub = parser.add_subparsers(dest="task_command", metavar="ACTION", required=True)

    new = add_parser(sub, "new", "create a task skeleton to fill in")
    new.add_argument("path", type=Path, help="directory to create (must not exist or be empty)")
    new.add_argument("--id", required=True, help="task id, e.g. mybench/drop_01")
    new.add_argument("--title", default="TODO: title", help="one-line title")
    new.add_argument(
        "--video", type=Path, help="make visible/ from this video (r2s capture from-video)"
    )
    new.set_defaults(handler=_new)

    validate = add_parser(
        sub,
        "validate",
        "check task directories against r2s-task/1.0",
        "Check task directories: spec, hidden-data leaks into visible/, missing files,\n"
        "references valid in the output format, scoring.\n"
        "Exit status 1 if any task has errors.",
    )
    validate.add_argument(
        "paths", type=Path, nargs="+", help="task directories, or directories of tasks"
    )
    validate.add_argument("--json", action="store_true", help="print the reports as JSON")
    validate.add_argument(
        "--no-references", action="store_true", help="skip checking reference outputs"
    )
    validate.set_defaults(handler=_validate)

    show = add_parser(sub, "show", "show a task's metadata or the task statement agents get")
    show.add_argument("path", type=Path)
    show.add_argument(
        "--statement", action="store_true", help="print the task statement agents get"
    )
    show.add_argument("--json", action="store_true")
    show.set_defaults(handler=_show)

    listing = add_parser(sub, "list", "list the tasks under directories")
    listing.add_argument("paths", type=Path, nargs="*", default=[Path(".")])
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(handler=_list)


def _new(args: argparse.Namespace) -> int:
    from real2simlab.task.scaffold import new_task

    print(new_task(args.path, task_id=args.id, title=args.title, video=args.video))
    return 0


def _validate(args: argparse.Namespace) -> int:
    from real2simlab.task import find_tasks, validate_task

    roots = [root for path in args.paths for root in (find_tasks(path) or [path])]
    reports = [validate_task(root, check_references=not args.no_references) for root in roots]
    if args.json:
        emit([report.to_json() for report in reports], True, "")
    else:
        for report in reports:
            status = "ok" if report.ok else f"{len(report.errors)} error(s)"
            warnings = f", {len(report.warnings)} warning(s)" if report.warnings else ""
            print(f"{report.task_id or report.root}: {status}{warnings}")
            for issue in report.issues:
                print(f"  {issue}")
    return 0 if all(report.ok for report in reports) else 1


def _show(args: argparse.Namespace) -> int:
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    task = load_task(args.path)
    if args.statement:
        print(task_statement(task))
        return 0
    data = task.describe()
    text = "\n".join(f"{key}: {value}" for key, value in data.items())
    emit(data, args.json, text)
    return 0


def _list(args: argparse.Namespace) -> int:
    from real2simlab.errors import InputNotFoundError
    from real2simlab.task import find_tasks, load_task

    rows: list[list[Any]] = []
    data: list[dict[str, Any]] = []
    for path in args.paths:
        if not path.exists():
            raise InputNotFoundError(f"{path} does not exist", fix="give task directories")
        for root in find_tasks(path):
            task = load_task(root)
            info = task.describe()
            data.append(info)
            scorer = info["native_scorer"] or "-"
            rows.append([task.id, info["protocol"], scorer, info["output_format"], root])
    emit(data, args.json, table(rows, ["task", "protocol", "scorer", "format", "path"]))
    return 0


def register_bench(subparsers: Any) -> None:
    parser = add_parser(subparsers, "bench", "list benchmark adapters and build their tasks")
    sub = parser.add_subparsers(dest="bench_command", metavar="ACTION", required=True)
    listing = add_parser(sub, "list", "list the registered benchmark adapters")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(handler=_bench_list)
    build = add_parser(
        sub,
        "build",
        "build a benchmark's task directories",
        "Build a benchmark's task directories.\n\nExample: r2s bench build synthetic --out tasks/ "
        "--select bouncing_ball --select pendulum --tier calibrated",
    )
    build.add_argument("name", help="a benchmark from `r2s bench list`")
    build.add_argument("--out", type=Path, required=True, help="directory to write tasks into")
    build.add_argument("--select", action="append", help="build only these (repeatable)")
    build.add_argument("--tier", action="append", dest="tiers", help="visibility tiers (synthetic)")
    build.add_argument("--option", action="append", default=[], help="adapter option key=value")
    build.add_argument(
        "--split",
        default="eval",
        help="eval (default) or train, when the benchmark releases training data",
    )
    build.add_argument("--overwrite", action="store_true", help="replace existing task directories")
    build.set_defaults(
        handler=_bench_build,
        on_interrupt=TASKS_INTERRUPTED,
    )


def _bench_list(args: argparse.Namespace) -> int:
    from real2simlab.benchmarks.base import benchmark_names, get_benchmark

    infos = [get_benchmark(name).info() for name in benchmark_names()]
    rows = [
        [i.name, i.status, ", ".join(i.protocols), ", ".join(i.splits), i.license, i.summary]
        for i in infos
    ]
    emit(
        [i.__dict__ for i in infos],
        args.json,
        table(rows, ["benchmark", "status", "protocols", "splits", "license", "summary"]),
    )
    return 0


def _bench_build(args: argparse.Namespace) -> int:
    from real2simlab.benchmarks.base import get_benchmark
    from real2simlab.cli.common import parse_options
    from real2simlab.errors import UsageError

    adapter = get_benchmark(args.name)
    options = parse_options(args.option)
    for name, flag in (("select", "--select"), ("out_dir", "--out")):
        if name in options:
            raise UsageError(f"{name} is not an --option; give it with {flag}")
    if args.tiers:
        options["tiers"] = args.tiers
    if args.overwrite:
        options["overwrite"] = True
    paths = adapter.build_split(args.out, args.split, select=args.select, **options)
    for path in paths:
        print(path)
    print(f"built {len(paths)} task(s) in {args.out}")
    note_other_tasks(args.out, paths)
    return 0


def register_capture(subparsers: Any) -> None:
    parser = add_parser(subparsers, "capture", "create captures (r2s-capture/1.0)")
    sub = parser.add_subparsers(dest="capture_command", metavar="ACTION", required=True)
    video = add_parser(
        sub,
        "from-video",
        "make a single-view capture from a video file",
        "Make a single-view capture from an ordinary video file.\n\n"
        "Example: r2s capture from-video drop.mp4 visible/ --capture-id lab/drop_01 \\\n"
        "           --fov 62 --object ball=0.067 --text 'A tennis ball is dropped.'",
    )
    video.add_argument("video", type=Path)
    video.add_argument("out", type=Path, help="capture directory to create")
    video.add_argument("--capture-id", required=True)
    group = video.add_mutually_exclusive_group()
    group.add_argument("--intrinsics", help="fx,fy,cx,cy in pixels (measured)")
    group.add_argument(
        "--fov",
        type=_degrees_of_view,
        help="horizontal field of view in degrees, above 0 and below 180 (estimated intrinsics)",
    )
    video.add_argument(
        "--extrinsics", type=Path, help="JSON file with a 4x4 camera-to-world matrix"
    )
    video.add_argument(
        "--object", action="append", default=[], help="NAME=SIZE_M: a known object size (scale)"
    )
    video.add_argument("--text", help="what the video shows")
    video.set_defaults(handler=_from_video)


def _degrees_of_view(text: str) -> float:
    """An argparse type: a horizontal field of view, in degrees, above 0 and below 180."""
    import math

    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of degrees, not {text!r}") from None
    if not (math.isfinite(value) and 0 < value < 180):
        raise argparse.ArgumentTypeError(f"must be above 0 and below 180 degrees, not {text}")
    return value


def _from_video(args: argparse.Namespace) -> int:
    from real2simlab.capture.from_video import capture_from_video, intrinsics_from_fov
    from real2simlab.capture.provenance import Provenance
    from real2simlab.errors import UsageError
    from real2simlab.media import probe_video

    intrinsics = None
    provenance = Provenance.MEASURED
    if args.intrinsics is not None:
        try:
            fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
        except ValueError:
            raise UsageError(
                f"--intrinsics takes fx,fy,cx,cy (four numbers, pixels), not {args.intrinsics!r}"
            ) from None
        intrinsics = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    elif args.fov is not None:
        info = probe_video(args.video)
        intrinsics = intrinsics_from_fov(info.width, info.height, args.fov)
        provenance = Provenance.ESTIMATED
    extrinsics = None
    if args.extrinsics is not None:
        try:
            extrinsics = json.loads(args.extrinsics.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UsageError(
                f"--extrinsics {args.extrinsics}: not a JSON 4x4 matrix ({error})"
            ) from None
        if not isinstance(extrinsics, list):
            raise UsageError(
                f"--extrinsics {args.extrinsics}: holds a JSON {type(extrinsics).__name__}, "
                "not a 4x4 matrix (a list of four rows of four numbers, camera-to-world)"
            )
    sizes: dict[str, float] = {}
    for item in args.object:
        name, _, size = item.partition("=")
        try:
            sizes[name] = float(size)
        except ValueError:
            raise UsageError(
                f"--object takes NAME=SIZE_M (a size in metres), not {item!r}"
            ) from None
    capture = capture_from_video(
        args.video,
        args.out,
        capture_id=args.capture_id,
        intrinsics=intrinsics,
        intrinsics_provenance=provenance,
        extrinsics=extrinsics,
        object_sizes=sizes,
        scene_description=args.text,
    )
    view = next(iter(capture.views.values()))
    print(f"wrote {args.out}/capture.json: {view.rgb.num_frames} frames at {view.rgb.fps:g} fps")
    return 0
