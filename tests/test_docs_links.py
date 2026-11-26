"""Every relative link in the documentation, the READMEs and the GitHub pages reaches a file,
and so does every link into this repository (``.../blob/main/<path>``); the commands the
pages show run in the order a reader follows them.


``mkdocs build --strict`` checks links between pages, but not links to directories (which
neither GitHub nor the site can show) or to files outside the site.

GitHub also shows the pages in ``.github/`` (the security policy) on the repository's front
page, where it resolves a relative link from the repository root, not from ``.github/``: no
relative link there works in both places, so those pages link with absolute URLs. The
contributing guide sits at the root, where relative links work in both places.
"""

from __future__ import annotations

import ast
import importlib.metadata
import os
import re
import shlex
import shutil
from collections.abc import Iterable
from pathlib import Path

import pytest

from real2simlab._project import REPOSITORY

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(
    r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)"  # [text](target "title"), ![alt](target)
    r"|(?:href|src|srcset)=\"([^\"]+)\""  # HTML attributes
)
FENCE = re.compile(r"^```.*?^```", re.S | re.M)
OWN = re.compile(re.escape(REPOSITORY) + r"/(blob|tree)/main/([^#?]+)")


def documents() -> list[Path]:
    pages = sorted((ROOT / "docs").rglob("*.md"))
    return [*pages, *sorted(ROOT.glob("*.md")), *sorted((ROOT / ".github").glob("*.md"))]


def link_targets(page: Path) -> list[str]:
    """Each link target of ``page`` as written, outside code blocks."""
    text = FENCE.sub("", page.read_text(encoding="utf-8"))
    return [next(group for group in match.groups() if group) for match in LINK.finditer(text)]


def relative_path(page: Path, target: str) -> Path | None:
    """The path a relative link points to, or None for a URL, an anchor or a template."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(("#", "{")):
        return None
    path = target.split("#", 1)[0].split("?", 1)[0]
    return page.parent / path if path else None


def relative_links(page: Path) -> list[tuple[str, Path]]:
    """Each relative link of ``page``: the target as written and the path it points to."""
    return [
        (target, resolved)
        for target in link_targets(page)
        if (resolved := relative_path(page, target)) is not None
    ]


def broken_links(page: Path, root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    for target in link_targets(page):
        where = f"{page.relative_to(root).as_posix()}: {target}"
        own = OWN.match(target)
        if own:  # a link into this repository: the file (blob) or directory (tree) exists
            resolved = root / own.group(2)
            if not (resolved.is_file() if own.group(1) == "blob" else resolved.is_dir()):
                problems.append(f"{where} (missing)")
            continue
        resolved = relative_path(page, target)
        if resolved is None:
            continue
        if page.parent.name == ".github":
            problems.append(f"{where} (relative; GitHub's front page resolves it from the root)")
            continue
        if not resolved.is_file():
            kind = "a directory" if resolved.is_dir() else "missing"
            problems.append(f"{where} ({kind})")
    return problems


def test_relative_links_reach_files() -> None:
    problems = [problem for page in documents() for problem in broken_links(page)]
    assert not problems, "\n".join(problems)


def test_the_hugging_face_login_named_is_one_the_install_has() -> None:
    """The install allows huggingface_hub 1.0 and later, which have no ``huggingface-cli``: a
    page (or a message) that asks for a Hugging Face login names ``hf auth login``."""
    texts = [*documents(), *sorted((ROOT / "real2simlab").rglob("*.py"))]
    stale = [
        str(path.relative_to(ROOT))
        for path in texts
        if "huggingface-cli login" in (text := path.read_text(encoding="utf-8"))
        and "hf auth login" not in text
    ]
    assert not stale, stale


def test_the_check_catches_directory_and_missing_links(tmp_path: Path) -> None:
    page = tmp_path / "docs" / "page.md"
    (tmp_path / "docs" / "schemas").mkdir(parents=True)
    page.write_text("[a](schemas/) [b](gone.md) [c](https://x.org/) [d](#here) [e](page.md)\n")
    assert broken_links(page, tmp_path) == [
        "docs/page.md: schemas/ (a directory)",
        "docs/page.md: gone.md (missing)",
    ]


def test_the_check_catches_github_page_and_own_repository_links(tmp_path: Path) -> None:
    page = tmp_path / ".github" / "SECURITY.md"
    page.parent.mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("")
    blob, tree = f"{REPOSITORY}/blob/main", f"{REPOSITORY}/tree/main"
    page.write_text(
        f"[a](../docs/a.md) [b]({blob}/docs/a.md#x) [c]({blob}/docs/gone.md)"
        f" [d]({tree}/docs) [e]({blob}/docs) [f](#here)\n"
    )
    assert broken_links(page, tmp_path) == [
        ".github/SECURITY.md: ../docs/a.md (relative; GitHub's front page resolves it from"
        " the root)",
        f".github/SECURITY.md: {blob}/docs/gone.md (missing)",
        f".github/SECURITY.md: {blob}/docs (missing)",
    ]


WRITERS = {("bench", "build"), ("train", "tasks"), ("train", "export"), ("train", "fit")}
"""Commands that never write over tasks, a dataset or a checkpoint an earlier command wrote to
their ``--out``, unless given ``--overwrite``."""
QUICKSTART = ROOT / "docs" / "quickstart.md"
TRAINING = ROOT / "docs" / "guides" / "training.md"
TRAINING_EXAMPLE = "Example: a method that learns"


def bash_blocks(page: Path) -> list[tuple[str, str]]:
    """The page's ```bash blocks in order, each with the heading it is under."""
    heading, language, lines = "", None, []
    blocks: list[tuple[str, str]] = []
    for line in page.read_text(encoding="utf-8").splitlines():
        if language is None and line.startswith("```"):
            language, lines = line[3:].strip(), []
        elif language is not None and line.startswith("```"):
            if language == "bash":
                blocks.append((heading, "\n".join(lines)))
            language = None
        elif language is not None:
            lines.append(line)
        elif line.startswith("#"):
            heading = line.lstrip("#").strip()
    return blocks


def r2s_commands(page: Path) -> list[tuple[str, list[str]]]:
    """The ``r2s`` commands of the page's bash blocks in order, as their words (continued lines
    joined, comments left out), each with the heading it is under."""
    return [
        (heading, shlex.split(line, comments=True))
        for heading, block in bash_blocks(page)
        for line in block.replace("\\\n", " ").splitlines()
        if line.startswith("r2s ")
    ]


def written_again(commands: Iterable[list[str]]) -> list[str]:
    """The commands that, run in this order in one directory, stop because an earlier one
    wrote their ``--out``."""
    written: set[str] = set()
    stopped: list[str] = []
    for words in commands:
        if tuple(words[1:3]) not in WRITERS or "--out" not in words:
            continue
        out = os.path.normpath(words[words.index("--out") + 1])
        if out in written and "--overwrite" not in words:
            stopped.append(shlex.join(words))
        written.add(out)
    return stopped


def test_no_page_writes_an_output_twice() -> None:
    """Each page's commands run in order in one directory: the training guide's worked example
    stopped on the `train/tasks` its first section had written."""
    stopped = [
        f"{page.relative_to(ROOT).as_posix()}: {command}"
        for page in documents()
        for command in written_again(words for _, words in r2s_commands(page))
    ]
    assert not stopped, "\n".join(stopped)


def test_the_pages_after_the_quickstart_build_over_its_tasks_only_with_overwrite() -> None:
    """The quickstart's "Next" pages, and the synthetic suite's page, are read after it, in the
    same directory: building the synthetic tasks into its `tasks/` again stopped (`task
    directory already exists`)."""
    following = QUICKSTART.read_text(encoding="utf-8").split("\n## Next\n", 1)[1]
    pages = [
        QUICKSTART.parent / match.group(1).split("#", 1)[0]
        for match in LINK.finditer(following)
        if match.group(1) and not re.match(r"^[a-z][a-z0-9+.-]*:", match.group(1))
    ]
    pages.append(ROOT / "docs" / "benchmarks" / "synthetic.md")
    quickstart = [words for _, words in r2s_commands(QUICKSTART)]
    stopped = [
        f"{page.relative_to(ROOT).as_posix()}: {command}"
        for page in pages
        for command in written_again([*quickstart, *(words for _, words in r2s_commands(page))])
    ]
    assert len(pages) > 1 and not stopped, "\n".join(stopped)


def test_the_training_example_writes_nothing_the_readme_wrote() -> None:
    """The training guide's worked example goes from training tasks to a scorecard; run after
    the README's training commands, its first two stopped on their `train/tasks` and
    `eval/tasks`."""
    readme = [words for _, words in r2s_commands(ROOT / "README.md")]
    example = [words for heading, words in r2s_commands(TRAINING) if heading == TRAINING_EXAMPLE]
    stopped = written_again([*readme, *example])
    assert example and not stopped, "\n".join(stopped)


def test_every_checkpoint_a_page_evaluates_is_trained_on_it_first() -> None:
    """The training guide's worked example evaluated `checkpoint=train/model-40` (and its table
    three more checkpoints) while the page trained only `train/model-5`."""
    untrained: list[str] = []
    for page in documents():
        trained: set[str] = set()
        for _, words in r2s_commands(page):
            if words[1:3] == ["train", "fit"] and "--out" in words:
                trained.add(os.path.normpath(words[words.index("--out") + 1]))
            for match in re.finditer(r"checkpoint=([^,\s]+)", " ".join(words)):
                if os.path.normpath(match.group(1)) not in trained:
                    untrained.append(f"{page.relative_to(ROOT).as_posix()}: {match.group(1)}")
    assert not untrained, "\n".join(untrained)


def test_a_page_that_runs_the_tests_says_how_to_get_them() -> None:
    """`pip install -e .`, the install of the README and the quickstart, has no pytest: it comes
    with the `dev` extra, which the agent tutorial's `pytest -m live` did not mention."""
    runs_tests = re.compile(r"^(?:[A-Z_][A-Z0-9_]*=\S+\s+)*(?:python3? -m )?pytest\b", re.M)
    silent = [
        page.relative_to(ROOT).as_posix()
        for page in documents()
        if any(runs_tests.search(block) for _, block in bash_blocks(page))
        and ".[dev" not in page.read_text(encoding="utf-8")
    ]
    assert not silent, silent


def test_the_pages_name_each_record_that_keeps_a_host_path(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An evaluation given host paths keeps them in `experiment.json` and in each build
    record's method options. The specification's rules said no record holds one; its records
    page and the evaluation guide (the page on sharing results) said only `experiment.json`
    does."""
    from real2simlab.cli.main import main

    twin = tmp_path / "twin"
    shutil.copytree(bouncing_task / "hidden" / "generator", twin)
    run = tmp_path / "run"
    argv = ["eval", "--tasks", str(bouncing_task), "--method", f"hand=program:path={twin}",
            "--backend", "local", "--out", str(run)]  # fmt: skip
    assert main(argv) == 0, capsys.readouterr().err
    given = str(twin).encode()
    holding = {
        path.name for path in run.rglob("*") if path.is_file() and given in path.read_bytes()
    }
    named = {"experiment.json": "`experiment.json`", "build_record.json": "build record"}
    assert holding - {"r2s.log"} == set(named)  # r2s.log is the command's log, not a record
    spec = (ROOT / "docs" / "spec" / "index.md").read_text(encoding="utf-8")
    records = (ROOT / "docs" / "spec" / "records.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "guides" / "evaluation.md").read_text(encoding="utf-8")
    sharing = guide.split("The run directory holds all", 1)[1].split("\n\n", 1)[0]
    pages = {
        "spec/index.md, its rules": spec.split("## Rules that hold everywhere", 1)[1],
        "spec/records.md, its opening": records.split("\n## ", 1)[0],
        "guides/evaluation.md, on sharing a run": sharing,
    }
    unnamed = [
        f"{page}: {name}"
        for page, text in pages.items()
        for name in named.values()
        if name not in " ".join(text.split())
    ]
    assert not unnamed, unnamed


def test_the_training_guide_says_which_files_content_leaves_out() -> None:
    """An export's `content` leaves out `export.json` itself and the card `README.md`
    (`UNRECORDED`), and `r2s train fit` checks the rest; the guide said `content` holds every
    file it wrote and that `r2s train fit` checks every file."""
    from real2simlab.train.dataset import UNRECORDED

    guide = " ".join(TRAINING.read_text(encoding="utf-8").split())
    parts = {
        "content": guide.split("Each export writes", 1)[1].split("overlap-prints.npz", 1)[0],
        "r2s train fit": guide.split("**As exported.**", 1)[1].split("## ", 1)[0],
    }
    unsaid = [f"{part}: {name}" for part, text in parts.items() for name in sorted(UNRECORDED)
              if f"`{name}`" not in text]  # fmt: skip
    assert not unsaid, unsaid


BENCHMARK_NAMES = {
    "Physics-IQ": "physicsiq",
    "PhysCodeGym": "physcodegym",
    "4DCodeBench": "fourdcodebench",
    "PhySimCode": "physimcode",
    "Video2World": "video2world",
    "IRIS": "iris",
}
"""How the pages name each benchmark, and its adapter's package."""


def adapter_variants(package: str) -> set[str]:
    """The task variants an adapter offers (its ``VARIANTS``), read from its source."""
    source = (ROOT / "real2simlab" / "benchmarks" / package / "__init__.py").read_text("utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "VARIANTS" for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    return set()


def test_a_variant_a_page_names_for_a_benchmark_is_one_its_adapter_offers() -> None:
    """The benchmarks overview and the add-a-benchmark tutorial gave 4DCodeBench a `rerun`
    variant, as Physics-IQ and PhysCodeGym have; its adapter's re-run variant is `lab`, and
    `variants=rerun` is refused. A variant named in a page is read as meant for the benchmarks
    named after it, up to the next variant, the end of the clause or the table cell."""
    offered = {name: adapter_variants(package) for name, package in BENCHMARK_NAMES.items()}
    variants = "|".join(sorted(set().union(*offered.values()) - {"native"}))
    wrong: list[str] = []
    for page in documents():
        for paragraph in FENCE.sub("", page.read_text(encoding="utf-8")).split("\n\n"):
            text = " ".join(paragraph.split())
            for match in re.finditer(rf"`({variants})`", text):
                after = re.split(rf"[;|]|\.\s|`(?:{variants})`", text[match.end() :], maxsplit=1)
                wrong += [
                    f"{page.relative_to(ROOT).as_posix()}: `{match.group(1)}` for {name}"
                    for name, has in offered.items()
                    if name in after[0] and match.group(1) not in has
                ]
    assert not wrong, wrong


def test_the_readmes_count_harbors_coding_agents() -> None:
    """The READMEs count every agent Harbor registers (47 in Harbor 0.27), as the paper and the
    website do."""
    factory = pytest.importorskip("harbor.agents.factory")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stated = re.search(r"in Harbor (\d+\.\d+)", readme)
    installed = importlib.metadata.version("harbor")
    assert stated is not None
    if not installed.startswith(f"{stated.group(1)}."):
        pytest.skip(f"the READMEs count Harbor {stated.group(1)}'s agents; {installed} is here")
    registered = set(factory.AgentFactory.registered_names())
    assert f"{len(registered)} coding agents" in readme
    zh = (ROOT / "README_zh.md").read_text(encoding="utf-8")
    assert f"{len(registered)} \u4e2a coding agent" in zh  # the Chinese counter word


def test_the_sdist_ships_what_its_documents_link_to() -> None:
    """The source distribution ships the documentation, the READMEs, ``AGENTS.md`` and the
    tests: what their relative links reach ships too (``AGENTS.md`` links the skills in
    ``.agents/``), or ``test_relative_links_reach_files`` fails in the unpacked sdist."""
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sdist = config["tool"]["hatch"]["build"]["targets"]["sdist"]
    shipped = [ROOT / entry.lstrip("/") for entry in sdist["include"]]

    def ships(path: Path) -> bool:
        path = path.resolve()
        return any(path == top or top in path.parents for top in shipped)

    left_out = [
        f"{page.relative_to(ROOT).as_posix()}: {target}"
        for page in documents()
        if ships(page)
        for target, resolved in relative_links(page)
        if resolved.is_file() and not ships(resolved)
    ]
    assert not left_out, "\n".join(left_out)
