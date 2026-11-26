# Specification 1.0

Real2SimLab's specification fixes the files that pass between benchmarks, methods and the Lab,
so that any method's delivery can be built, frozen, re-run and scored the same way on any
benchmark, and its result read the same way. It is **1.0** and frozen: see
[Compatibility](#compatibility).

| Object | Version string | Page | JSON Schema |
|---|---|---|---|
| Task directory, `task.toml` | `r2s-task/1.0` | [Tasks](task.md) | [r2s-task](schemas/r2s-task.schema.json) |
| Capture, `capture.json` | `r2s-capture/1.0` | [Captures](capture.md) | [r2s-capture](schemas/r2s-capture.schema.json) |
| Twin package, `twin.json` | `r2s-twin/1.0` | [Twins](twin.md) | [r2s-twin](schemas/r2s-twin.schema.json) |
| `tracks` output format | `r2s-tracks/1.0` | [Output formats](formats.md) | [meta](schemas/r2s-tracks-meta.schema.json), [camera](schemas/r2s-tracks-camera.schema.json) |
| Scoring protocols | `reproduction/1.0`, `native` | [Scoring](scoring.md) | — |
| Run record | `r2s-run/1.0` | [Records](records.md) | [r2s-run](schemas/r2s-run.schema.json) |
| Build record | `r2s-build/1.0` | [Records](records.md) | [r2s-build](schemas/r2s-build.schema.json) |
| Agent record | `r2s-agent-run/1.0` | [Records](records.md) | [r2s-agent-run](schemas/r2s-agent-run.schema.json) |
| Task result | `r2s-task-result/1.0` | [Records](records.md) | [r2s-task-result](schemas/r2s-task-result.schema.json) |
| Aggregate results and scorecard | `r2s-results/1.0` | [Records](records.md) | [r2s-results](schemas/r2s-results.schema.json) |
| Experiment configuration | `r2s-experiment/1.0` | [Records](records.md#experiments) | [r2s-experiment](schemas/r2s-experiment.schema.json) |
| Training record of a checkpoint, `training.json` | `r2s-training/1.0` | [Records](records.md#r2s-training10-a-checkpoint) | [r2s-training](schemas/r2s-training.schema.json) |

Regenerate the schemas with `r2s spec schema --out docs/spec/schemas`; a test fails when they
are stale.

## Versions

- Every document names its object and version in a `spec` field, e.g.
  `spec = "r2s-task/1.0"`.
- A reader accepts the same major version with a minor version up to its own. A newer minor
  version may add optional fields; an older Lab rejects such a document with a message that
  asks for an upgrade, instead of silently dropping fields.
- The specification versions are independent of the package version.

## Compatibility

The specification 1.0 is frozen. Two rules govern every later change:

1. **Within 1.x, only additions.** A minor version (1.1, 1.2, ...) may add optional fields,
   new objects, output formats, delivery conventions, metrics and protocols. It never removes
   or renames anything, never changes the meaning or unit of an existing field, and never makes
   an optional field required. Every 1.0 document stays valid and means the same.
2. **A breaking change needs a new major version** (2.0) of the object it changes, together with
   a migration note in the changelog that says what changed and how to convert documents. Labs
   reject documents of a major version they do not implement, with a message saying so.

## Rules that hold everywhere

- **Strict models.** Every object rejects unknown fields. Extensions go into the places made
  for them (`Capture.extensions`, task `metadata`).
- **Explicit failure.** Missing inputs, wrong formats and broken environments raise clear
  errors; nothing degrades silently. A method that delivers no valid twin is a *result*
  (scored as zero), not an exception.
- **Units and frames.** Metres, seconds, radians, kilograms. The world frame is right-handed
  with +Z up; gravity points along −Z. Cameras use OpenCV axes (+x right, +y down, +z forward);
  `intrinsics` is a 3×3 pixel matrix, `extrinsics` a 4×4 camera-to-world transform. Poses are
  `[x, y, z, qw, qx, qy, qz]`. Frame `i` is at time `i / fps`.
- **Provenance.** Every non-media quantity in a capture carries how it was obtained:
  `measured`, `estimated`, `default`, `human_edited` or `synthetic`.
- **Paths.** Paths inside documents are relative to the document's directory and may not
  contain `..`; symbolic links are not allowed in tasks, twins or outputs. The exception is
  what a user points to: an experiment file's task paths and the method options that name files
  (`path`, `checkpoint`) start from its directory and may contain `..` or be absolute
  ([Experiments](records.md#experiments)), and records keep the paths a command was given as
  they were given (next rule).
- **No credentials** in any record. **No host paths** either, except where a record keeps
  what a command was given: `experiment.json` (the experiment as given, and where its tasks
  are), a build record's method `options`, and a checkpoint's `dataset` ([Records](records.md)).
