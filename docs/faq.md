# FAQ

**Why score a frozen re-run instead of what the agent produced?**
An agent's session can show anything: a render made by hand, a file copied from elsewhere, a
result that depends on its environment. The frozen re-run is what the delivered program actually
does, offline and read-only, so it can be reproduced by anyone with the record. The Lab still
compares it with the agent's own output and reports whether they agree.

**Are the scores comparable with each benchmark's published numbers?**
The Lab runs each benchmark's official scorer from a pinned checkout, on the benchmark's own
task statement, and each adapter page says how it was checked against the official pipeline
(Physics-IQ matches to the last digit). Two caveats are stated per page: the Lab may run a
subset of a benchmark's tasks, and scores of the frozen re-run can differ from scores of the
session output when an agent's program does not reproduce itself.

**Why is the scorecard not one number?**
Each benchmark measures something different in its own unit. The scorecard keeps one column per
benchmark, with its reference points (a static scene, a second recording, the benchmark's
reference solution), so a value can be read against what is easy and what is achievable.

**Can a method read the hidden data?**
No. `hidden/` (references, other recordings, what official scorers need) is never mounted into an
agent's workspace or a twin's sandbox, `r2s task validate` checks that nothing of it leaks into
`visible/` or the statement, and the program is fingerprinted before its first run and checked
after every run.

**Do I need a GPU?**
Not for the synthetic suite, IRIS or PhySimCode. Physics-IQ's and PhysCodeGym's Blender scenes,
4DCodeBench's official scorer and Video2World's evaluator (SAPIEN) use one; `r2s doctor --gpu`
checks it.

**Do I need Docker?**
Not to run twins: `--backend bwrap` runs them on Linux without root, where bubblewrap may create
user namespaces and mount a new `/proc` (`r2s doctor` checks both; many unprivileged containers
do not allow the second; see [clusters](tutorials/clusters.md)). Two official scorers run in
Docker whatever the backend, with a GPU: 4DCodeBench's and Video2World's evaluator, so those two
benchmarks need Docker. `--backend local` runs without isolation, for debugging your own code
only.

**Where are my API keys stored?**
Nowhere by Real2SimLab: they are read from your environment (or your agent's login directory)
when an agent starts and passed only to its sandbox; agent output is redacted before it is
written. See [Security](security.md).

**How does this relate to Harbor?**
Harbor runs agents on tasks in general; Lab tasks export to Harbor, and Harbor's agents run in
the Lab: see [Related work](related.md#harbor) and [the Harbor guide](guides/harbor.md).

**Is the specification stable?**
Yes. Specification 1.0 is frozen: within 1.x it only gains optional additions, a breaking
change needs a new major version with a migration note, and every record names the version it
was written with. See [Compatibility](spec/index.md#compatibility).
