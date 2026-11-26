# Security

Real2SimLab runs code written by coding agents and other methods. This page states what its
sandboxes do and do not protect, and how credentials are handled.

## Reporting a vulnerability

Please report security issues privately to the maintainers (a GitHub security advisory on the
repository), not in a public issue. Include the backend, the command and what you observed.

## Sandbox boundaries

| | `docker` (default) | `bwrap` | `local` |
|---|---|---|---|
| Twin re-runs: network | none (`--network none`) | none (all namespaces unshared) | **not isolated** |
| Program directory | read-only | read-only | **writable** |
| Writable | the output directory, `/tmp` (tmpfs) | the output directory, `/tmp` (tmpfs) | anything |
| Hidden task data | never mounted | never mounted | **reachable by path** |
| User | the host user's uid, all capabilities dropped, `no-new-privileges`, read-only root | the host user, user namespace | the host user |
| Limits | time, CPUs, memory, processes | time; CPUs where `taskset` is installed; memory and processes where a systemd user scope can limit them | time |

`local` exists to debug your own code; it prints a warning and every record says
`isolated: false`. On Linux it runs each program (a twin, an agent's bootstrap and CLI) under a
supervisor: what the program starts (a process in a session of its own too) ends with its run,
and the program ends when `r2s` is killed (`kill -9`); the commands of a `mini_swe` session
are not supervised. Use `docker` or `bwrap` for evaluation. The one evaluation that uses it is
the verifier of a task exported to Harbor, inside Harbor's offline verifier container: there
the task's `hidden/` directory is made readable by root only and the solution runs as an
unprivileged user (`nobody`) who owns only its output directory, so it cannot read the
reference it is scored against; without root the verifier refuses to score.

What a program or an agent prints is kept up to a limit on every backend: a run's log 16 MiB
(its beginning and its last MiB), an agent's trajectory and stderr 256 MiB each, always under
the machine's file-size limit (`ulimit -f`), with a line saying what was left out. The
`docker` backend turns Docker's own log of its containers off (`--log-driver none`), which
would keep a second copy, without a limit, on the daemon's disk. The agents whose harness runs
on the host keep as much: what Harbor or the mini-swe-agent harness prints (16 MiB), the copy
of Harbor's trial and mini-swe-agent's trajectory (256 MiB); Harbor's own job directory
(`work/harbor/jobs`) holds what Harbor wrote there. The files a program writes into its output
directory, and an agent into its workspace, have no size limit of their own: give untrusted
methods a disk quota or a file system of their own.

Agents building twins run in the same kind of sandbox, with the workspace (visible data,
documents, the SDK, `solution/`) as the only writable mount. Their network is, by default, an
allowlist: an internal Docker network whose only route out is an egress proxy that forwards
connections to the listed hosts (the model API) and logs every connection; `mini_swe` agents'
commands have no network at all (the harness on the host talks to the model). An agent run
with `network = "open"` is labelled as such in every report.

What an agent leaves in its writable directories (its workspace and its log directory) is
data it controls, and the Lab reads it on the host once the sandbox has ended. It reads it as
it is: a symbolic link there is never followed (it would reach, on the host, what the sandbox
kept from the agent, such as the task's hidden data) and a named pipe is never waited on. A
`solution/` that is a symbolic link is not delivered; a session output that is one, or holds
one, a pipe, a socket or a device, or a file the Lab cannot read, is not used (as if the agent
left none: a task scored on it scores no output); a log of the agent's in such a state is not
read, recorded or exported; and the Lab's own `agent_record.json` is written as a new file,
never through what the agent left under that name.

A container is not a virtual machine: a kernel or container-runtime vulnerability can break
out of it. Run untrusted agents on machines you can afford to lose, keep Docker and the kernel
updated, and do not mount sensitive host directories.

## Credentials

- Keys and login files are never written into the repository, task or run directories, logs,
  records or image layers.
- They are injected only at run time: secret environment variables are passed to `docker run`
  by name (`-e NAME`), so their values never appear on a command line; credential files (e.g.
  Codex's `auth.json`) are passed base64-encoded in such a variable and written by the
  in-sandbox bootstrap into a tmpfs that disappears with the container. The `harbor` method
  passes `agent_env = {KEY = "$NAME"}` to Harbor as `KEY=${NAME}`, which Harbor reads from its
  own environment; an unset variable is an error.
- Options are written into records, so an option never holds a credential: one named like a
  credential (`api_key`, `token`, `password`, ... anywhere in its name and at any depth, e.g.
  `model_kwargs.api_key` or `AWS_BEARER_TOKEN_BEDROCK`, but not a name that only names one,
  like `api_key_env`), a key given with its flag in `extra_args`, or a URL with a password is
  refused before anything runs. Name the variable that holds it instead (`api_key_env`,
  Harbor's `agent_env`).
- Agent output is redacted *inside* the sandbox before it reaches the disk: the injected
  values, and strings shaped like common keys (OpenAI, Anthropic, Google, GitHub, AWS, Hugging
  Face, Slack, JWTs, bearer tokens, private keys, `api_key=` assignments). Of an injected login
  file, every value named like a credential counts as injected (from 6 characters, as a
  variable's value does), and a small file also in full and line by line.
- Files an agent writes itself (its workspace and solution, or straight into its log
  directory, also ones it made unreadable) are checked on the host when it stops: the injected
  values found in them are replaced with `[REDACTED]`, and a file that cannot be rewritten is
  removed, before the solution is frozen, shown in a report or exported. A report or a dataset
  shows an agent's trajectory only once its record says so, which is written after this check.
  When this machine fails meanwhile (a full disk), the build is not a result and is redone.
  Until the agent stops (or while the Lab is stopped before the check), the values it wrote
  are in its build directory. The record counts all the redactions.
- The `mini_swe` harness keeps the API key in its own process; agent commands run with a fixed
  environment that does not contain it.

Redaction by pattern is a safety net, not a guarantee: an agent that prints a transformed key
defeats it. Use keys with spending limits for evaluations.
