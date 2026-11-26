"""Options given by name on the command line or in a configuration file (``--option k=v``,
``backend_options``, a benchmark's ``options``), checked against the code that takes them
before it runs: an unknown name or a value of the wrong type is a usage error that says what
is expected, not a ``TypeError`` from deep inside.

Options are written into records (``experiment.json``, ``build_record.json``) and some reach a
command line, so an option never holds a credential: :func:`refuse_credentials`."""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, cast
from urllib.parse import parse_qsl, urlsplit

from pydantic import TypeAdapter, ValidationError

from real2simlab.errors import SetupError, UnknownOptionError, UsageError

LISTS = ("Sequence[", "list[", "tuple[", "Collection[", "Iterable[")

_WORD = (
    r"(?:api_?keys?|access_?key|secret_?(?:access_?)?key|private_?key|token|secrets?"
    r"|passwords?|passwd|pwd|passphrase|credentials?|authorization|cookies?|pat)"
)
CREDENTIAL_OPTION = re.compile(rf"(?:^|_){_WORD}(?:_|$)")
"""A word of an option's name (in snake case, between ``_``) that says the option holds a
credential, wherever it is: ``OPENAI_API_KEY``, ``AWS_BEARER_TOKEN_BEDROCK``,
``MINI_SWE_AGENT_MODEL_API_KEY_JSON``. ``tokens`` is a count (``max_tokens``), not that word."""
_FLAG_TAKING = re.compile(rf"(?:^|_){_WORD}$")
"""A command-line flag whose next argument is a credential ends with the word (``--api-key
VALUE``); in the middle, it is a switch (``--skip-credential-checks``)."""
NAMING_SUFFIX = re.compile(
    r"_(?:env|envvar|var|variable|names?|ids?|path|file|dir|url|uri|type"
    r"|count|limit|budget|ttl|expiry|expiration)$"
)
"""The end of a name that names, locates or measures a credential instead of holding it:
``api_key_env``, ``CODEX_AUTH_JSON_PATH``, ``AWS_ACCESS_KEY_ID``, ``token_type``."""
_REFERENCE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
_FLAG = re.compile(r"-{1,2}(?P<name>[A-Za-z][A-Za-z0-9_.-]*)(?:=(?P<value>.*))?", re.DOTALL)
_ASSIGNMENT = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)=(?P<value>.*)", re.DOTALL)
"""``NAME=VALUE`` as an argument: an environment setting (``--env NAME=VALUE``) or a
configuration override (codex's ``-c a.b.name=value``), whose last name part says what it is."""
_HEADER = re.compile(r"(?P<name>[A-Za-z][A-Za-z0-9_-]*):[ \t]*(?P<value>\S.*)", re.DOTALL)
"""``Name: value``, an HTTP header given as an argument (``--header "Authorization: ..."``)."""
CREDENTIAL_FIX = (
    "set it in the environment instead, where it is read at run time (credentials are never "
    "options: options are recorded)"
)


def credential_option(name: str) -> bool:
    """Whether an option's name says that it holds a credential: ``api_key``,
    ``OPENAI_API_KEY``, ``hf_token``, ``client_secret``, ``Authorization``,
    ``AWS_BEARER_TOKEN_BEDROCK``... but not ``api_key_env`` (it names a variable),
    ``token_type`` or ``max_tokens``."""
    words = _words(name)
    return bool(CREDENTIAL_OPTION.search(words)) and NAMING_SUFFIX.search(words) is None


def _words(name: str) -> str:
    """``name`` in snake case: ``apiKey``, ``api-key`` and ``API_KEY`` are ``api_key``."""
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", snake).lower().strip("_")


def _takes_credential(flag: str) -> bool:
    """Whether a flag given without ``=`` takes a credential as its next argument: its name
    ends with a credential's word, and it does not turn something off (``--no-secrets``)."""
    words = _words(flag)
    return bool(_FLAG_TAKING.search(words)) and not words.startswith("no_")


def credential_options(options: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Where ``options`` hold a credential, as paths (``model_kwargs.api_key``): a value under
    a credential's name, a credential's command-line flag with its value (``extra_args``), a
    URL that carries a password or a key. A reference to an environment variable (``$NAME``,
    ``${NAME}``) is a name, not a credential."""
    found: list[str] = []
    for name, value in options.items():
        path = f"{prefix}{name}"
        if credential_option(str(name)) and _literal(value):
            found.append(path)
        elif isinstance(value, Mapping):
            found += credential_options(cast(Mapping[str, Any], value), f"{path}.")
        elif isinstance(value, list | tuple):
            found += _in_sequence(cast(Sequence[Any], value), path)
        elif isinstance(value, str) and _in_text(value):
            found.append(path)
    return found


class CredentialOptionError(SetupError):
    """An option holds a credential (:func:`refuse_credentials`). ``message`` is the error
    without its fix, for a caller that knows a better one (a task's scorer options)."""

    def __init__(self, message: str, fix: str) -> None:
        self.message = message
        super().__init__(message, fix=fix)


def refuse_credentials(options: Mapping[str, Any], what: str, fix: str = CREDENTIAL_FIX) -> None:
    """Raise :class:`CredentialOptionError` when ``options`` hold a credential
    (:func:`credential_options`), naming the options and never their values."""
    found = credential_options(options)
    if found:
        named = (f"option {found[0]} holds" if len(found) == 1 else
                 f"options {', '.join(found)} hold")  # fmt: skip
        raise CredentialOptionError(
            f"{what}: the {named} a credential, and options are recorded (in a run's "
            "experiment.json and build records, in a task's file)",
            fix=fix,
        )


_WORDS = frozenset({"true", "false", "yes", "no", "on", "off", "0", "1"})
"""A switch written as text (``SKIP_CREDENTIAL_CHECKS = "1"``): never a credential."""


def _literal(value: Any) -> bool:
    """Whether ``value`` holds anything but references to environment variables (or a
    switch)."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        text = value.strip()
        return bool(text) and _REFERENCE.fullmatch(text) is None and text.lower() not in _WORDS
    if isinstance(value, Mapping):
        return any(_literal(item) for item in cast(Mapping[str, Any], value).values())
    if isinstance(value, list | tuple | set):
        return any(_literal(item) for item in cast(Collection[Any], value))
    return True  # a number, or another value given as the credential itself


def _in_sequence(items: Sequence[Any], path: str) -> list[str]:
    found: list[str] = []
    for index, item in enumerate(items):
        where = f"{path}[{index}]"
        if isinstance(item, Mapping):
            found += credential_options(cast(Mapping[str, Any], item), f"{where}.")
        elif isinstance(item, list | tuple):
            found += _in_sequence(cast(Sequence[Any], item), where)
        elif isinstance(item, str):
            flag = _FLAG.fullmatch(item)
            if flag is not None and flag["value"] is None and _takes_credential(flag["name"]):
                # The flag's value is the next argument (``--api-key VALUE``).
                following = items[index + 1] if index + 1 < len(items) else None
                if (
                    isinstance(following, str)
                    and not following.startswith("-")
                    and _literal(following)
                ):
                    found.append(where)
            elif _in_text(item):
                found.append(where)
    return found


def _in_text(text: str) -> bool:
    """Whether one text holds a credential: a credential's flag with its value after ``=``, a
    credential's ``NAME=VALUE`` (the last part of a dotted name counts) or ``Name: value``
    header, or a URL whose user part has a password or whose query has a key."""
    flag = _FLAG.fullmatch(text)
    if flag is not None and flag["value"] is not None:
        return credential_option(flag["name"]) and _literal(flag["value"])
    for form in (_ASSIGNMENT, _HEADER):
        found = form.fullmatch(text.strip())
        if (
            found is not None
            and _literal(found["value"])
            and credential_option(found["name"].rsplit(".", 1)[-1])
        ):
            return True
    if "://" not in text:
        return False
    try:
        parts = urlsplit(text.strip())
        return parts.password is not None or any(
            (name.lower() == "key" or credential_option(name)) and _literal(value)
            for name, value in parse_qsl(parts.query)
        )
    except ValueError:
        return False


def checked_options(
    target: Callable[..., Any],
    options: Mapping[str, Any],
    what: str,
    *,
    reserved: Collection[str] = (),
) -> dict[str, Any]:
    """``options`` for ``target`` (a function or a class), checked against its signature.

    A name it does not take is refused, unless it takes ``**options`` (then it checks them
    itself). A value is checked against the parameter's annotation and converted as a
    configuration value would be (``"3"`` for an ``int``); a single name for a parameter that
    takes several is a list of that name (``variants=rerun`` is ``["rerun"]``, not letters).
    ``reserved`` are parameters the caller sets itself (e.g. ``select``, from ``--select``).
    A credential among the options is refused first (:func:`refuse_credentials`).
    """
    refuse_credentials(options, what)
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return dict(options)
    parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    }
    open_ended = any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())
    taken = sorted(name for name in parameters if name not in reserved)
    for name in options:
        if name in reserved:
            raise UsageError(
                f"{what}: {name!r} is not an option; it has its own setting (for example "
                "--select or --out on the command line, select = [...] in an experiment file)"
            )
    unknown = sorted(name for name in options if name not in parameters)
    if unknown and not open_ended:
        raise UnknownOptionError(
            f"unknown options for {what}: {unknown}; it takes: {taken or 'none'}"
        )
    hints = _hints(target)
    checked: dict[str, Any] = {}
    for name, value in options.items():
        hint = hints.get(name)
        if name not in parameters or hint is None:
            checked[name] = value
            continue
        text = str(parameters[name].annotation)
        if isinstance(value, str) and any(kind in text for kind in LISTS):
            value = [value]
        try:
            checked[name] = TypeAdapter(hint).validate_python(value)
        except ValidationError:
            raise UsageError(f"{what}: option {name}={value!r} must be {_readable(text)}") from None
        except Exception:  # an annotation pydantic cannot check: leave it to the target
            checked[name] = value
    return checked


def _hints(target: Callable[..., Any]) -> dict[str, Any]:
    function = target.__init__ if inspect.isclass(target) else target
    try:
        return typing.get_type_hints(function)
    except Exception:  # an annotation that does not resolve here: check names only
        return {}


def _readable(annotation: str) -> str:
    return (
        annotation.replace("Sequence[str]", "a list of names")
        .replace("Path", "a path")
        .replace("int", "a whole number")
        .replace("float", "a number")
        .replace("bool", "true or false")
        .replace("str", "text")
        .replace(" | None", " (or omitted)")
    )


__all__ = [
    "CredentialOptionError",
    "checked_options",
    "credential_option",
    "credential_options",
    "refuse_credentials",
]
