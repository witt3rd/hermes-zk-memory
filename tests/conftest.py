"""Shared fixtures for hermes-zk-memory tests.

The plugin module lives at the repo root as ``__init__.py`` (loaded by
Hermes as ``plugins.memory.zk-memory``). For test isolation we import it
as a synthetic ``hermes_zk_memory`` module via importlib so we don't
collide with the ``tests`` package.

``__init__.py`` does ``from agent.memory_provider import MemoryProvider``
at module scope, so a real (or stub) ``agent`` package must be importable
before we load it. We prefer a real hermes-agent checkout if one exists
on this machine; otherwise we fall back to a minimal local stub package
under ``tests/_agent_stub/`` so the suite still runs on a bare machine/CI
with no hermes-agent install, no network, no docker, and no real LLM
provider key.

The plugin is a thin adapter over the sibling ``zk-memory`` library, so
that checkout must be importable too — same pattern as hermes-prospecta
putting ``prospecta`` on sys.path.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STUB_AGENT_DIR = Path(__file__).resolve().parent / "_agent_stub"
# The sibling zk-memory library. NOTE: the worktree lives under
# hermes-zk-memory.wt/, so REPO_ROOT.parent is NOT the source parent —
# hardcode the candidate like the hermes-agent candidates below.
_SIBLING_ZK_MEMORY_CANDIDATES = [
    Path("/home/dt/src/witt3rd/zk-memory"),
]

# Candidate real hermes-agent checkouts on this machine. NOTE: an older
# sibling plugin's conftest.py hardcodes /home/dt/src/ext/hermes-agent,
# which does not exist here -- don't copy that path blindly.
_REAL_HERMES_CANDIDATES = [
    Path("/home/dt/.hermes/hermes-agent"),
    Path("/home/dt/src/ext/hermes-agent"),
]


def _agent_package_importable(path: Path) -> bool:
    return (path / "agent" / "memory_provider.py").is_file() and (
        path / "agent" / "auxiliary_client.py"
    ).is_file()


def _ensure_agent_package_on_path() -> str:
    """Put a real hermes-agent checkout on sys.path if one is available;
    otherwise fall back to the local stub package. Returns which source
    was used, for diagnostics."""
    for candidate in _REAL_HERMES_CANDIDATES:
        if _agent_package_importable(candidate):
            p = str(candidate)
            if p not in sys.path:
                sys.path.insert(0, p)
            try:
                import agent.memory_provider  # noqa: F401
                import agent.auxiliary_client  # noqa: F401
                return f"real:{candidate}"
            except Exception:
                # Real checkout present but not actually importable
                # (missing transitive deps, etc) -- fall through to stub.
                sys.path.remove(p)
                continue

    p = str(STUB_AGENT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return "stub"


_AGENT_SOURCE = _ensure_agent_package_on_path()

# The sibling zk-memory library must be importable: __init__.py does
# ``from zk_memory import Memory`` (not a relative sibling import).
for _candidate in _SIBLING_ZK_MEMORY_CANDIDATES:
    if (_candidate / "zk_memory" / "__init__.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

# Repo root must be on sys.path too: __init__.py does plain top-level
# ``import llm as _llm``, not a relative import.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_plugin_module():
    """Load the plugin's root __init__.py the way hermes does — as a package
    under the synthetic user-namespace (_hermes_user_memory.<name>), with
    siblings registered as submodules — so its RELATIVE sibling imports
    (``from . import llm as _llm``) resolve the same as at runtime.

    hermes' loader (plugins/memory/__init__.py:_load_provider_from_dir) does
    exactly this for a user-installed provider. Tests must reproduce it or the
    plugin's relative imports fail under the flat import."""
    pkg_name = "_hermes_user_memory"
    name = "hermes_zk_memory"
    module_name = f"{pkg_name}.{name}"

    if pkg_name not in sys.modules:
        parent = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(pkg_name, None, is_package=True)
        )
        parent.__path__ = []
        sys.modules[pkg_name] = parent
    for sub_file in sorted(REPO_ROOT.glob("*.py")):
        if sub_file.name == "__init__.py":
            continue
        sub = sub_file.stem
        full = f"{module_name}.{sub}"
        if full not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(full, str(sub_file))
            if sub_spec and sub_spec.loader:
                sub_mod = importlib.util.module_from_spec(sub_spec)
                sys.modules[full] = sub_mod
                try:
                    sub_spec.loader.exec_module(sub_mod)
                except Exception:
                    pass
    if module_name in sys.modules and getattr(sys.modules[module_name], "__file__", None):
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, str(REPO_ROOT / "__init__.py"),
        submodule_search_locations=[str(REPO_ROOT)],
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = module_name  # required for `from . import llm`
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def agent_source() -> str:
    """Which agent package backed the import: 'real:<path>' or 'stub'."""
    return _AGENT_SOURCE


@pytest.fixture(scope="session")
def plugin_module():
    return _load_plugin_module()


@pytest.fixture(scope="session")
def corpus_module():
    """The sibling zk-memory corpus module (for linlink patching etc)."""
    import zk_memory.corpus as corpus
    return corpus


@pytest.fixture(scope="session")
def llm_module():
    """The plugin's Hermes StructuredLLM adapter module."""
    import llm as llm_mod
    return llm_mod


class _StubCtx:
    """Captures register_* calls made by the plugin's register(ctx)."""

    def __init__(self):
        self.auxiliary_tasks = []
        self.memory_providers = []

    def register_auxiliary_task(self, *, key, display_name=None, description=None, defaults=None, **kwargs):
        self.auxiliary_tasks.append(
            {
                "key": key,
                "display_name": display_name,
                "description": description,
                "defaults": defaults,
                **kwargs,
            }
        )

    def register_memory_provider(self, provider):
        self.memory_providers.append(provider)


@pytest.fixture
def stub_ctx():
    return _StubCtx()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A tmp HERMES_HOME with $HERMES_HOME set; corpus root is home/zk."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def zk_root(hermes_home):
    """The zk corpus dir under hermes_home (not pre-created; write()/etc
    create it lazily, matching real usage)."""
    return hermes_home / "zk"


@pytest.fixture
def provider(plugin_module, hermes_home):
    """An initialized ZkMemoryProvider rooted at hermes_home/zk."""
    p = plugin_module.ZkMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(hermes_home))
    yield p
    p.shutdown()


@pytest.fixture(autouse=True)
def _no_linlink_by_default(monkeypatch, corpus_module):
    """Force corpus.write()/tend() down their no-linlink fallback path by
    default, regardless of whether linlink happens to be installed on the
    machine running the tests. Tests that specifically want to exercise
    the "linlink present" branch install their own fake linlink and
    monkeypatch zk_memory.corpus.shutil.which again inside the test body,
    which takes precedence over this default (both use the same per-test
    monkeypatch fixture, and the later call wins)."""
    # corpus_module.shutil IS the real (shared) shutil module -- capture
    # the original `which` before patching it, since patching in place
    # would otherwise make the "real" fallback call itself recursively.
    _real_which = corpus_module.shutil.which

    def _which_no_linlink(name):
        if name == "linlink":
            return None
        return _real_which(name)

    monkeypatch.setattr(corpus_module.shutil, "which", _which_no_linlink)


@pytest.fixture(autouse=True)
def _hermetic_config(monkeypatch):
    """Isolate tests from the live profile's config.

    The adapter's ``initialize()`` calls ``load_config_readonly()`` which
    reads the real profile's ``memory.zk_corpus_root`` — a production
    provisioned value. Tests inject ``hermes_home`` pointing at a tmp_path
    and assert root resolution against that, but the live config silently
    overrides it. Patch to empty dict so every test resolves against its own
    sandbox, never the live root. Tests that need specific config values
    can override via their own ``monkeypatch.setattr``."""
    try:
        import hermes_cli.config
        monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: {})
    except ImportError:
        # No hermes_cli available (standalone CI) — config isn't reachable anyway.
        pass