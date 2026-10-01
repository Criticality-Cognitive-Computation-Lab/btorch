"""Tests for ``btorch.jit`` -- the optional TorchScript shim.

``btorch.jit`` exposes ``script``, ``script_method``, ``ignore``,
``ScriptModule`` and ``Attribute``. When the environment variable
``BTORCH_JIT`` is falsy, these must degrade to harmless pass-throughs so the
same model code runs in plain eager mode. Because ``JIT_ENABLED`` is read at
import time, each mode is exercised in a fresh subprocess.
"""

import subprocess
import sys
import textwrap

import pytest


def _run(code: str, jit_flag: str, tmp_path) -> str:
    """Run ``code`` in a fresh interpreter with ``BTORCH_JIT=jit_flag``.

    The snippet is written to a real file because TorchScript needs access to
    the function source (``-c`` snippets have none).
    """
    import os

    script = tmp_path / "snippet.py"
    script.write_text(textwrap.dedent(code))
    env = dict(os.environ, BTORCH_JIT=jit_flag)
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_jit_disabled_is_passthrough(tmp_path):
    """With BTORCH_JIT=0 every decorator returns the function unchanged."""
    out = _run(
        """
        import torch
        from btorch import jit
        from btorch.config import JIT_ENABLED

        def f(x):
            return x + 1

        # decorators are identity functions: same object back, no ScriptFunction
        assert jit.script(f) is f
        assert jit.ignore(f) is f
        assert jit.script_method(f) is f
        # ScriptModule falls back to a plain nn.Module
        assert jit.ScriptModule is torch.nn.Module
        # Attribute(val, type) just returns the value
        assert jit.Attribute(3, int) == 3
        print(JIT_ENABLED)
        """,
        "0",
        tmp_path,
    )
    assert out == "False"


def test_jit_enabled_uses_torchscript(tmp_path):
    """With BTORCH_JIT=1 the decorators are the real torch.jit ones."""
    out = _run(
        """
        import torch
        from btorch import jit
        from btorch.config import JIT_ENABLED

        @jit.script
        def f(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        assert isinstance(f, torch.jit.ScriptFunction)
        assert jit.ScriptModule is torch.jit.ScriptModule
        assert float(f(torch.tensor(2.0))) == 4.0
        print(JIT_ENABLED)
        """,
        "1",
        tmp_path,
    )
    assert out == "True"


@pytest.mark.parametrize("flag", ["0", "1"])
def test_jit_scripted_function_matches_eager(flag, tmp_path):
    """A function decorated with ``jit.script`` computes the same numbers
    regardless of whether scripting is on, so toggling the flag is safe."""
    out = _run(
        """
        import torch
        from btorch import jit

        def raw(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x) - 0.5 * x

        wrapped = jit.script(raw)
        x = torch.linspace(-2, 2, 9)
        assert torch.equal(wrapped(x), raw(x))
        print("ok")
        """,
        flag,
        tmp_path,
    )
    assert out == "ok"
