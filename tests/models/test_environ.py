"""Tests for btorch.models.environ: global defaults + thread-local overrides.

``environ.set`` stores process-global defaults (visible from every thread);
``environ.context`` pushes thread-local overrides that never leak to other
threads and always win over the global default inside their ``with`` block.
"""

import threading

import pytest

from btorch.models import environ


KEY = "_test_environ_key"


@pytest.fixture(autouse=True)
def _clean_global_default():
    # Make sure the global default is removed before and after each test.
    environ.unset(KEY)
    yield
    environ.unset(KEY)


def _in_thread(fn):
    """Run ``fn`` in a new thread and return its result (or re-raise)."""
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_set_is_visible_from_other_threads():
    """environ.set defaults are process-global, not per-thread."""
    environ.set(**{KEY: 7})
    assert environ.get(KEY) == 7
    assert _in_thread(lambda: environ.get(KEY)) == 7
    assert _in_thread(lambda: environ.all()[KEY]) == 7


def test_set_from_worker_thread_is_visible_in_main_thread():
    _in_thread(lambda: environ.set(**{KEY: 3}))
    assert environ.get(KEY) == 3


def test_context_overrides_global_only_in_its_own_thread():
    """A ``with environ.context`` override is thread-local."""
    environ.set(**{KEY: 1})
    entered, release = threading.Event(), threading.Event()
    seen = {}

    def worker():
        with environ.context(**{KEY: 99}):
            seen["inside"] = environ.get(KEY)
            entered.set()
            release.wait(5)
        seen["after"] = environ.get(KEY)

    t = threading.Thread(target=worker)
    t.start()
    assert entered.wait(5)
    # While the worker is inside its context the main thread still sees the
    # global default.
    assert environ.get(KEY) == 1
    release.set()
    t.join()
    assert seen == {"inside": 99, "after": 1}


def test_context_stack_nesting_and_restore():
    environ.set(**{KEY: 1})
    with environ.context(**{KEY: 2}):
        with environ.context(**{KEY: 3}):
            assert environ.get(KEY) == 3
        assert environ.get(KEY) == 2
    assert environ.get(KEY) == 1


def test_get_missing_key_raises():
    with pytest.raises(KeyError):
        environ.get(KEY)


def test_unset_removes_global_default_but_not_context():
    """``unset`` drops a global default; unknown keys are ignored."""
    environ.set(**{KEY: 5})
    environ.unset(KEY, "_never_set_key")
    with pytest.raises(KeyError):
        environ.get(KEY)
    # A context override is thread-local state and independent of ``unset``.
    with environ.context(**{KEY: 6}):
        environ.unset(KEY)
        assert environ.get(KEY) == 6


def test_concurrent_set_and_read_are_consistent():
    """Readers racing writers always see a complete, consistent snapshot.

    Writers keep two keys equal (``a == b``) in every ``set`` call. Because
    reads go through a copy-on-write snapshot, ``environ.all()`` can never
    observe a state where only one of the two keys has been updated, and
    ``get`` never raises or returns a torn value.
    """
    key_a, key_b = KEY + "_a", KEY + "_b"
    environ.set(**{key_a: 0, key_b: 0})
    stop = threading.Event()
    errors = []

    def writer(offset):
        for i in range(2000):
            environ.set(**{key_a: offset + i, key_b: offset + i})
        stop.set()

    def reader():
        while not stop.is_set():
            snap = environ.all()
            if snap[key_a] != snap[key_b]:
                errors.append(snap)
                return
            environ.get(key_a)

    try:
        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=writer, args=(10_000,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
    finally:
        environ.unset(key_a, key_b)


def test_concurrent_writers_do_not_lose_updates():
    """Parallel ``set`` calls on distinct keys must all survive (no lost
    write)."""
    keys = [f"{KEY}_{i}" for i in range(8)]

    def write(k):
        for i in range(200):
            environ.set(**{k: i})

    try:
        threads = [threading.Thread(target=write, args=(k,)) for k in keys]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(environ.get(k) == 199 for k in keys)
    finally:
        environ.unset(*keys)
