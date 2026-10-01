"""``btorch.models`` exposes every public submodule, and ``__all__`` is
accurate."""

import importlib
import pkgutil

import btorch.models as models


def test_all_names_resolve():
    """Every name listed in ``__all__`` is an attribute of the package."""
    missing = [n for n in models.__all__ if not hasattr(models, n)]
    assert missing == []
    assert len(set(models.__all__)) == len(models.__all__)


def test_every_public_submodule_is_exported():
    """No public (non-underscore) submodule or subpackage is left out."""
    on_disk = {
        m.name
        for m in pkgutil.iter_modules(models.__path__)
        if not m.name.startswith("_")
    }
    assert on_disk <= set(models.__all__)


def test_every_neuron_submodule_is_exported_from_neurons_package():
    """Neuron modules are reachable from ``neurons`` and (flat) from
    ``models``."""
    neurons = importlib.import_module("btorch.models.neurons")
    on_disk = {
        m.name
        for m in pkgutil.iter_modules(neurons.__path__)
        if not m.name.startswith("_")
    }
    assert on_disk <= set(neurons.__all__)
    for name in on_disk:
        assert getattr(models, name) is getattr(neurons, name)


def test_dendritic_lif_lives_in_neurons():
    """``DendriticLIF`` and friends moved to ``btorch.models.neurons.dlif``."""
    from btorch.models.neurons import dlif

    assert models.DendriticLIF is dlif.DendriticLIF
    assert models.DLIF is dlif.DLIF and models.DBNN is dlif.DBNN
