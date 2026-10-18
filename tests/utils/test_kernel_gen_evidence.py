"""Check scientific evidence gates against actual validation artifacts."""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt


_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "skills/btorch-kernel-gen/scripts/check_model.py"
)
_SPEC = importlib.util.spec_from_file_location("kernel_gen_evidence", _SOURCE)
workflow = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(workflow)


def _evidence(tmp_path):
    # Gates consume numerical results from model-specific suites; they do not
    # invent a universal neuroscience model or infer correctness from a plot.
    source = tmp_path / "model.py"
    source.write_text("# A model-specific suite supplies numerical assertions\n")
    checks = []
    for level in workflow.LEVELS:
        figure, axis = plt.subplots()
        axis.plot([0, 1, 2], [0, 1, 0])
        image = tmp_path / f"{level}.png"
        figure.savefig(image)
        plt.close(figure)
        checks.append(
            {
                "name": f"{level}/transient",
                "level": level,
                "status": "PASS",
                "detail": "Suite-specific response assertions passed",
                "figures": [str(image)],
            }
        )
    return workflow.seal_report(
        {
            "source_files": [str(source)],
            "checks": checks,
            "requires_model_confirmation": True,
            "metadata": {"execution": "inference", "gradient": "not tested"},
        }
    )


def test_confirmation_survives_measurements_but_not_changed_scientific_evidence(
    tmp_path,
):
    report = _evidence(tmp_path)
    assert workflow.gate_report(report) == [
        "New/modified Python model awaits human confirmation"
    ]
    approval = {
        "reviewer": "fixture human",
        "report_digest": workflow.confirmation_digest(report),
    }
    # Repeating the same validation or measuring another performance candidate
    # must not erase an unchanged model review solely because time advanced.
    report["created_at"] = "a later validation run"
    report["metrics"] = {"latency_ms": 1.2}
    assert workflow.gate_report(report, approval) == []
    source = Path(report["source_files"][0])
    source.write_text("# The model changed after review\n")
    assert any(
        "Changed evidence" in failure
        for failure in workflow.gate_report(report, approval)
    )
    # Resealing makes the new execution current, but it cannot restore the
    # human's approval of a different model/source version.
    refreshed = workflow.seal_report(report)
    assert any(
        "stale" in failure for failure in workflow.gate_report(refreshed, approval)
    )

    # A helper and the model are different source roles. Swapping their contents
    # can change execution even though the unordered set of hashes is identical.
    helper = tmp_path / "helper.py"
    helper.write_text("# Different helper semantics\n")
    refreshed["source_files"].append(str(helper))
    refreshed = workflow.seal_report(refreshed)
    digest = workflow.confirmation_digest(refreshed)
    model_content, helper_content = source.read_bytes(), helper.read_bytes()
    source.write_bytes(helper_content)
    helper.write_bytes(model_content)
    assert workflow.confirmation_digest(workflow.seal_report(refreshed)) != digest


def test_missing_figures_or_optional_capabilities_never_count_as_pass(tmp_path):
    report = _evidence(tmp_path)
    report["requires_model_confirmation"] = False
    report["checks"][1]["status"] = "UNSUPPORTED"
    report["checks"][1]["detail"] = "No connection adapter"
    failures = workflow.gate_report(report)
    assert any("UNSUPPORTED" in failure for failure in failures)
    assert "No passing pair model response check" in failures
    report["checks"][1]["status"] = "PASS"
    Path(report["checks"][2]["figures"][0]).unlink()
    assert any(
        "Missing evidence" in failure for failure in workflow.gate_report(report)
    )
    # A suite cannot waive the required scientific activity scales in its own
    # report, even when it declares its remaining numerical checks optional.
    report["checks"] = report["checks"][:1]
    report["required_levels"] = []
    report["required_checks"] = []
    assert "No passing pair model response check" in workflow.gate_report(report)


def test_failed_temporal_check_cannot_be_excluded_from_acceptance(tmp_path):
    report = _evidence(tmp_path)
    report["requires_model_confirmation"] = False
    report["required_checks"] = [check["name"] for check in report["checks"]]
    report["checks"].append(
        {
            "name": "delay/arrival",
            "status": "FAIL",
            "detail": "PSC arrived before the declared delay",
        }
    )
    assert any("delay/arrival: FAIL" in item for item in workflow.gate_report(report))


def test_source_text_cannot_substitute_for_activity_figures(tmp_path):
    report = _evidence(tmp_path)
    report["requires_model_confirmation"] = False
    for check in report["checks"]:
        check["figures"] = report["source_files"]
    report = workflow.seal_report(report)
    assert any("Invalid figure" in item for item in workflow.gate_report(report))


def test_unchanged_upstream_relocation_keeps_model_confirmation(tmp_path):
    report = _evidence(tmp_path)
    approval = {
        "reviewer": "fixture human",
        "report_digest": workflow.confirmation_digest(report),
    }
    relocated = tmp_path / "upstream"
    relocated.mkdir()
    replacements = {}
    for path in report["files"]:
        destination = relocated / Path(path).name
        destination.write_bytes(Path(path).read_bytes())
        replacements[path] = str(destination)
    report["source_files"] = [replacements[path] for path in report["source_files"]]
    for check in report["checks"]:
        check["figures"] = [replacements[path] for path in check["figures"]]
    report = workflow.seal_report(report)
    assert workflow.gate_report(report, approval) == []
