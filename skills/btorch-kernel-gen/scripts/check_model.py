"""Execute model validation suites and check their scientific evidence."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


LEVELS = ("single", "pair", "network")
STATUSES = {"PASS", "FAIL", "BLOCKED", "UNSUPPORTED", "NOT_RUN"}


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _load_suite(identifier):
    module_name, separator, function = identifier.rpartition(":")
    if not separator:
        raise ValueError("Suite must be module:callable or source.py:callable")
    path = Path(module_name)
    if path.is_file():
        name = "_kernel_gen_suite_" + _digest(str(path.resolve()))[:12]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, function)


def seal_report(report):
    """Attach file provenance to a suite's actual numerical and visual
    evidence."""
    report = dict(report)
    for key in (
        "source_files",
        "model_source_files",
        "figures",
        "artifacts",
        "model_artifacts",
    ):
        if key in report:
            report[key] = [str(Path(path).resolve()) for path in report[key]]
    report["checks"] = [dict(check) for check in report.get("checks", [])]
    for check in report["checks"]:
        if "figures" in check:
            check["figures"] = [str(Path(path).resolve()) for path in check["figures"]]
    paths = list(report.get("source_files", []))
    paths.extend(report.get("model_source_files", []))
    paths.extend(report.get("figures", []))
    paths.extend(report.get("artifacts", []))
    paths.extend(report.get("model_artifacts", []))
    for check in report.get("checks", []):
        paths.extend(check.get("figures", []))
    report["files"] = {
        str(Path(path).resolve()): _sha(path) for path in dict.fromkeys(paths)
    }
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["schema_version"] = 1
    return report


def confirmation_digest(report):
    """Identify model evidence independently of timestamps and optimization
    metrics."""
    checks = [
        check for check in report.get("checks", []) if check.get("level") in LEVELS
    ]
    paths = list(report.get("model_source_files", report.get("source_files", [])))
    for check in checks:
        paths.extend(check.get("figures", []))
    paths.extend(report.get("model_artifacts", report.get("artifacts", [])))
    files = report.get("files", {})
    scientific_checks = []
    for check in checks:
        content = {key: value for key, value in check.items() if key != "figures"}
        content["figure_hashes"] = [
            files.get(path) for path in check.get("figures", [])
        ]
        scientific_checks.append(content)
    return _digest(
        {
            "checks": scientific_checks,
            # Moving unchanged OOT evidence into an upstream worktree is not a
            # new model. Validate current paths separately, approve content.
            "file_hashes": [files.get(path, "missing") for path in paths],
            "model_contract": report.get("model_contract", report.get("metadata", {})),
        }
    )


def gate_report(report, confirmation=None):
    """Return actionable failures without counting missing capabilities as
    passes."""
    failures = []
    checks = report.get("checks", [])
    names = [check.get("name") for check in checks]
    if not checks or any(not isinstance(name, str) or not name for name in names):
        failures.append("No named checks executed")
    if len(set(names)) != len(names):
        failures.append("Check names must identify independent results")
    required = report.get("required_checks", names)
    lookup = {check.get("name"): check for check in checks}
    for name in required:
        check = lookup.get(name)
        if check is None or check.get("status") != "PASS":
            detail = "missing" if check is None else check.get("detail", "")
            status = "NOT_RUN" if check is None else check.get("status")
            failures.append(f"{name}: {status}; {detail}")
    for check in checks:
        if check.get("status") not in STATUSES:
            failures.append(f"{check.get('name')}: unknown status")
        elif check.get("status") == "FAIL" and check.get("name") not in required:
            failures.append(f"{check.get('name')}: FAIL; {check.get('detail', '')}")
    files = report.get("files", {})
    if not report.get("source_files"):
        failures.append("No model/case source provenance recorded")
    required_paths = list(report.get("source_files", []))
    required_paths.extend(report.get("model_source_files", []))
    required_paths.extend(report.get("figures", []))
    required_paths.extend(report.get("artifacts", []))
    required_paths.extend(report.get("model_artifacts", []))
    for check in checks:
        required_paths.extend(check.get("figures", []))
    for path in required_paths:
        if str(Path(path).resolve()) not in files:
            failures.append(f"Unsealed evidence: {path}")
    for path, expected in files.items():
        source = Path(path)
        if not source.is_file():
            failures.append(f"Missing evidence: {path}")
        elif _sha(source) != expected:
            failures.append(f"Changed evidence: {path}; rerun validation")
    for level in LEVELS:
        evidence = [
            check
            for check in checks
            if check.get("level") == level and check.get("status") == "PASS"
        ]
        if not evidence:
            failures.append(f"No passing {level} model response check")
        elif not any(check.get("figures") for check in evidence):
            failures.append(f"Missing {level} activity figure")
    figure_paths = list(report.get("figures", []))
    for check in checks:
        figure_paths.extend(check.get("figures", []))
    for path in dict.fromkeys(figure_paths):
        try:
            if Path(path).suffix.lower() == ".svg":
                if ET.parse(path).getroot().tag.split("}")[-1] != "svg":
                    raise ValueError("Not an SVG document")
            elif Path(path).suffix.lower() == ".pdf":
                if not Path(path).read_bytes().startswith(b"%PDF-"):
                    raise ValueError("Not a PDF document")
            else:
                with Image.open(path) as image:
                    image.verify()
        except (OSError, ValueError, ET.ParseError) as error:
            failures.append(f"Invalid figure: {path}; {error}")
    if report.get("requires_model_confirmation", False):
        if confirmation is None:
            failures.append("New/modified Python model awaits human confirmation")
        elif confirmation.get("report_digest") != confirmation_digest(report):
            failures.append("Model confirmation is stale; review new activity figures")
        elif not confirmation.get("reviewer"):
            failures.append("Model confirmation has no human reviewer")
    return failures


def verify(args):
    report = seal_report(_load_suite(args.suite)())
    _write(args.output, report)
    failures = gate_report(report)
    print(json.dumps({"report": args.output, "failures": failures}, indent=2))
    return (
        2 if any("confirmation" in failure for failure in failures) else bool(failures)
    )


def confirm(args):
    # This command records a user's decision; the skill must never invoke it
    # on the user's behalf. It is not an autonomous scientific approval tool.
    report = _read(args.report)
    provisional = dict(report, requires_model_confirmation=False)
    failures = gate_report(provisional)
    if failures:
        raise ValueError("Cannot confirm invalid evidence: " + "; ".join(failures))
    output = args.output or str(Path(args.report).with_suffix(".confirmation.json"))
    _write(
        output,
        {
            "reviewer": args.reviewer,
            "report_digest": confirmation_digest(report),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(output)
    return 0


def gate(args):
    confirmation = _read(args.confirmation) if args.confirmation else None
    failures = gate_report(_read(args.report), confirmation)
    print(json.dumps({"passed": not failures, "failures": failures}, indent=2))
    return int(bool(failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("verify", help="Execute an actual model/kernel suite")
    p.add_argument("--suite", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=verify)
    p = commands.add_parser("confirm", help="Record a human's model/figure review")
    p.add_argument("--report", required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--output")
    p.set_defaults(func=confirm)
    p = commands.add_parser("gate")
    p.add_argument("--report", required=True)
    p.add_argument("--confirmation")
    p.set_defaults(func=gate)
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, OSError) as error:
        print(json.dumps({"status": "BLOCKED", "detail": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
