import contextlib
import logging

import pytest
import torch

from btorch.models.rnn import make_rnn
from btorch.utils.file import fig_path
from tests.models.test_rnn import SimpleRNNCell


@contextlib.contextmanager
def file_logger_ctx(name_suffix):
    """Context manager to capture logs to a file using fig_path."""
    log_dir = fig_path(__file__)
    log_file = log_dir / f"compilation_{name_suffix}.log"

    # Remove existing log file if it exists
    if log_file.exists():
        log_file.unlink()

    # Create file handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Add handler to torch dynammo/inductor loggers explicitly
    loggers = [
        logging.getLogger("torch._dynamo"),
        logging.getLogger("torch._inductor"),
        logging.getLogger("torch.fx.experimental.symbolic_shapes"),
    ]
    for logger in loggers:
        logger.addHandler(file_handler)
        # Ensure level is set low enough to capture
        logger.setLevel(logging.DEBUG)

    try:
        yield log_file
    finally:
        # Teardown
        for logger in loggers:
            logger.removeHandler(file_handler)
        file_handler.close()


@pytest.mark.skip("Large output file")
@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_compile_logging(grad_checkpoint):
    with file_logger_ctx(f"checkpoint_{grad_checkpoint}") as log_file:
        print(f"Capturing logs to: {log_file}")

        # Enable torch.compile logging
        # Output code is useful to see if it compiled
        torch._logging.set_logs(
            dynamo=logging.DEBUG, inductor=logging.DEBUG, output_code=True
        )

        # Create a simple RNN model
        # make_rnn(cell_instance) returns a wrapped instance
        cell = SimpleRNNCell(input_size=10, hidden_size=20)
        # For logging test, use unroll=10 to see structure
        model = make_rnn(cell, unroll=10, grad_checkpoint=grad_checkpoint)

        # Compile the model
        compiled_model = torch.compile(model)

        # Run a forward pass to trigger compilation
        T = 25  # enough for 2 chunks + remainder with unroll=10
        input_data = torch.randn(1, T, 10)
        output, _ = compiled_model(input_data)

        assert output is not None

        # Capture explanation
        # Note: explain analyzes the eager model
        explanation_result = torch._dynamo.explain(model)(input_data)

        explain_file = log_file.with_suffix(".explain.txt")
        explain_file.write_text(str(explanation_result))
        print(f"Explanation saved to: {explain_file}")

        # Reset logs to avoid polluting other tests
        torch._logging.set_logs()

        # Analyze logs
        assert log_file.exists(), "Log file was not created"
        log_content = log_file.read_text()

        print("\n--- Log Analysis ---")
        print(f"Log file size: {len(log_content)} bytes")

        has_dynamo = "torch._dynamo" in log_content or "TorchDynamo" in log_content
        has_inductor = "torch._inductor" in log_content or "Inductor" in log_content

        print(f"Contains Dynamo logs: {has_dynamo}")
        print(f"Contains Inductor logs: {has_inductor}")

        if not has_dynamo:
            print("WARNING: Dynamo logs missing.")
            print(log_content[:1000])

        assert has_dynamo, "Failed to capture Dynamo logs"


@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_compiled_unroll_recompilation(grad_checkpoint):
    """Verify that unrolled chunks are compiled once and reused.

    With T=52 and unroll=10, we expect:
    - 5 chunks of size 10 (compiled once, reused 4 times)
    - 1 chunk of size 2 (compiled once)
    - Total captures for inner loop: 2
    - Plus potentially 1 outer capture if it's partial
    """

    # Clean up any previous compilation stats
    torch._dynamo.reset()

    T = 52
    batch_size = 1
    input_size = 10
    hidden_size = 20
    unroll_size = 10

    cell = SimpleRNNCell(input_size=input_size, hidden_size=hidden_size)
    model = make_rnn(cell, unroll=unroll_size, grad_checkpoint=grad_checkpoint)

    # Check graph measurements using a custom backend
    graph_count = 0

    def count_backend(gm, example_inputs):
        nonlocal graph_count
        graph_count += 1
        return gm.forward

    compiled_model = torch.compile(model, backend=count_backend)

    input_data = torch.randn(batch_size, T, input_size)

    # Run forward pass
    output, states = compiled_model(input_data)

    assert output is not None
    assert output.shape == (batch_size, T, hidden_size)

    print(f"\nTotal graph captures (grad_checkpoint={grad_checkpoint}): {graph_count}")

    # Expectation:
    # 1. Outer trace (partial due to disable) = 1 capture
    # 2. Inner unrolled function (size 10) = 1 capture (reused 5 times)
    # 3. Inner unrolled function (size 2, remainder) = 1 capture
    # Total = 3

    expected_captures = 3
    # Allow small variance if implementation details change (e.g. if outer
    # doesn't capture, or something captures extra)
    # But it strictly shouldn't be >= 5 (recompiling every chunk) + 1 (remainder) = 6+

    assert (
        graph_count <= expected_captures
    ), f"Too many graph captures ({graph_count})! Expected ~{expected_captures}."
    assert graph_count >= 1, "Too few graph captures? Expected at least 1."
