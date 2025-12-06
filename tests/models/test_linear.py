import scipy.sparse
import torch
from btorch.models.constrain import constrain_net
from btorch.models.linear import DenseConn, SparseConn, SparseConstrainedConn


def test_equivalent_behavior():
    """Test that all three connection classes behave the same for identical
    weights."""
    torch.manual_seed(42)

    # Create a small dense weight matrix
    W = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]])  # 3x3 matrix

    # Test input
    x = torch.tensor([1.0, 2.0, 3.0])

    # 1. Dense connection
    dense = DenseConn(3, 3, weight=W, bias=None)

    # 2. Sparse COO connection (convert dense to sparse)
    W_sparse = scipy.sparse.coo_array(W.numpy())
    sparse_coo = SparseConn(W_sparse, bias=None, enforce_dale=False)

    # 3. Constrained sparse connection (each weight is its own group)
    # Create constraint matrix where each non-zero gets unique group ID
    constraint_data = []
    constraint_rows = []
    constraint_cols = []
    group_id = 1

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] != 0:
                constraint_data.append(group_id)
                constraint_rows.append(i)
                constraint_cols.append(j)
                group_id += 1

    constraint = scipy.sparse.coo_array(
        (constraint_data, (constraint_rows, constraint_cols)), shape=W.shape
    )

    constrained = SparseConstrainedConn(
        W_sparse, constraint, enforce_dale=False, bias=None
    )

    # Forward pass
    out_dense = dense(x)
    out_sparse = sparse_coo(x)
    out_constrained = constrained(x)

    print("Dense output:", out_dense)
    print("Sparse output:", out_sparse)
    print("Constrained output:", out_constrained)

    # Check they're all the same
    assert torch.allclose(out_dense, out_sparse, atol=1e-6)
    assert torch.allclose(out_dense, out_constrained, atol=1e-6)
    print("✓ All three methods produce identical outputs")


def test_constraint_optimization():
    """Test that constraints work correctly in gradient optimization."""
    torch.manual_seed(42)

    # Create weight matrix where some weights should be tied together
    W_sparse = scipy.sparse.coo_array(
        ([1.0, -2, -3, 1], ([0, 1, 0, 1], [0, 0, 1, 1])),
        shape=(2, 2),
    )

    # Create constraint matrix: positions (0,0) and (1,1) share group 1
    # positions (0,1) and (1,0) have separate groups
    constraint = scipy.sparse.coo_array(
        ([1, 2, 3, 1], ([0, 0, 1, 1], [0, 1, 0, 1])),  # groups: 1,2,3,1
        shape=(2, 2),
    )

    # Test both enable_dale=True and enable_dale=False
    for enable_dale in [False, True]:
        print(f"\n{'='*60}")
        print(f"Testing with enable_dale={enable_dale}")
        print(f"{'='*60}")

        model = SparseConstrainedConn(
            W_sparse, constraint, enforce_dale=enable_dale, bias=None
        )
        constrain_net(model)

        # Target: we want the output to be [10, 20] for input [1, 1]
        x = torch.tensor([1.0, 1.0])
        target = torch.tensor([10.0, 20.0])

        print("Initial weights:")
        print(
            "Group 1 magnitude (affects positions (0,0) and (1,1)):",
            model.magnitude[0].item(),
        )
        print("Group 2 magnitude (affects position (0,1)):", model.magnitude[1].item())
        print("Group 3 magnitude (affects position (1,0)):", model.magnitude[2].item())

        # Check initial effective weights and signs
        magnitudes = model.magnitude[model._constraint_scatter_indices]
        effective_weights = model.initial_weight * magnitudes

        print("Initial effective weight matrix:")
        print(f"  [{effective_weights[0]:.3f}, {effective_weights[1]:.3f}]")
        print(f"  [{effective_weights[2]:.3f}, {effective_weights[3]:.3f}]")

        # Store initial signs for Dale's law verification
        if enable_dale:
            initial_signs = torch.sign(effective_weights)
            print("Initial signs for Dale's law verification:")
            print(f"  [{initial_signs[0]:.0f}, {initial_signs[1]:.0f}]")
            print(f"  [{initial_signs[2]:.0f}, {initial_signs[3]:.0f}]")

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(10):  # Increased epochs for better convergence
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            constrain_net(model)

            # Check Dale's law during optimization if enabled
            if enable_dale and epoch % 2 == 0:
                current_magnitudes = model.magnitude[model._constraint_scatter_indices]
                current_effective = model.initial_weight * current_magnitudes
                current_signs = torch.sign(current_effective)

                print(
                    f"Epoch {epoch}: loss={loss.item():.4f}, "
                    f"output={output.detach().numpy()}"
                )
                print(
                    f"  Current signs: [{current_signs[0]:.0f}, "
                    f"{current_signs[1]:.0f}], [{current_signs[2]:.0f},"
                    f" {current_signs[3]:.0f}]"
                )
            elif not enable_dale and epoch % 2 == 0:
                print(
                    f"Epoch {epoch}: loss={loss.item():.4f}, "
                    f"output={output.detach().numpy()}"
                )

        print("\nFinal weights:")
        print("Group 1 magnitude:", model.magnitude[0].item())
        print("Group 2 magnitude:", model.magnitude[1].item())
        print("Group 3 magnitude:", model.magnitude[2].item())

        # Verify constraints and Dale's law
        final_magnitudes = model.magnitude[model._constraint_scatter_indices]
        final_weights = model.initial_weight * final_magnitudes

        print("\nFinal effective weight matrix:")
        print(f"  [{final_weights[0]:.3f}, {final_weights[1]:.3f}]")
        print(f"  [{final_weights[2]:.3f}, {final_weights[3]:.3f}]")

        # Test 1: Constraint verification
        # - positions (0,0) and (1,1) should have same effective value
        pos_00_effective = final_weights[0]  # 1.0 * magnitude[0]
        pos_11_effective = final_weights[3]  # 1.0 * magnitude[0]
        assert torch.allclose(pos_00_effective, pos_11_effective, atol=1e-6), (
            f"Constraint failed: pos(0,0)={pos_00_effective:.6f} "
            f"!= pos(1,1)={pos_11_effective:.6f}"
        )
        print("✓ Constraint verified: tied weights remain equal during optimization")

        # Test 2: Dale's law verification
        # - signs should be preserved when enable_dale=True
        if enable_dale:
            final_signs = torch.sign(final_weights)
            print("\nDale's law verification:")
            print(
                f"Initial signs: [{initial_signs[0]:.0f}, "
                f"{initial_signs[1]:.0f}], [{initial_signs[2]:.0f}, "
                f"{initial_signs[3]:.0f}]"
            )
            print(
                f"Final signs:   [{final_signs[0]:.0f}, "
                f"{final_signs[1]:.0f}], [{final_signs[2]:.0f}, {final_signs[3]:.0f}]"
            )

            # Check that signs are preserved
            # (allowing for zero weights to have any sign)
            for i in range(len(initial_signs)):
                if (
                    abs(model.initial_weight[i]) > 1e-8
                ):  # Only check non-zero initial weights
                    initial_sign = initial_signs[i].item()
                    final_sign = final_signs[i].item()
                    assert initial_sign == final_sign or abs(final_weights[i]) < 1e-8, (
                        f"Dale's law violated at position {i}: "
                        f"initial_sign={initial_sign}, final_sign={final_sign}, "
                        f"initial_weight={model.initial_weight[i]:.6f}, "
                        f"final_weight={final_weights[i]:.6f}"
                    )

            print(
                "✓ Dale's law verified: effective weights maintain "
                "same signs as initial weights"
            )
        else:
            print("✓ Dale's law not enforced (enable_dale=False)")

    print(f"\n{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Testing equivalent behavior...")
    test_equivalent_behavior()
    print("\nTesting constraint optimization...")
    test_constraint_optimization()
    print("\nAll tests passed! ✓")
