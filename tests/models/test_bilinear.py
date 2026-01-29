import matplotlib.pyplot as plt
import torch

from btorch.models.bilinear import SymmetricBilinear
from btorch.models.constrain import constrain_net
from btorch.utils.file import save_fig


def test_symmetric_bilinear_forward() -> None:
    torch.manual_seed(42)
    in_features = 4
    out_features = 2
    model = SymmetricBilinear(in_features, out_features, bias=True)

    # Random input
    x = torch.randn(10, in_features)

    # Forward pass
    out = model(x)

    # Let's use nn.Bilinear to check against
    ref_bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=True)
    # copy weights
    ref_bilinear.weight.data = model.weight.data.clone()
    if model.bias is not None and ref_bilinear.bias is not None:
        ref_bilinear.bias.data = model.bias.data.clone()

    ref_out = ref_bilinear(x, x)

    torch.testing.assert_close(out, ref_out)

    # Test without bias
    model_no_bias = SymmetricBilinear(in_features, out_features, bias=False)
    assert model_no_bias.bias is None
    out_no_bias = model_no_bias(x)
    assert out_no_bias.shape == (10, out_features)


def test_symmetric_bilinear_constraints():
    torch.manual_seed(42)
    in_features = 5
    out_features = 3

    # 1. Test Masking (float)
    density = 0.3
    model = SymmetricBilinear(
        in_features, out_features, mask=density, enforce_dale=False
    )
    assert model.mask is not None
    assert model.mask.shape == (out_features, in_features, in_features)

    constrain_net(model)
    assert torch.all(model.weight.data[model.mask == 0] == 0)

    # 2. Test Dale's Law
    model_dale = SymmetricBilinear(in_features, out_features, enforce_dale=True)

    # Flip signs
    model_dale.weight.data *= -1
    constrain_net(model_dale)
    assert torch.all(model_dale.weight.data == 0)

    # Visual Verification
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # We'll just show one slice of the bilinear weight
    model_vis = SymmetricBilinear(10, 10, mask=0.2, enforce_dale=True)
    constrain_net(model_vis)

    im0 = axes[0].imshow(
        model_vis.weight.data[0].cpu().numpy(), cmap="RdBu", vmin=-1, vmax=1
    )
    axes[0].set_title("Bilinear Weight Slice 0")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        model_vis.weight.data[1].cpu().numpy(), cmap="RdBu", vmin=-1, vmax=1
    )
    axes[1].set_title("Bilinear Weight Slice 1")
    plt.colorbar(im1, ax=axes[1])

    save_fig(fig, "symmetric_bilinear_constraints", suffix="png")
    plt.close(fig)
