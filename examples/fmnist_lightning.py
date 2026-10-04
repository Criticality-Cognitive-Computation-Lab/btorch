# modified from spikingjelly's mnist example

from dataclasses import asdict, dataclass, field
from typing import Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
from brevitas.nn import QuantLinear
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from btorch.models import (
    functional,
    glif,
    init,
    rnn,
    synapse,
)
from btorch.models.regularizer import VoltageRegularizer
from btorch.models.scale import scale_state_


@dataclass
class NeuronParam:
    v_threshold: float = -45.0  # mV
    v_reset: float = -60.0  # mV
    c_m: float = 2.0  # pfarad
    tau: float = 20.0  # ms
    k: list[float] = field(default_factory=lambda: [1.0 / 80])  # ms^-1
    asc_amps: list[float] = field(default_factory=lambda: [-0.2])  # pA
    tau_ref: float = 2.0  # ms
    hard_reset: bool = False


@dataclass
class NetworkConfig:
    """Network configuration parameters."""

    n_neuron: int = 256
    num_input_neuron: int = 128
    num_output_neuron: int = 20
    n_e_ratio: float = 0.8

    scale: bool = True
    base_current: float = 0.12  # .5
    input_weight_scaling: float = 1.0
    input_scaling: float = 1.0
    rec_weight_scaling: float = 5.0
    damping_factor: float = 1.0
    surrogate_function: str = "Erf"  # ATan

    # Neuron parameters
    neuron_param: NeuronParam = field(default_factory=NeuronParam)

    # Synapse parameters
    synapse_params: dict = field(
        default_factory=lambda: {"tau_syn": {"E": 5.8, "I": 6.5}}
    )

    _seed: int = 32

    def __post_init__(self):
        """Initialize network topology."""
        self.n_e_neuron = int(self.n_neuron * self.n_e_ratio)
        self.n_i_neuron = self.n_neuron - self.n_e_neuron
        self.num_input_neuron = min(self.num_input_neuron, self.n_e_neuron)

        # Setup synapse parameters
        self.synapse_params_concrete = {
            "tau_syn": np.concatenate(
                [
                    np.ones(self.n_e_neuron) * self.synapse_params["tau_syn"]["E"],
                    np.ones(self.n_i_neuron) * self.synapse_params["tau_syn"]["I"],
                ]
            )
        }

        # Generate random indices
        rand = np.random.RandomState(self._seed)
        self.input_indices = rand.choice(
            self.n_e_neuron, self.num_input_neuron, replace=False
        )
        self.output_indices = rand.choice(
            self.n_e_neuron, self.num_output_neuron, replace=False
        )

    def get_neuron_args(self, scale: bool | None = None):
        scale = scale or self.scale
        ret = {
            **asdict(self.neuron_param),
        }
        scale_state_(ret)
        return ret

    def get_synapse_args(self, scale: bool | None = None):
        scale = scale or self.scale
        ret = {
            **self.synapse_params_concrete,
        }
        return ret


class GLifNet(nn.Module):
    """Spiking neural network for MNIST classification."""

    input_indices: torch.Tensor

    def __init__(
        self,
        network_config: NetworkConfig,
        T: int,
        device: str,
        rate_out: bool = True,
        enable_quant: bool = False,
        weight_bitwidth: int = 8,
        num_input: int = 784,
        num_output: int = 10,
        dtype=torch.float32,
    ):
        super().__init__()
        self.network_config = network_config
        self.T = T
        self.rate_out = rate_out
        self.enable_quant = enable_quant
        self.weight_bitwidth = weight_bitwidth

        # Build spiking neuron layer
        self._build_spiking_layer(device, dtype)

        # Build input/output layers
        self._build_input_layer(num_input, device, dtype)
        self._build_output_layer(num_output, device, dtype)

        # Initialize voltage regularization
        self.voltage_reg = VoltageRegularizer(
            network_config.neuron_param.v_threshold,
            network_config.neuron_param.v_reset,
            voltage_cost=1.0,
        )

    def _build_spiking_layer(self, device, dtype):
        """Build the core spiking neural network layer."""
        n_neuron = self.network_config.n_neuron

        # Create GLIF neuron
        neuron_module = glif.GLIF3(
            n_neuron=n_neuron,
            v_threshold=self.network_config.neuron_param.v_threshold,
            v_reset=self.network_config.neuron_param.v_reset,
            c_m=self.network_config.neuron_param.c_m,
            tau=self.network_config.neuron_param.tau,
            k=self.network_config.neuron_param.k,
            asc_amps=self.network_config.neuron_param.asc_amps,
            tau_ref=self.network_config.neuron_param.tau_ref,
            hard_reset=self.network_config.neuron_param.hard_reset,
            detach_reset=False,
            step_mode="s",
            backend="torch",
        )

        # Create recurrent connections
        if self.enable_quant:
            conn = QuantLinear(
                n_neuron,
                n_neuron,
                weight_bitwidth=self.weight_bitwidth,
                bias=False,
                device=device,
                dtype=dtype,
            )
        else:
            conn = nn.Linear(n_neuron, n_neuron, bias=False, device=device, dtype=dtype)

        # Create synaptic layer
        psc_module = synapse.AlphaPSCBilleh(
            n_neuron,
            tau_syn=self.network_config.synapse_params["tau_syn"],
            linear=conn,
            step_mode="s",
        )

        # Create recurrent layer
        self.brain = rnn.RecurrentNN(
            neuron=neuron_module,
            synapse=psc_module,
            step_mode="m",
            update_state_names=("neuron.v", "synapse.psc"),
            grad_checkpoint=False,  # enable if you have large vram issue
        )

    def _build_input_layer(self, num_input: int, device, dtype):
        """Build input projection layer."""
        if self.enable_quant:
            self.fc1 = QuantLinear(
                num_input,
                self.network_config.num_input_neuron,
                weight_bitwidth=self.weight_bitwidth,
                act_quant=None,
                bias=False,
                device=device,
                dtype=dtype,
            )
        else:
            self.fc1 = nn.Linear(
                num_input,
                self.network_config.num_input_neuron,
                bias=False,
                device=device,
                dtype=dtype,
            )

        self.register_buffer(
            "input_indices",
            torch.tensor(
                self.network_config.input_indices, dtype=torch.long, device=device
            ),
        )

    def _build_output_layer(self, num_output: int, device, dtype):
        """Build output readout layer."""
        if self.enable_quant:
            self.fc2 = QuantLinear(
                self.network_config.num_output_neuron,
                num_output,
                act_quant=None,
                bias=False,
                device=device,
                dtype=dtype,
            )
        else:
            self.fc2 = nn.Linear(
                self.network_config.num_output_neuron,
                num_output,
                bias=False,
                device=device,
                dtype=dtype,
            )

        self.register_buffer(
            "output_indices",
            torch.tensor(
                self.network_config.output_indices, dtype=torch.long, device=device
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Input projection
        x = self.fc1(x)

        # Create input for brain
        v: torch.Tensor = self.brain.neuron.v
        inp = torch.zeros(
            (x.shape[0],) + v.shape,
            device=v.device,
            dtype=v.dtype,
        )
        inp[..., self.input_indices] = x

        # Process through brain
        spike, states = self.brain(inp)

        # Output readout
        if self.rate_out:
            x = self.fc2(spike[..., self.output_indices].mean(dim=0))
        else:
            x = self.fc2(states["neuron.v"][..., self.output_indices])

        return x


class FashionMNISTDataModule(L.LightningDataModule):
    """DataModule to wrap the FashionMNIST dataloaders."""

    def __init__(
        self, data_dir: str = "./data", batch_size: int = 128, num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        torchvision.datasets.FashionMNIST(root=self.data_dir, train=True, download=True)
        torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_set = torchvision.datasets.FashionMNIST(
                root=self.data_dir,
                train=True,
                transform=self.transform,
                download=False,
            )
            self.val_set = torchvision.datasets.FashionMNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform,
                download=False,
            )
        if stage in (None, "test"):
            self.test_set = torchvision.datasets.FashionMNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform,
                download=False,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class GLifLightningModule(L.LightningModule):
    """Lightning module wrapping the GLifNet and training logic."""

    def __init__(
        self,
        network_config: NetworkConfig = NetworkConfig(),
        T: int = 28,
        device: str = "cpu",
        rate_out: bool = True,
        enable_quant: bool = False,
        weight_bitwidth: int = 8,
        num_input: int = 28,
        num_output: int = 10,
        opt: str = "adam",
        lr: float = 0.001,
        momentum: float = 0.9,
        epochs: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["network_config"])
        self.model = GLifNet(
            network_config=network_config,
            T=T,
            device=device,
            rate_out=rate_out,
            enable_quant=enable_quant,
            weight_bitwidth=weight_bitwidth,
            num_input=num_input,
            num_output=num_output,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _prepare_state(self, batch_size: int, random_init: bool):
        functional.reset_net(self.model, batch_size=batch_size)
        if random_init:
            init.uniform_v_(self.model.brain.neuron, rand_batch=True)

    def _cleanup_state(self):
        functional.reset_net(self.model)

    def _shared_step(self, batch, stage: str):
        img, label = batch
        img = img.squeeze(1)  # [N, H, W]
        img = img.permute(2, 0, 1)  # [W, N, H]
        batch_size = img.shape[1]
        self._prepare_state(batch_size=batch_size, random_init=stage == "train")

        label_onehot = F.one_hot(label, 10).float()
        out_fr = self(img)
        loss = self.criterion(out_fr, label_onehot)
        acc = (out_fr.argmax(dim=1) == label).float().mean()

        self._cleanup_state()
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=stage == "train",
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        if self.hparams.opt == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum
            )
        elif self.hparams.opt == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError(self.hparams.opt)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.hparams.epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
        }

    def on_fit_start(self):
        functional.init_net_state(self.model)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["memories_rv"] = functional.named_memory_reset_values(self.model)
        checkpoint["memories"] = functional.named_memory_values(self.model)

    def on_load_checkpoint(self, checkpoint):
        if "memories_rv" in checkpoint:
            functional.set_memory_reset_values(self.model, checkpoint["memories_rv"])
        if "memories" in checkpoint:
            functional.set_memory_values(self.model, checkpoint["memories"])


class TrainCLI(LightningCLI):
    """CLI that auto-runs fit then test with the best checkpoint."""

    def __init__(self):
        super().__init__(
            GLifLightningModule,
            FashionMNISTDataModule,
            seed_everything_default=None,
            save_config_overwrite=True,
            trainer_defaults={
                "max_epochs": 64,
                "log_every_n_steps": 1,
                "logger": TensorBoardLogger(save_dir="./logs", name="glifnet"),
                "callbacks": [
                    ModelCheckpoint(
                        monitor="val_acc",
                        mode="max",
                        save_top_k=1,
                        save_last=True,
                        filename="epoch{epoch:02d}-val_acc{val_acc:.4f}",
                    ),
                    LearningRateMonitor(logging_interval="epoch"),
                ],
            },
        )

    def after_fit(self):
        self.trainer.test(ckpt_path="best")


def main():
    TrainCLI()


if __name__ == "__main__":
    main()
