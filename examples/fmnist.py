# modified from spikingjelly's mnist example
# TODO: check the training performance

import argparse
import datetime
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets
from brevitas.nn import QuantLinear
from torch.cuda import amp
from torch.utils.tensorboard.writer import SummaryWriter

from btorch.models import (
    functional,
    glif,
    init,
    rnn,
    synapse,
)
from btorch.models.regularizer import VoltageRegularizer


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
        # these can be numpy or float, automatically convert to Tensor
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
            update_state_names=("neuron.v", "synapse.psc"),  # neuron.Iasc
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


def main():
    parser = argparse.ArgumentParser(description="Classify Sequential Fashion-MNIST")
    parser.add_argument("-device", default="cuda:0", help="device")
    parser.add_argument("-b", default=128, type=int, help="batch size")
    parser.add_argument(
        "-epochs",
        default=64,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument("-data-dir", type=str, help="root dir of Fashion-MNIST dataset")
    parser.add_argument(
        "-out-dir",
        type=str,
        default="./logs",
        help="root dir for saving logs and checkpoint",
    )
    parser.add_argument("-resume", type=str, help="resume from the checkpoint path")
    parser.add_argument(
        "-amp", action="store_true", help="automatic mixed precision training"
    )
    parser.add_argument(
        "-opt", type=str, default="adam", help="use which optimizer. SGD or Adam"
    )
    parser.add_argument("-momentum", default=0.9, type=float, help="momentum for SGD")
    parser.add_argument("-lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("-T", default=28, type=int, help="simulating time-steps")
    parser.add_argument(
        "-rate-out", action="store_true", help="use rate-based output (spike count)"
    )
    parser.add_argument(
        "-enable-quant", action="store_true", help="enable quantized weights"
    )
    parser.add_argument(
        "-weight-bitwidth", default=8, type=int, help="bit-width for quantized weights"
    )
    parser.add_argument(
        "-model", default="glifnet", type=str, help="model name for logging"
    )

    args = parser.parse_args()
    print(args)

    net = GLifNet(
        network_config=NetworkConfig(),
        T=args.T,
        device=args.device,
        rate_out=args.rate_out,
        enable_quant=args.enable_quant,
        weight_bitwidth=args.weight_bitwidth,
        num_input=28,  # Input is per-row
        num_output=10,
    )

    train_set = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    test_set = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True,
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True,
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.lr, momentum=args.momentum
        )
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(checkpoint["net"])
        functional.set_memory_reset_values(net, checkpoint["memories_rv"])
        functional.init_net_state(net)
        if "memories" in checkpoint:
            functional.set_memory_values(net, checkpoint["memories"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        max_test_acc = checkpoint["max_test_acc"]
    else:
        functional.init_net_state(net)

    net.to(args.device)
    print(net)

    out_dir = os.path.join(
        args.out_dir, f"{args.model}_b{args.b}_{args.opt}_lr{args.lr}"
    )

    if args.amp:
        out_dir += "_amp"

    if args.enable_quant:
        out_dir += f"_quant{args.weight_bitwidth}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Mkdir {out_dir}.")

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as args_txt:
        args_txt.write(str(args))
        args_txt.write("\n")
        args_txt.write(" ".join(sys.argv))

    # init net state
    # fixed but randomized initial v
    # init.uniform_v_(net.brain.neuron, rand_batch=False, set_reset_value=True)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            # img.shape = [N, 1, H, W]
            img.squeeze_(1)  # [N, H, W]
            img = img.permute(2, 0, 1)  # [W, N, H]
            batch_size = img.shape[1]  # N
            # we regard [W, N, H] as [T, N, H]
            label_onehot = F.one_hot(label, 10).float()

            # keep same initial state
            functional.reset_net(net, batch_size=batch_size)
            # or random as an regulariser
            init.uniform_v_(net.brain.neuron, rand_batch=True)

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(img)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()
            # if you use SparseConn from models.linear, and enforce Dale's law
            # constrain_net(net)

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                # img.shape = [N, 1, H, W]
                img.squeeze_(1)  # [N, H, W]
                img = img.permute(2, 0, 1)  # [W, N, H]
                # we regard [W, N, H] as [T, N, H]

                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            "net": net.state_dict(),
            # if you wish to store v, and Iasc state,
            # "memories": functional.named_memory_values(net),
            "memories_rv": functional.named_memory_reset_values(net),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "max_test_acc": max_test_acc,
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, "checkpoint_max.pth"))

        torch.save(checkpoint, os.path.join(out_dir, "checkpoint_latest.pth"))

        print(args)
        print(out_dir)
        print(
            f"epoch = {epoch}, train_loss ={train_loss: .4f}, "
            f"train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, "
            f"test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}"
        )
        print(
            f"train speed ={train_speed: .4f} images/s, "
            f"test speed ={test_speed: .4f} images/s"
        )
        remaining = (time.time() - start_time) * (args.epochs - epoch)
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
        print(f"escape time = {eta.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
