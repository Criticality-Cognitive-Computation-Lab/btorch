from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import pandas as pd
import torch

from btorch.models import environ
from btorch.models.neurons.glif import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.shape import expand_leading_dims
from btorch.models.surrogate import Erf
from btorch.models.synapse import AlphaPSC
from btorch.utils.dict_utils import unflatten_dict


def get_simple_id(df: pd.DataFrame) -> np.ndarray:
    return df.simple_id.to_numpy()


class NeuronEmbedMapLayer(torch.nn.Module):
    def __init__(
        self,
        neurons: pd.DataFrame,
        scale: torch.Tensor | float | None = None,
    ):
        super().__init__()
        self.neuron_embed_map: dict[str, Any] = {}
        self.n_neuron = len(neurons)
        self.scale = torch.as_tensor(scale) if scale is not None else None
        self.define_default(neurons)

    @overload
    def register_neuron_embed(
        self, key: dict, embed: None = ..., neuron_id: None = ..., *args, override=False
    ): ...
    @overload
    def register_neuron_embed(
        self, key: str, embed: torch.nn.Module, neuron_id, *args, override=False
    ): ...
    def register_neuron_embed(
        self, key, embed=None, neuron_id=None, *args, override=False
    ):
        if isinstance(key, dict):
            dict_k_emb_arg = key
            for k, emb_arg in dict_k_emb_arg.items():
                self.register_neuron_embed(k, *emb_arg, override)
            return

        assert override or key not in self.neuron_embed_map
        # assume neuron_id_name is never modified afterwards,
        # so storing a referencce is safe
        self.neuron_embed_map[key] = (embed, neuron_id, *args)
        # so that embed appears in self.modules
        setattr(self, key, embed)

    def define_default(self, neurons):
        pass


# remeber to scale the input weight by neuron scale
class EnvInputLayer(NeuronEmbedMapLayer):
    def define_default(self, neurons):
        from ...connectome import neuron_population

        vision_neuron_id = np.hstack(
            [
                get_simple_id(v)
                for v in neuron_population.get_optics(neurons, ("R7", "R8")).values()
            ]
        )
        vision_embed = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=len(vision_neuron_id)),
            # torch.nn.LayerNorm(len(vision_neuron_id)),
            torch.nn.ReLU(),
        )
        self.register_neuron_embed("vision", vision_embed, vision_neuron_id)
        johnston_neuron_id = np.hstack(
            [
                get_simple_id(v)
                for v in neuron_population.get_johnston(neurons, ("C", "E")).values()
            ]
        )
        johnston_embed = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=len(johnston_neuron_id)),
            # torch.nn.LayerNorm(len(johnston_neuron_id)),
            torch.nn.ReLU(),
        )
        if self.scale is not None:
            johnston_embed[0].weight.data /= self.scale
        self.register_neuron_embed("wind_gravity", johnston_embed, johnston_neuron_id)
        an_neuron_id = get_simple_id(neuron_population.get_AN(neurons))
        an_embed = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=len(an_neuron_id)),
            # torch.nn.LayerNorm(len(an_neuron_id)),
            torch.nn.ReLU(),
        )
        self.register_neuron_embed("an", an_embed, an_neuron_id)

        fake_target_neuron_id = get_simple_id(
            neuron_population.fake_target_encoding_neuron(neurons)
        )
        ft_embed = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=len(fake_target_neuron_id)),
            # torch.nn.LayerNorm(len(an_neuron_id)),
            torch.nn.ReLU(),
        )
        self.register_neuron_embed("fake_target", ft_embed, fake_target_neuron_id)

    def forward(self, observ: dict[str, Any]):
        # TODO: merge these into one scatter
        all_srcs = []
        all_indices = []
        batch_shape = None
        device = None
        dtype = None

        for k, v in observ.items():
            embed, neuron_id = self.neuron_embed_map[k]
            src = embed(v)
            if batch_shape is None:
                batch_shape = src.shape[:-1]
                device = src.device
                dtype = src.dtype

            neuron_id = torch.tensor(neuron_id, dtype=torch.long, device=device)
            index = expand_leading_dims(neuron_id, batch_shape)

            all_srcs.append(src)
            all_indices.append(index)

        if not all_srcs:
            return None

        merged_src = torch.cat(all_srcs, dim=-1)
        merged_index = torch.cat(all_indices, dim=-1)

        ret = torch.zeros(
            batch_shape + (self.n_neuron,),
            device=device,
            dtype=dtype,
        )
        ret = ret.scatter_add(dim=-1, index=merged_index, src=merged_src)
        return ret


class DetectionWindow(torch.nn.Module):
    def __init__(self, pred_window: float | int | None):
        super().__init__()
        self.pred_window = pred_window

    def forward(self, x):
        pred_window = self.pred_window
        if pred_window is None:
            return x
        if isinstance(pred_window, float):
            pred_window = int(x.shape[0] * pred_window)
        x = x[-pred_window:, ...]
        return x


class EnvOutputLayer(NeuronEmbedMapLayer):
    def __init__(
        self,
        neurons: pd.DataFrame,
        pred_window: float | int | None = None,
        rate: bool = False,
        scale: torch.Tensor | float | None = None,
    ):
        super().__init__(neurons, scale)
        self.rate = rate
        self.pred_window = DetectionWindow(pred_window)

    def define_default(self, neurons):
        from ...connectome import neuron_population

        dn_neuron_id = get_simple_id(neuron_population.get_DN(neurons))
        dn_embed = torch.nn.Identity()
        self.register_neuron_embed("dn", dn_embed, dn_neuron_id, "neuron.v")
        mbon_neuron_id = get_simple_id(neuron_population.get_mbon(neurons))
        mbon_embed = torch.nn.Identity()
        self.register_neuron_embed("mbon", mbon_embed, mbon_neuron_id, "neuron.v")

    def forward(self, x: dict[str, Any], output_choice: Sequence[str] | None = None):
        # TODO: merge these into one scatter
        ret = {}
        x["neuron.v"] = self.pred_window(x["neuron.v"])
        x["neuron.spike"] = self.pred_window(x["neuron.spike"])
        if self.rate:
            x["neuron.rate"] = (
                x["neuron.spike"].mean(dim=0) * environ.get("dt") * 1e3
            )  # Hz
        if output_choice is None:
            output_choice = tuple(self.neuron_embed_map.keys())
        # Group by out_attr to merge gathers
        grouped_requests = {}
        for name in output_choice:
            embed, neuron_id, out_attr = self.neuron_embed_map[name]
            if out_attr not in grouped_requests:
                grouped_requests[out_attr] = []
            grouped_requests[out_attr].append((name, embed, neuron_id))

        for out_attr, requests in grouped_requests.items():
            # requests is list of (name, embed, neuron_id)
            if not requests:
                continue

            names, embeds, neuron_ids = zip(*requests)

            # If only one item, simple path? (Optional optimization, but strictly
            # following 'merged')
            # Keeping consistent merged path:

            # Ensure neuron_ids are tensors on correct device
            # Note: We assume x[out_attr] is on the device we want to compute on.
            device = x[out_attr].device
            neuron_ids_tensors = [
                torch.as_tensor(nid, device=device, dtype=torch.long)
                for nid in neuron_ids
            ]
            split_sections = [nid.shape[0] for nid in neuron_ids_tensors]

            all_ids = torch.cat(neuron_ids_tensors)

            # Gather
            gathered_all = x[out_attr][..., all_ids]

            # Split
            gathered_splits = torch.split(gathered_all, split_sections, dim=-1)

            # Apply embeds
            for i, name in enumerate(names):
                ret[name] = embeds[i](gathered_splits[i])

        return ret


class Brain(torch.nn.Module):
    def __init__(
        self,
        synapse_args: dict,
        neuron_args: dict,
        neuron_module_type: type = GLIF3,
        input_layer: torch.nn.Module | None = None,
        output_layer: torch.nn.Module | None = None,
    ):
        super().__init__()

        # neurons
        neuron = neuron_module_type(
            **{
                "v_rest": -52.0,  # mV
                "v_threshold": -50.0,  # mV
                "v_reset": -60.0,  # mV
                "tau": 10.0,  # ms
                "hard_reset": True,
                "surrogate_function": Erf(variance=0.5, damping_factor=1.0),
                **neuron_args,
            },
        )
        synapse = AlphaPSC(n_neuron=neuron_args["n_neuron"], **synapse_args)
        self.brain = RecurrentNN(
            neuron=neuron,
            synapse=synapse,
            update_state_names=("neuron.v", "synapse.psc"),
            step_mode="m",
        )
        self.input_layer = input_layer
        self.output_layer = output_layer

    def forward(self, observ: torch.Tensor | dict[str, Any], output_choice=None):
        if self.input_layer is not None:
            inp = self.input_layer(observ)
        else:
            inp = observ
        spike, states = self.brain(inp)
        states["neuron.spike"] = spike

        if self.output_layer is not None:
            out = self.output_layer(states, output_choice)
        else:
            out = None

        # states = functional.unscale_state(self.brain, states, enforce="ignore")
        brain_out = unflatten_dict(states, dot=True)

        return out, brain_out
