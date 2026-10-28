# Kernel coverage map

Kernel tests use explicit named cases. Each case keeps an independent Python
or module oracle and targets a distinct failure mechanism. Routine validation
does not form a Cartesian product of models, backends, dtypes and options.

## Model Map

| Template | Physical equation oracle | Network or extension coverage |
| --- | --- | --- |
| `if`, `lif`, `glif3`, `izhikevich` | `test_reference_matches_existing_single_step` | `test_fused_psc_then_neuron_receptor_order` |
| `alpha`, `exponential`, `alpha_billeh`, `dual_exponential` | `test_reference_matches_existing_single_step` | `test_fused_psc_then_neuron_receptor_order` |
| `alif`, `elif` | `test_torch_sequence_matches_module_for_reset_and_refractory_options` | `test_native_sequence_preserves_module_full_bptt` |
| `test_external_psc` | `test_external_psc_template_renders_and_matches_full_torch_vjp` | `test_external_psc_template_runs_triton_primitive_and_fused_vjp` |
| GLIF3 + AlphaPSC | `test_public_network_case_matches_python_state_and_weight_vjp` | delayed, heterogeneous whole-network case |
| Izhikevich + DualExponentialPSC | `test_public_network_case_matches_python_state_and_weight_vjp` | heterogeneous whole-network case |

The eight built-in neuron and PSC templates in the first two rows are checked
against existing stateful modules for state ordering, records and one-step
values. ALIF and ELIF have their own adaptation and refractory oracle. The
external row proves that a third-party `ModelTemplate` can provide metadata,
primitive source, fused source and a complete VJP without central dispatch
changes. `ModelParityCase` can represent either a cell or a whole network;
the two public network factories are explicit named cases, not a hidden matrix.

## Failure Mechanisms

| Mechanism | Representative tests |
| --- | --- |
| State continuation and record cotangents | `test_sequence_operator_gradients_and_record_losses`, `test_recurrent_charged_history_losses_and_multidimensional_states` |
| Noncontiguous and aliased parameter storage | `test_fused_cell_broadcast_views_match_eager_forward_and_vjp`, `test_recurrent_module_strided_parameters_match_torch_vjp` |
| Dtype and numerical precision | `test_low_precision_state_accumulation_and_multidimensional_batch`, precision tests under `test_precision.py` |
| Dispatch and trainability | `test_dispatch_tracks_actual_inputs_and_retains_only_needed_history`, parameter-only and frozen-state tests |
| Primitive and fused rendering | `test_renderers.py`, `test_external_psc_template_renders_and_matches_full_torch_vjp` |
| Backend runtime VJP | selected Triton, CuPy and TileLang cases in `test_backends.py` and `test_fused_parameter_layout.py` |
| Compile and capture | named tests in `test_reference.py`, `test_recurrent_checkpoints.py`, and `test_benchmark_capture.py` |
| Whole-network state and weight VJP | `test_public_network_case_matches_python_state_and_weight_vjp` |
| Reset and refractory ordering | `test_torch_sequence_matches_module_for_reset_and_refractory_options`, `test_native_sequence_preserves_module_full_bptt` |
| Delays and heterogeneous receptors | `test_sparse_heter_delay_preserves_initial_ring_adjoint`, `test_native_cudagraph_delayed_sparse_continuation` |
| AMP and capture precision changes | `test_autocast_checkpoint_training_matches_module`, `test_autocast_cudagraph_reselects_precision_and_replays_changed_input` |
| Checkpoint and CPU offload state adjoints | `test_checkpoint_chunks_keep_full_bptt_and_state_records`, `test_native_checkpointed_recurrence_matches_module_vjp` |
| Runtime sequence lengths and time-strided storage | `test_runtime_steps.py` source/cache anchors and GPU numerical cases |
| Charged-state storage ownership and continuation VJP | `test_history_storage.py` standalone and recurrent oracles |
| CPU saved-tensor offload and version checks | `tests/models/rnn/test_history_memory.py` |
| Graph trajectory ownership, keyword drives and stochastic RNG | `tests/models/rnn/test_cudagraph.py` |
