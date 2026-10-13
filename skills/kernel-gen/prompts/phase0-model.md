# Phase 0: model and scientific response

Work from the user's model description or provided Torch Module. Read the task repository and btorch modelling conventions. Describe state, parameters, units, tensor axes, dt, event order, initialization and continuation. For stochastic/plastic mechanisms, describe RNG ownership, traces and weight-update timing.

Use the actual model/network code to run single neuron, connected pair and network protocols. Include no-input, impulse, step-on/off, sustained drive, delay and model-relevant limits. Save the numerical checks, raw states and all three activity figures. Show all relevant state channels and first unexpected event/state.

Use the installed Humanize skill for planning and review; this prompt supplies only the btorch scientific requirements. Ask the user to confirm a newly generated or modified model using these actual outputs before freezing it as golden. Do not silently alter its equations to make a kernel match.
