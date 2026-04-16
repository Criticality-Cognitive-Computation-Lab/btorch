# btorch.connectome

## Connectome Data Handling

::: btorch.connectome
    options:
      members:
        - simple_id_to_root_id

## Connection Matrices

::: btorch.connectome.connection
    options:
      members:
        - make_sparse_mat
        - neuron_subset_to_conn_mat

## Connection Conversion

::: btorch.models.connection_conversion
    options:
      members:
        - heter2base
        - base2heter
        - convert_connection_layer
        - convert_connection_layer_from_checkpoint
