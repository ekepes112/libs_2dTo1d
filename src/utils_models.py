from tensorflow.keras import layers, models
import tensorflow_addons as tfa


def squeeze_model(
    base_model,
    input_dims,
    stochastic_depth: float = .5
):
    print_cond = False

    layer_references = {}

    layer_references[base_model.layers[0].name] = layers.Input(input_dims)

    for layer in base_model.layers[1:]:
        if print_cond:
            print(layer.name)
        # if print_cond: print(layer_references)

        inbound_connections = []

        for inbound_layer, _, _, _ in layer\
                ._inbound_nodes[0]\
                .iterate_inbound():

            if print_cond:
                print(inbound_layer.name)
            inbound_connections.append(inbound_layer.name)

        if isinstance(
            layer,
            layers.Conv2D
        ):

            if print_cond:
                print('adding Conv1D layer')
            layer_references[layer.name] = layers.Conv1D(
                filters=layer.filters,
                kernel_size=layer.kernel_size[0],
                padding=layer.padding,
                name=layer.name,
                strides=layer.strides[0],
                data_format=layer.data_format,
                dilation_rate=layer.dilation_rate[0],
                groups=layer.groups,
                activation=layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint
            )(
                layer_references[inbound_connections[0]]
            )

        if isinstance(
            layer,
            layers.Activation
        ):

            if print_cond:
                print('adding Activation layer')
            layer_references[layer.name] = layers.Activation(
                layer.activation,
                name=layer.name
            )(
                layer_references[inbound_connections[0]]
            )

        if isinstance(
            layer,
            layers.Dense
        ):

            print('adding Dense layer')
            dropout_layer = layers.Dropout(.2)(
                layer_references[inbound_connections[0]]
            )
            layer_references[layer.name] = layers.Dense(
                # units = layer.units,
                units=1,
                name=layer.name,
                # activation = layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint
            )(
                dropout_layer
            )

        if isinstance(
            layer,
            layers.Add
        ):

            if print_cond:
                print('adding Add layer')
            if stochastic_depth > 0:
                # https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth
                layer_references[layer.name] = tfa.layers.StochasticDepth(
                    survival_probability=stochastic_depth,
                    name=layer.name
                )([
                    layer_references[connection]
                    for connection
                    in inbound_connections
                ])
            else:
                layer_references[layer.name] = layers.Add(
                    name=layer.name
                )([
                    layer_references[connection]
                    for connection
                    in inbound_connections
                ])

        if isinstance(
            layer,
            layers.BatchNormalization
        ):

            if print_cond:
                print('adding Batch Normalization layer')
            layer_references[layer.name] = layers.BatchNormalization(
                # axis=layer.axis
                axis=-1,
                name=layer.name,
                momentum=layer.momentum,
                epsilon=layer.epsilon,
                center=layer.center,
                scale=layer.scale,
                beta_initializer=layer.beta_initializer,
                gamma_initializer=layer.gamma_initializer,
                moving_mean_initializer=layer.moving_mean_initializer,
                moving_variance_initializer=layer.moving_variance_initializer,
                beta_regularizer=layer.beta_regularizer,
                gamma_regularizer=layer.gamma_regularizer,
                beta_constraint=layer.beta_constraint,
                gamma_constraint=layer.gamma_constraint
            )(
                layer_references[inbound_connections[0]]
            )

        if isinstance(
            layer,
            layers.GlobalAveragePooling2D
        ):

            if print_cond:
                print('adding Global Average Pooling layer')
            layer_references[layer.name] = layers.GlobalAveragePooling1D(
                name=layer.name
            )(
                layer_references[inbound_connections[0]]
            )

        if isinstance(
            layer,
            layers.MaxPooling2D
        ):

            if print_cond:
                print('adding Max Pooling layer')
            layer_references[layer.name] = layers.MaxPooling1D(
                name=layer.name,
                pool_size=layer.pool_size[0],
                strides=layer.strides[0],
                padding=layer.padding,
                data_format=layer.data_format
            )(
                layer_references[inbound_connections[0]]
            )

        if isinstance(
            layer,
            layers.ZeroPadding2D
        ):

            if print_cond:
                print('adding Zero Padding layer')
            layer_references[layer.name] = layers.ZeroPadding1D(
                padding=layer.padding[0],
                name=layer.name
            )(
                layer_references[inbound_connections[0]]
            )

    model = models.Model(
        layer_references[base_model.layers[0].name],
        layer_references[base_model.layers[-1].name],
        name=base_model.name
    )
    return (model)
