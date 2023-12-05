import os
from pathlib import Path
from typing import Callable

import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa


def squeeze_model(
    base_model,
    input_dims,
    stochastic_depth: float = .5,
    keep_batch_norm: bool = True,
) -> models.Model:
    """
    Generates a squeezed model by converting a base model to a 1D model.

    Args:
        base_model (keras.models.Model): The base model to be squeezed.
        input_dims (tuple): The input dimensions of the model.
        stochastic_depth (float, optional): The probability of dropping out a layer in the model. Defaults to 0.5.
        keep_batch_norm (bool, optional): Whether to keep the batch normalization layers in the model. Defaults to True.

    Returns:
        keras.models.Model: The squeezed model.
    """
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
            layers.BatchNormalization,
        ):
            if print_cond:
                print('adding Batch Normalization layer')
            if keep_batch_norm:
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
            else:
                layer_references[layer.name] = layers.Activation('linear')

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
    return model


def validate_checkpoint_dir(
    experiment_tag: str,
    checkpoint_dir: Path
):
    """
    Validates the checkpoint directory for a given experiment tag.

    Args:
        experiment_tag (str): The tag of the experiment.
        checkpoint_dir (Path): The directory where the checkpoints are stored.

    Returns:
        The path to the checkpoint file.
    """
    checkpoint_path = checkpoint_dir.joinpath(
        f'{experiment_tag}'
    ).joinpath('cp-min_val_loss.ckpt')

    if not checkpoint_path.parent.exists():
        print('creating directory')
        checkpoint_path.parent.mkdir(
            parents=True
        )
    else:
        print('directory already exists')

    return checkpoint_path


def create_checkpoint_callback(
    checkpoint_path: str,
    start_at: float,
):
    """
    Creates a checkpoint callback for saving the best model weights during training.

    Args:
        checkpoint_path (str): The path to the directory where the checkpoint files will be saved.
        start_at (float): The initial value threshold for the monitored metric.

    Returns:
        ModelCheckpoint: The checkpoint callback object.

    """
    callback_checkpointing = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
        monitor='val_root_mean_squared_error',
        mode='min',
        initial_value_threshold=start_at
    )
    return callback_checkpointing


def seed_everything(
    seed_value: int = 42
) -> None:
    """
    Sets the seed value for random number generators used in the program.

    Args:
        seed_value (int): The seed value to be set.

    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(),
        config=session_conf
    )
    tf.compat.v1.keras.backend.set_session(sess)
    return None


def project_layer_to_1d(
    layer_weights_from_2d_model: np.ndarray,
    projection_func: Callable,
    **kwargs: dict,
) -> list:
    """
    Project a 2D layer to a 1D list using a projection function.

    Args:
        layer_weights_from_2d_model (np.ndarray): The weights tensor of the 2D layer conv layer from a pre-trained model.
        projection_func (Callable): The projection function to apply to each kernel.

    Returns:
        list: A list of projected kernels.
    """
    projected_kernels = list()

    kernel_count = layer_weights_from_2d_model.shape[-1]
    channel_count = layer_weights_from_2d_model.shape[-2]

    for kernel_ndx in range(kernel_count):
        for channel_ndx in range(channel_count):
            projected_kernels.append(
                projection_func(
                    layer_weights_from_2d_model[..., channel_ndx, kernel_ndx],
                    **kwargs
                )
            )

    return projected_kernels


def plot_projected_kernels(
    projected_kernels: list,
    plot_title: str,
) -> plt.Figure:
    """
    Generate a figure with subplots to plot a list of projected kernels.

    Args:
        projected_kernels (list): A list of projected kernels to plot.
        plot_title (str): The title of the plot.

    Returns:
        plt.Figure: The generated figure object.
    """

    fig_shape = int(np.sqrt(len(projected_kernels)))
    fig, ax = plt.subplots(
        ncols=fig_shape,
        nrows=fig_shape,
        figsize=(24, 24)
    )

    for i, kernel in enumerate(projected_kernels):
        ax[i // fig_shape, i % fig_shape].plot(kernel)

    fig.suptitle(
        t=plot_title,
        y=1.01,
        x=0,
        horizontalalignment='left',
        fontsize=16
    )
    fig.tight_layout()
    fig.show()

    return fig


def project_to_npcs(
    kernel_2d: np.ndarray,
    n_pc: int,
) -> np.ndarray:
    """
    Projects a 2D kernel onto the first n_pc principal components and averages these two to obtain a 1D kernel.

    Args:
        kernel_2d (np.ndarray): The 2D kernel to be projected.
        n_pc (int): The number of principal components to be used.

    Returns:
        np.ndarray: The mean of the first n_pc principal components.
    """
    pca_of_kernel = PCA()
    return pca_of_kernel\
        .fit_transform(kernel_2d)[:, :n_pc+1]\
        .mean(axis=1)


def project_to_nth_pc(
    kernel_2d: np.ndarray,
    n_pc: int,
) -> np.ndarray:
    """
    Calculates the projection of a 2D kernel onto its first principal component.

    Args:
        kernel_2d (np.ndarray): The 2D kernel to be projected.
        n_pc (int): The ordinal number of the principal component to be used (indexed from 0).

    Returns:
        np.ndarray: The projection of the kernel onto its first principal component.
    """
    pca_of_kernel = PCA()
    return pca_of_kernel\
        .fit_transform(kernel_2d)[:, n_pc]


def project_to_95variance_pcs(
    kernel_2d: np.ndarray,
    var_limit: float = .95,
) -> np.ndarray:
    """
    Projects a given 2D kernel array onto the principal components that explain var_limit % of the variance.

    Args:
        kernel_2d (np.ndarray): The 2D kernel to be projected..
        var_limit (float): The percentage of variance to be explained by the principal components. Defaults to .95.

    Returns:
        np.ndarray: The mean of the principal components that explain var_limit % of the variance.
    """
    pca_of_kernel = PCA()
    transformed_kernel = pca_of_kernel.fit_transform(kernel_2d)
    pc_count = min([
        x
        for x
        in range(1, len(pca_of_kernel.explained_variance_ratio_))
        if pca_of_kernel.explained_variance_ratio_[:x].sum() >= var_limit
    ])

    return transformed_kernel[:, :pc_count]\
        .mean(axis=1)


def project_to_to_x(
    kernel_2d: np.ndarray
) -> np.ndarray:
    """
    Calculate the projection of a 2D kernel onto the x-axis (by taking the column-wise mean).

    Args:
        kernel_2d (ndarray): The 2D kernel to project.

    Returns:
        ndarray: The projection of the kernel onto the x-axis.
    """
    return kernel_2d.mean(axis=0)


def project_to_to_y(
    kernel_2d: np.ndarray
) -> np.ndarray:
    """
    Calculate the projection of a 2D kernel onto the y-axis (by taking the row-wise mean).

    Args:
        kernel_2d: A 2D numpy array representing the kernel.

    Returns:
        A 1D numpy array containing the mean of each row in the kernel.
    """
    return kernel_2d.mean(axis=1)


def project_to_diagonal(
    kernel_2d
) -> np.ndarray:
    """
    Generate the diagonal projection of a 2D kernel.

    Args:
        kernel_2d (numpy.ndarray): The 2D kernel to be projected.

    Returns:
        numpy.ndarray: The diagonal projection of the 2D kernel.
    """
    return np.diagonal(kernel_2d)


def centered_project_to_x(
    kernel_2d: np.ndarray
) -> np.ndarray:
    """
    Generates the centered projection of a 2D kernel onto the x-axis.

    Args:
        kernel_2d (np.ndarray): The input 2D kernel.

    Returns:
        np.ndarray: The transformed kernel, which is the centered projection of the input kernel onto the x-axis.
    """

    transformed_kernel = np.zeros(len(kernel_2d)*2)

    for row_ndx in range(len(kernel_2d)):
        arg_max = np.argmax(kernel_2d[row_ndx, :])
        transformed_kernel[
            (len(kernel_2d)-arg_max):(2*len(kernel_2d)-arg_max)
        ] += kernel_2d[row_ndx, :]

    transformed_kernel = transformed_kernel[
        int(np.ceil(len(kernel_2d) / 2)):(int(np.ceil(len(kernel_2d) / 2)) + len(kernel_2d))
    ]
    return transformed_kernel / len(kernel_2d)


def centered_project_to_y(
    kernel_2d: np.ndarray
) -> np.ndarray:
    """
    Generates a centered projection of a 2D kernel onto the y-axis.

    Args:
        kernel_2d (np.ndarray): The 2D kernel to be transformed.

    Returns:
        np.ndarray: The transformed kernel.

    """
    kernel_2d = kernel_2d.T
    transformed_kernel = np.zeros(len(kernel_2d)*2)

    for row_ndx in range(len(kernel_2d)):
        arg_max = np.argmax(kernel_2d[row_ndx, :])
        transformed_kernel[
            (len(kernel_2d)-arg_max):(2*len(kernel_2d)-arg_max)
        ] += kernel_2d[row_ndx, :]

    transformed_kernel = transformed_kernel[
        int(np.ceil(len(kernel_2d) / 2)):(int(np.ceil(len(kernel_2d) / 2)) + len(kernel_2d))
    ]
    return transformed_kernel / len(kernel_2d)


def save_projected_kernel_fig(
    fig: plt.Figure,
    plot_save_name: str,
    save_dir: Path,
) -> None:
    """
    Saves the projected kernel figure to a file.

    Args:
        fig (plt.Figure): The figure object to save.
        plot_save_name (str): The name of the saved plot.
        save_dir (Path): The directory to save the plot in.

    Returns:
        None
    """
    fig.savefig(
        save_dir.joinpath(f'{plot_save_name}.png')
    )
    return None


def get_2d_conv_layer_names(
    model
) -> list:
    """
    Retrieve the names of all 2D convolutional layers in the given model.

    Args:
        model (object): The model containing the layers.

    Returns:
        list: A list of names of all 2D convolutional layers in the model.
    """

    return [
        layer.name
        for layer
        in model.layers
        if isinstance(layer, layers.Conv2D)
    ]


def project_weights_to_1d(
    model,
    projection_func: Callable,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    Generate the projected weight matrices for each 2D convolutional layer in the model.

    Args:
        model: The model object.
        projection_func (Callable): The function used to project the layer weights to 1D.
        verbose (bool): Whether to print verbose information.

    Returns:
        dict: A dictionary containing the projected weight matrices for each 2D convolutional layer.
    """
    projected_weight_matrices = dict()

    for layer_name in get_2d_conv_layer_names(model):
        projected_weight_matrices[layer_name] = dict()

        weights_2d_conv = model.get_layer(layer_name).get_weights()
        projected_weight_matrices[layer_name]['bias'] = weights_2d_conv[1]

        if layer_name == 'conv1_conv':  # TODO THIS might need adjustments
            weights_2d_conv[0] = weights_2d_conv[0].mean(
                axis=2,
                keepdims=True
            )

        if verbose:
            print('{}; 2D :: {}'.format(
                layer_name,
                weights_2d_conv[0].shape
            ))

        if weights_2d_conv[0].shape[:2] == (1, 1):
            projected_weight_matrices[layer_name]['weights'] = weights_2d_conv[0][0, 0, :, :]
            continue

        projected_weight_matrices[layer_name]['weights'] = project_layer_to_1d(
            layer_weights_from_2d_model=weights_2d_conv[0],
            projection_func=projection_func,
            **kwargs
        )

    return projected_weight_matrices


def assign_projected_weights(
    model,
    projected_weights: dict,
) -> models.Model:
    """
    Assigns projected weights to a given model.

    Args:
        model: The model to assign the projected weights to.
        projected_weights (dict): A dictionary of projected weights, where the keys are layer names and the values are dictionaries with 'weights' and 'bias' keys.

    Returns:
        keras.models.Model: The model with the assigned projected weights.
    """
    model = models.clone_model(model)

    for layer_name in projected_weights:
        print(layer_name)
        print(model.get_layer(layer_name).get_weights()[0].shape)

        new_weights = projected_weights.get(layer_name).get('weights')
        if isinstance(new_weights, list):
            new_weights = np.array(new_weights)

        new_weights = new_weights.T
        new_weights = new_weights.reshape(
            model.get_layer(layer_name).get_weights()[0].shape
        )

        model.get_layer(layer_name).set_weights([
            new_weights,
            projected_weights.get(layer_name).get('bias')
        ])

    return model


def save_model(
    model,
    model_name: str,
    save_dir: Path,
) -> None:
    """
    Save the model with a new name in the specified directory.

    Args:
        model: The model object to be saved.
        model_name (str): The new name for the model.
        save_dir (Path): The directory to save the model in.

    Returns:
        None
    """
    model._name = f'{model.name}_projected_{model_name}'
    model.save(save_dir.joinpath(model.name))
    return None


class RandomMultiplierLayer(Layer):
    def __init__(
        self,
        random_range_min: float,
        random_range_max: float,
        **kwargs
    ) -> None:
        """
        Initializes a RandomMultiplierLayer object.

        Args:
            random_range_min (float): The minimum value for the random range.
            random_range_max (float): The maximum value for the random range.

        Returns:
            None
        """
        super(RandomMultiplierLayer, self).__init__(**kwargs)
        self.min_val = random_range_min
        self.max_val = random_range_max

    def call(
        self,
        inputs,
        training: bool,
    ):
        """
        Applies a random multiplier to the inputs if training is set to True.

        Args:
            inputs: The input tensor.
            training (bool): Whether the model is in training mode or not.

        Returns:
            The multiplied inputs if training is True, otherwise the inputs are returned as is.
        """
        if not training:
            return inputs
        random_multiplier = uniform(
            [1],
            minval=self.min_val,
            maxval=self.max_val
        )
        return multiply(inputs, 1+random_multiplier)


class RandomMask(Layer):
    def __init__(
        self,
        size: int,
        **kwargs,
    ) -> None:
        """
        Initialize the RandomMask object.

        Parameters:
            size (int): The size of the object.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super(RandomMask, self).__init__(**kwargs)
        self.size = size

    def call(
        self,
        inputs,
        training: bool,
    ):
        """
        Generates the function comment for the given function body.

        Args:
            inputs (ndarray): The input array.
            training (bool): Whether the model is in training mode.

        Returns:
            ndarray: The masked input array.
        """
        if not training:
            return inputs
        index = uniform(
            shape=[],
            minval=0,
            maxval=inputs.shape[1]-self.size,
            dtype=dtypes.int32,
            seed=None,
            name=None
        )
        mask = concat(
            [ones(
                [index],
                dtype=dtypes.float32
            ),
                zeros(
                [self.size],
                dtype=dtypes.float32
            ),
                ones(
                [inputs.shape[1] - index - self.size],
                dtype=dtypes.float32
            )],
            axis=0
        )

        return multiply(inputs, mask)
