import jax
import matplotlib.pyplot as plt
from pathlib import Path

# Enable float64 before creating any JAX arrays or model parameters.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
from flax import linen as nn
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.tensorboard import SummaryWriter
import datetime
from SDEs import (
    MomentumHASME,
    ComplexMomentumPF,
    MomentumFirstOrderFlow,
    MomentumSecondOrderFlow,
    DEBUG,
)
from SDEs import (
    MomentumHASME,
    ComplexMomentumPF,
    MomentumFirstOrderFlow,
    MomentumSecondOrderFlow,
    MomentumDALOptimizer,
    DEBUG,
)
import numpy as np
from torchvision import datasets
from functools import partial
import argparse


# ==========================================================
# Command-line arguments
# ==========================================================

parser = argparse.ArgumentParser(
    description="Run momentum optimization experiments."
)

parser.add_argument(
    "--recurrence_local_errors",
    action="store_true",
    help=(
        "Compute the two-sided local GDM recurrence error "
        "using Phi(+eta) + mu*Phi(-eta), together with a "
        "the signed reconstructed recurrence loss error"
    ),
)

parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random initialization seed",
)

parser.add_argument(
    "--gpu",
    action="store_true",
    help="Enable GPU acceleration",
)

parser.add_argument(
    "--load_pretrained",
    action="store_true",
    help="Load pretrained weights",
)

parser.add_argument(
    "--mu",
    type=float,
    default=0.0,
    help="Momentum coefficient; must satisfy 0 <= mu < 1",
)

parser.add_argument(
    "--eta",
    type=float,
    default=1e-1,
    help="Learning rate",
)

parser.add_argument(
    "--eta_dt_ratio",
    type=float,
    default=5e1,
    help="Ratio eta / dt",
)

parser.add_argument(
    "--sigma",
    type=float,
    default=1e-4,
    help="Noise standard deviation",
)

parser.add_argument(
    "--num_steps",
    type=int,
    default=10,
    help="Number of optimization steps",
)

parser.add_argument(
    "--num_runs",
    type=int,
    default=100,
    help="Number of Monte Carlo runs",
)

parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="TensorBoard log directory",
)

parser.add_argument(
    "--pretrained_weights",
    type=int,
    default=1000,
    help="Pretrained checkpoint number",
)

parser.add_argument(
    "--hidden_widths",
    type=int,
    nargs="+",
    default=[5, 5],
    help=(
        "Widths of all hidden layers. For example, "
        "'--hidden_widths 10 10 10 10' constructs four "
        "hidden layers of width 10."
    ),
)

parser.add_argument(
    "--activation",
    type=str,
    choices=(
        "tanh",
        "sigmoid",
        "silu",
        "softplus",
    ),
    default="silu",
    help="Smooth hidden-layer activation function.",
)

parser.add_argument(
    "--brownian_tol_ratio",
    type=float,
    default=0.1,
    help="Set VirtualBrownianTree tolerance to this value times dt.",
)

parser.add_argument(
    "--plot_dir",
    type=str,
    default="plots",
    help="Directory for pointwise weak-error plots.",
)

parser.add_argument(
    "--problem",
    type=str,
    choices=("nn", "quadratic"),
    default="quadratic",
    help=(
        "Objective used in the experiment. "
        "Use 'nn' for Iris and 'quadratic' for the "
        "two-dimensional quadratic test."
    ),
)

parser.add_argument(
    "--quad_h11",
    type=float,
    default=3.0,
)

parser.add_argument(
    "--quad_h12",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--quad_h22",
    type=float,
    default=2.0,
)

parser.add_argument(
    "--quad_target_1",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--quad_target_2",
    type=float,
    default=0.6,
)

parser.add_argument(
    "--quad_init_1",
    type=float,
    default=2.8,
)

parser.add_argument(
    "--quad_init_2",
    type=float,
    default=3.5,
)

parser.add_argument(
    "--sde_batch_size",
    type=int,
    default=100,
    help=(
        "Number of SDE trajectories solved in parallel. "
        "Use a smaller value if memory is limited."
    ),
)

parser.add_argument(
    "--hessian_diagnostic_only",
    action="store_true",
    help=(
        "Initialize the selected objective, compute the "
        "initial Hessian spectrum, print the result, and exit "
        "without running SGDM or the SDE models."
    ),
)

parser.add_argument(
    "--complex_pf",
    action="store_true",
    help=(
        "Include the experimental complex momentum principal "
        "flow. It is deterministic for --discrete_method gdm "
        "and stochastic for --discrete_method sgdm."
    ),
)

parser.add_argument(
    "--discrete_method",
    type=str,
    choices=(
        "gdm",
        "sgdm",
    ),
    default="gdm",
    help=(
        "Reference momentum method: "
        "'gdm' for deterministic gradient descent with "
        "momentum, or 'sgdm' for stochastic gradient "
        "descent with momentum."
    ),
)

parser.add_argument(
    "--local_error_start_step",
    type=int,
    default=1,
    help=(
        "First GDM step included in the Rosca Figure-10-style "
        "local-error plots."
    ),
)

parser.add_argument(
    "--local_error_end_step",
    type=int,
    default=None,
    help=(
        "Last GDM step included in the Rosca Figure-10-style "
        "local-error plots. Defaults to --num_steps."
    ),
)

parser.add_argument(
    "--fig13_gdm",
    action="store_true",
    help=(
        "Produce a Rosca Figure-13-style diagnostic for GDM: "
        "local negative/positive momentum-gradient flows, "
        "lambda_0, loss, and the leading stability coefficient."
    ),
)

parser.add_argument(
    "--fig13_start_step",
    type=int,
    default=1,
    help=(
        "First GDM transition shown in the Figure-13-style plot."
    ),
)

parser.add_argument(
    "--fig13_end_step",
    type=int,
    default=None,
    help=(
        "Last GDM transition shown in the Figure-13-style plot. "
        "Defaults to --num_steps."
    ),
)

parser.add_argument(
    "--fig13_flow_substeps",
    type=int,
    default=0,
    help=(
        "Euler substeps used for each local negative/positive "
        "gradient-flow solve. A value of 0 uses the rounded "
        "--eta_dt_ratio."
    ),
)

parser.add_argument(
    "--momentum_dal_fig19",
    action="store_true",
    help=(
        "Run a cheap full-batch Iris analogue of Rosca "
        "Figure 19 for GD with momentum. Compares fixed "
        "learning-rate GDM with momentum-adjusted DAL and "
        "then exits without running the SME experiments."
    ),
)

parser.add_argument(
    "--dal_fixed_etas",
    type=float,
    nargs="+",
    default=[
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
    ],
    help=(
        "Fixed GDM learning rates included in the "
        "Figure-19-style comparison."
    ),
)

parser.add_argument(
    "--dal_target_ratio",
    type=float,
    default=1.0,
    help=(
        "Target intrinsic-drift-to-gradient-signal ratio tau. "
        "tau=1 gives the M-DAL rule derived in the text."
    ),
)

parser.add_argument(
    "--dal_min_eta",
    type=float,
    default=1e-8,
    help="Minimum effective M-DAL learning rate.",
)

parser.add_argument(
    "--dal_max_eta",
    type=float,
    default=0.5,
    help=(
        "Maximum effective M-DAL learning rate. This is a "
        "numerical safeguard analogous to the learning-rate "
        "cap in the original DAL implementation."
    ),
)

parser.add_argument(
    "--dal_curvature_epsilon",
    type=float,
    default=1e-12,
    help=(
        "Positive floor applied to ||H g_hat|| in the M-DAL "
        "denominator."
    ),
)

parser.add_argument(
    "--architecture",
    type=str,
    choices=(
        "nn",
        "quadratic",
        "vgg",
        "resnet-18",
        "resnet-50",
    ),
    default="nn",
    help="Model architecture.",
)

parser.add_argument(
    "--dataset",
    type=str,
    choices=(
        "iris",
        "cifar-10",
        "imagenet",
    ),
    default="iris",
    help="Dataset used for the experiment.",
)

parser.add_argument(
    "--data_dir",
    type=str,
    default=None,
    help="Root directory for datasets requiring local files, e.g. ImageNet.",
)

parser.add_argument(
    "--max_train_samples",
    type=int,
    default=None,
    help=(
        "Maximum number of training samples to use. "
        "Use None for the full dataset."
    ),
)

parser.add_argument(
    "--max_test_samples",
    type=int,
    default=None,
    help=(
        "Maximum number of test samples to use. "
        "Use None for the full dataset."
    ),
)

parser.add_argument(
    "--figure13_only",
    action="store_true",
    help=(
        "Run only the algorithms required for Figure 13; "
        "do not run Momentum-HA-SME or Momentum-SME-2."
    ),
)



args = parser.parse_args()


# ==========================================================
# Platform configuration
# ==========================================================

if args.gpu:
    print("GPU enabled.")
    jax.config.update("jax_platform_name", "gpu")


# ==========================================================
# Experiment parameters
# ==========================================================

if args.figure13_only:
    args.fig13_gdm = True

if (
    args.recurrence_local_errors
    and args.discrete_method != "gdm"
):
    raise ValueError(
        "--recurrence_local_errors requires "
        "--discrete_method gdm."
    )

if not 0.0 < args.mu < 1.0:
    raise ValueError(
        "Momentum experiments require "
        "--mu to satisfy 0 < mu < 1."
    )

if args.eta <= 0.0:
    raise ValueError(
        "--eta must be positive"
    )

if args.eta_dt_ratio <= 0.0:
    raise ValueError(
        "--eta_dt_ratio must be positive"
    )

if args.num_steps <= 0:
    raise ValueError(
        "--num_steps must be positive"
    )

if args.num_runs <= 0:
    raise ValueError(
        "--num_runs must be positive"
    )

if args.sigma < 0.0:
    raise ValueError(
        "--sigma must be nonnegative"
    )

if args.sde_batch_size <= 0:
    raise ValueError(
        "--sde_batch_size must be positive"
    )

if len(args.hidden_widths) == 0:
    raise ValueError(
        "--hidden_widths must contain at least one width."
    )

if any(width <= 0 for width in args.hidden_widths):
    raise ValueError(
        "Every value passed to --hidden_widths must be positive. "
        f"Received: {args.hidden_widths}"
    )

if args.brownian_tol_ratio <= 0.0:
    raise ValueError(
        "--brownian_tol_ratio must be positive"
    )

if args.local_error_start_step < 1:
    raise ValueError(
        "--local_error_start_step must be at least 1."
    )

if (
    args.local_error_end_step is not None
    and args.local_error_end_step
    < args.local_error_start_step
):
    raise ValueError(
        "--local_error_end_step must be greater than or equal "
        "to --local_error_start_step."
    )

if args.fig13_start_step < 1:
    raise ValueError(
        "--fig13_start_step must be at least 1."
    )

if (
    args.fig13_end_step is not None
    and args.fig13_end_step
    < args.fig13_start_step
):
    raise ValueError(
        "--fig13_end_step must be greater than or equal to "
        "--fig13_start_step."
    )

if args.fig13_flow_substeps < 0:
    raise ValueError(
        "--fig13_flow_substeps must be nonnegative."
    )

if (
    args.fig13_gdm
    and args.discrete_method != "gdm"
):
    raise ValueError(
        "--fig13_gdm requires --discrete_method gdm."
    )

if args.momentum_dal_fig19:
    if args.problem != "nn":
        raise ValueError(
            "--momentum_dal_fig19 requires --problem nn. "
            "The selected cheap experiment uses the full-batch "
            "Iris neural-network objective."
        )

    if args.discrete_method != "gdm":
        raise ValueError(
            "--momentum_dal_fig19 requires "
            "--discrete_method gdm."
        )

    if len(args.dal_fixed_etas) == 0:
        raise ValueError(
            "--dal_fixed_etas must contain at least one value."
        )

    if any(
        value <= 0.0
        for value in args.dal_fixed_etas
    ):
        raise ValueError(
            "Every value in --dal_fixed_etas must be positive."
        )

    if args.dal_target_ratio <= 0.0:
        raise ValueError(
            "--dal_target_ratio must be positive."
        )

    if args.dal_min_eta <= 0.0:
        raise ValueError(
            "--dal_min_eta must be positive."
        )

    if args.dal_max_eta <= 0.0:
        raise ValueError(
            "--dal_max_eta must be positive."
        )

    if args.dal_max_eta < args.dal_min_eta:
        raise ValueError(
            "--dal_max_eta must be at least --dal_min_eta."
        )

    if args.dal_curvature_epsilon <= 0.0:
        raise ValueError(
            "--dal_curvature_epsilon must be positive."
        )


eta = args.eta
mu = args.mu
dt = eta / args.eta_dt_ratio
num_steps = args.num_steps

requested_sigma = args.sigma
requested_num_runs = args.num_runs


if args.discrete_method == "gdm":
    # Deterministic momentum GD.
    sigma = 0.0
    num_runs = 1
    stochastic_setting = False

    reference_name = (
        "Gradient Descent with Momentum"
    )

    reference_short_name = "GDM"

elif args.discrete_method == "sgdm":
    if args.sigma <= 0.0:
        raise ValueError(
            "--discrete_method sgdm requires --sigma > 0. "
            f"Received args.sigma={args.sigma}."
        )

    sigma = args.sigma
    num_runs = args.num_runs
    stochastic_setting = True

    reference_name = (
        "Stochastic Gradient Descent with Momentum"
    )

    reference_short_name = "SGDM"

else:
    raise ValueError(
        f"Unknown method: {args.discrete_method}"
    )


if DEBUG:
    num_steps = 1
    num_runs = 1


brownian_tol = (
    args.brownian_tol_ratio * dt
)

final_sde_time = (
    num_steps * eta
)


print(
    "Experiment configuration:"
)

print(
    "  reference method:",
    reference_name,
)

print(
    "  complex PF enabled:",
    args.complex_pf,
)

print(
    "  stochastic setting:",
    stochastic_setting,
)

print(
    "  eta:",
    eta,
)

print(
    "  mu:",
    mu,
)

print(
    "  effective sigma:",
    sigma,
)

print(
    "  effective runs:",
    num_runs,
)

print(
    "  integration dt:",
    dt,
)

print(
    "  final continuous time:",
    final_sde_time,
)
# This is diagnostic only. Experiment length is controlled
# exclusively by num_steps.
final_sde_time = num_steps * eta

print("Experiment time configuration:")
print("  eta:", eta)
print("  SDE dt:", dt)
print("  Brownian tolerance:", brownian_tol)
print("  number of SGDM steps:", num_steps)
print("  number of Monte Carlo runs:", num_runs)
print("  resulting final SDE time:", final_sde_time)


REFERENCE_ALGORITHM = (
    "Discrete-Reference"
)

COMPLEX_PF_ALGORITHM = (
    "Complex-Momentum-PF"
)

algorithms = [
    REFERENCE_ALGORITHM,
]

if args.figure13_only:
    # Figure 13 does not require HA-SME or SME-2.

    # Include SME-1 if Figure 13 uses it.
    algorithms.append(
        "Momentum-SME-1"
    )

else:
    if args.complex_pf:
        # Complex PF replaces the real-valued Momentum HA-SME.
        algorithms.append(
            COMPLEX_PF_ALGORITHM
        )

    else:
        # Run the real-valued HA-SME when complex PF is off.
        algorithms.append(
            "Momentum-HA-SME"
        )

    # SME-1 and SME-2 are included in the full experiment.
    algorithms.extend(
        [
            "Momentum-SME-1",
            "Momentum-SME-2",
        ]
    )

print(
    "Algorithms selected:",
    algorithms,
)

if DEBUG:
    algorithms = [
        REFERENCE_ALGORITHM,
    ]

    if args.figure13_only:
        algorithms.append(
            "Momentum-SME-1"
        )

    elif args.complex_pf:
        algorithms.append(
            COMPLEX_PF_ALGORITHM
        )

    else:
        algorithms.append(
            "Momentum-HA-SME"
        )

display_names = {
    REFERENCE_ALGORITHM: (
        reference_name
    ),
    COMPLEX_PF_ALGORITHM: (
        "Complex Momentum PF"
    ),
    "Momentum-HA-SME": (
        "Real Momentum HA-SME"
    ),
    "Momentum-SME-1": (
        "Momentum SME-1"
    ),
    "Momentum-SME-2": (
        "Momentum SME-2"
    ),
}


def load_cifar10(
    data_dir=None,
    max_train_samples=None,
    max_test_samples=None,
):
    """
    Load CIFAR-10 and return NumPy arrays suitable for the
    JAX/Flax training code.

    Returns
    -------
    X_train : ndarray
        Shape (N_train, 32, 32, 3), values in [0, 1].

    y_train : ndarray
        Shape (N_train, 10), one-hot encoded labels.

    X_test : ndarray
        Shape (N_test, 32, 32, 3), values in [0, 1].

    y_test : ndarray
        Shape (N_test, 10), one-hot encoded labels.
    """

    if data_dir is None:
        data_dir = "./data"

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
    )

    # ---------------------------------------------------------
    # Images
    # ---------------------------------------------------------

    X_train = np.asarray(
        train_dataset.data,
        dtype=np.float64,
    ) / 255.0

    X_test = np.asarray(
        test_dataset.data,
        dtype=np.float64,
    ) / 255.0

    # ---------------------------------------------------------
    # Integer labels
    # ---------------------------------------------------------

    y_train = np.asarray(
        train_dataset.targets,
        dtype=np.int32,
    )

    y_test = np.asarray(
        test_dataset.targets,
        dtype=np.int32,
    )

    # ---------------------------------------------------------
    # Limit the number of samples, if requested
    # ---------------------------------------------------------

    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError(
                "--max_train_samples must be positive "
                "when specified."
            )

        X_train = X_train[:max_train_samples]
        y_train = y_train[:max_train_samples]

    if max_test_samples is not None:
        if max_test_samples <= 0:
            raise ValueError(
                "--max_test_samples must be positive "
                "when specified."
            )

        X_test = X_test[:max_test_samples]
        y_test = y_test[:max_test_samples]

    # ---------------------------------------------------------
    # One-hot encode labels
    # ---------------------------------------------------------

    y_train = jax.nn.one_hot(
        y_train,
        10,
        dtype=jnp.float64,
    )

    y_test = jax.nn.one_hot(
        y_test,
        10,
        dtype=jnp.float64,
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )

# ==========================================================
# Load and preprocess Iris
# ==========================================================

if args.dataset == "iris":
    iris = load_iris()

    X = iris.data
    y = iris.target.reshape(-1, 1)

    encoder = OneHotEncoder(
        sparse_output=False
    )

    y_onehot = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_onehot,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

elif args.dataset == "cifar-10":
    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = load_cifar10(
        data_dir=args.data_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

elif args.dataset == "imagenet":
    raise NotImplementedError(
        "ImageNet loading has not been implemented yet."
    )

else:
    raise ValueError(
        f"Unknown dataset: {args.dataset}"
    )
# Explicitly convert all data to float64.
X_train = jnp.asarray(
    X_train,
    dtype=jnp.float64,
)

X_test = jnp.asarray(
    X_test,
    dtype=jnp.float64,
)

y_train = jnp.asarray(
    y_train,
    dtype=jnp.float64,
)

y_test = jnp.asarray(
    y_test,
    dtype=jnp.float64,
)

def apply_smooth_activation(
    x,
    activation,
):
    is_complex = jnp.issubdtype(
        x.dtype,
        jnp.complexfloating,
    )

    if activation == "tanh":
        return jnp.tanh(x)

    if activation == "sigmoid":
        if is_complex:
            return (
                1.0
                / (
                    1.0
                    + jnp.exp(-x)
                )
            )

        return jax.nn.sigmoid(x)

    if activation == "silu":
        if is_complex:
            sigmoid_x = (
                1.0
                / (
                    1.0
                    + jnp.exp(-x)
                )
            )

            return x * sigmoid_x

        return jax.nn.silu(x)

    if activation == "softplus":
        if is_complex:
            # Principal complex logarithm. This is holomorphic
            # away from logarithmic branch cuts and zeros of
            # 1 + exp(x).
            return jnp.log1p(
                jnp.exp(x)
            )

        return jax.nn.softplus(x)

    raise ValueError(
        f"Unknown activation: {activation}"
    )


class ConvBlock(nn.Module):
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="SAME",
            use_bias=False,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        x = nn.GroupNorm(
            num_groups=32,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        return nn.relu(x)

class VGG16(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        # VGG-16:
        # 64, 64
        # 128, 128
        # 256, 256, 256
        # 512, 512, 512
        # 512, 512, 512

        for features in (64, 64):
            x = ConvBlock(features=features)(x)

        x = nn.max_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
        )

        for features in (128, 128):
            x = ConvBlock(features=features)(x)

        x = nn.max_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
        )

        for features in (256, 256, 256):
            x = ConvBlock(features=features)(x)

        x = nn.max_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
        )

        for features in (512, 512, 512):
            x = ConvBlock(features=features)(x)

        x = nn.max_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
        )

        for features in (512, 512, 512):
            x = ConvBlock(features=features)(x)

        x = nn.max_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
        )

        # Global average pooling keeps the model compatible
        # with different image resolutions.
        x = jnp.mean(
            x,
            axis=(1, 2),
        )

        x = nn.Dense(
            features=self.num_classes,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        return x

class ResNetBasicBlock(nn.Module):
    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        residual = x

        y = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        y = nn.GroupNorm(
            num_groups=32,
            dtype=None,
            param_dtype=jnp.float64,
        )(y)

        y = nn.relu(y)

        y = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=None,
            param_dtype=jnp.float64,
        )(y)

        y = nn.GroupNorm(
            num_groups=32,
            dtype=None,
            param_dtype=jnp.float64,
        )(y)

        if (
            self.stride != 1
            or residual.shape[-1] != self.features
        ):
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                dtype=None,
                param_dtype=jnp.float64,
            )(residual)

            residual = nn.GroupNorm(
                num_groups=32,
                dtype=None,
                param_dtype=jnp.float64,
            )(residual)

        return nn.relu(
            y + residual
        )

class ResNet18(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        # Standard ResNet stem.
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        x = nn.GroupNorm(
            num_groups=32,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        x = nn.relu(x)

        x = nn.max_pool(
            x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )

        # ResNet-18: [2, 2, 2, 2].
        for features, num_blocks, stride in (
            (64, 2, 1),
            (128, 2, 2),
            (256, 2, 2),
            (512, 2, 2),
        ):
            x = ResNetBasicBlock(
                features=features,
                stride=stride,
            )(x)

            for _ in range(num_blocks - 1):
                x = ResNetBasicBlock(
                    features=features,
                    stride=1,
                )(x)

        x = jnp.mean(
            x,
            axis=(1, 2),
        )

        x = nn.Dense(
            features=self.num_classes,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        return x

class ResNetBottleneckBlock(nn.Module):
    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        residual = x

        y = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        y = nn.GroupNorm(
            num_groups=32,
            dtype=None,
            param_dtype=jnp.float64,
        )(y)

        y = nn.relu(y)

        y = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            dtype=None,
            param_dtype=jnp.float64,
        )(y)

        y = nn.GroupNorm(
            num_groups=32,
            dtype=None,
            param_dtype=jnp.float64,
        )(y)

        y = nn.relu(y)

        y = nn.Conv(
            features=4 * self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=None,
            param_dtype=jnp.float64,
        )(y)

        y = nn.GroupNorm(
            num_groups=32,
            dtype=None,
            param_dtype=jnp.float64,
        )(y)

        if (
            self.stride != 1
            or residual.shape[-1] != 4 * self.features
        ):
            residual = nn.Conv(
                features=4 * self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                dtype=None,
                param_dtype=jnp.float64,
            )(residual)

            residual = nn.GroupNorm(
                num_groups=32,
                dtype=None,
                param_dtype=jnp.float64,
            )(residual)

        return nn.relu(
            y + residual
        )


class ResNet50(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        # Standard ResNet stem.
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        x = nn.GroupNorm(
            num_groups=32,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        x = nn.relu(x)

        x = nn.max_pool(
            x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )

        # ResNet-50: [3, 4, 6, 3].
        for features, num_blocks, stride in (
            (64, 3, 1),
            (128, 4, 2),
            (256, 6, 2),
            (512, 3, 2),
        ):
            x = ResNetBottleneckBlock(
                features=features,
                stride=stride,
            )(x)

            for _ in range(num_blocks - 1):
                x = ResNetBottleneckBlock(
                    features=features,
                    stride=1,
                )(x)

        x = jnp.mean(
            x,
            axis=(1, 2),
        )

        x = nn.Dense(
            features=self.num_classes,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        return x






# ==========================================================
# Neural network
# ==========================================================

class SimpleNN(nn.Module):
    hidden_widths: tuple = (5, 5)
    activation: str = "tanh"
    num_classes: int = 3

    @nn.compact
    def __call__(self, x):
        if x.ndim > 2:
            x = x.reshape(
                x.shape[0],
                -1,
            )

        for width in self.hidden_widths:
            x = nn.Dense(
                features=width,
                dtype=None,
                param_dtype=jnp.float64,
            )(x)

            x = apply_smooth_activation(
                x,
                self.activation,
            )

        x = nn.Dense(
            features=self.num_classes,
            dtype=None,
            param_dtype=jnp.float64,
        )(x)

        return x




# ==========================================================
# Initial model structure
# ==========================================================

key = jax.random.PRNGKey(args.seed)

input_shape = X_train[0].shape
hidden_widths = tuple(
    args.hidden_widths
)

if args.dataset == "iris":
    num_classes = 3

elif args.dataset == "cifar-10":
    num_classes = 10

elif args.dataset == "imagenet":
    num_classes = 1000

else:
    raise ValueError(
        f"Unknown dataset: {args.dataset}"
    )

if args.architecture == "nn":
    model = SimpleNN(
        hidden_widths=hidden_widths,
        activation=args.activation,
        num_classes=num_classes,
    )

elif args.architecture == "vgg":
    model = VGG16(
        num_classes=num_classes,
    )

elif args.architecture == "resnet-18":
    model = ResNet18(
        num_classes=num_classes,
    )

elif args.architecture == "resnet-50":
    model = ResNet50(
        num_classes=num_classes,
    )

elif args.architecture == "quadratic":
    model = None

else:
    raise ValueError(
        f"Unknown architecture: {args.architecture}"
    )

if args.problem == "quadratic":

    params = None

    flat_params = jnp.asarray(
        [
            args.quad_init_1,
            args.quad_init_2,
        ],
        dtype=jnp.float64,
    )

    def unflatten_fn(x):
        return x

    d = 2

else:

    params = model.init(
        key,
        jnp.ones(
            input_shape,
            dtype=jnp.float64,
        ),
    )["params"]

    flat_params, unflatten_fn = ravel_pytree(
        params
    )

    d = flat_params.shape[0]

flat_params, unflatten_fn = ravel_pytree(params)
d = flat_params.shape[0]


# The covariance automatically uses the state's datatype.
def cov_mat_func(x):
    return (
        sigma**2
        * jnp.eye(
            x.shape[0],
            dtype=x.dtype,
        )
    )


# ==========================================================
# Loss
# ==========================================================

@partial(jax.jit, static_argnums=0)
def compute_loss(
    unflatten_fn,
    flat_params,
    inputs,
    labels,
):
    parameter_tree = unflatten_fn(flat_params)

    logits = model.apply(
        {"params": parameter_tree},
        inputs,
    )

    return jnp.mean(
        optax.softmax_cross_entropy(
            logits,
            labels,
        )
    )

@partial(
    jax.jit,
    static_argnums=0,
)
def compute_complex_loss(
    unflatten_fn,
    flat_params,
    inputs,
    labels,
):
    """
    Holomorphic extension of softmax cross-entropy.

    This deliberately does not use optax's stabilized real
    log-softmax, because operations such as max are not
    holomorphic on complex inputs.

    The principal complex logarithm is used.
    """

    parameter_tree = unflatten_fn(
        flat_params
    )

    complex_inputs = inputs.astype(
        flat_params.dtype
    )

    complex_labels = labels.astype(
        flat_params.dtype
    )

    logits = model.apply(
        {
            "params": parameter_tree
        },
        complex_inputs,
    )

    # Use the first logit as an analytic reference shift.
    # This avoids a non-holomorphic maximum operation.
    reference_logit = logits[
        :,
        :1,
    ]

    shifted_logits = (
        logits
        - reference_logit
    )

    log_partition = (
        reference_logit
        + jnp.log(
            jnp.sum(
                jnp.exp(
                    shifted_logits
                ),
                axis=-1,
                keepdims=True,
            )
        )
    )

    log_probabilities = (
        logits
        - log_partition
    )

    sample_losses = -jnp.sum(
        complex_labels
        * log_probabilities,
        axis=-1,
    )

    return jnp.mean(
        sample_losses
    )


# ==========================================================
# Select the experimental objective
# ==========================================================

if args.problem == "nn":
    loss_fn_flat = lambda x: compute_loss(
        unflatten_fn,
        x,
        X_train,
        y_train,
    )

    complex_loss_fn_flat = lambda z: (
        compute_complex_loss(
            unflatten_fn,
            z,
            X_train,
            y_train,
        )
    )

    quadratic_hessian = None
    quadratic_target = None

elif args.problem == "quadratic":
    quadratic_hessian = jnp.asarray(
        [
            [
                args.quad_h11,
                args.quad_h12,
            ],
            [
                args.quad_h12,
                args.quad_h22,
            ],
        ],
        dtype=jnp.float64,
    )

    quadratic_target = jnp.asarray(
        [
            args.quad_target_1,
            args.quad_target_2,
        ],
        dtype=jnp.float64,
    )

    quadratic_eigenvalues = (
        jnp.linalg.eigvalsh(
            quadratic_hessian
        )
    )

    if not bool(
        jnp.all(
            quadratic_eigenvalues > 0.0
        )
    ):
        raise ValueError(
            "The quadratic Hessian must be positive definite. "
            f"Eigenvalues: {quadratic_eigenvalues}"
        )

    @jax.jit
    def loss_fn_flat(x):
        displacement = (
            x - quadratic_target
        )
        
        return (
            0.5
            * displacement
            @ quadratic_hessian
            @ displacement
        )

    @jax.jit
    def complex_loss_fn_flat(z):
        # This is a holomorphic extension. Do not use
        # conjugation or jnp.vdot here.
        complex_hessian = (
            quadratic_hessian.astype(
                z.dtype
            )
        )

        complex_target = (
            quadratic_target.astype(
                z.dtype
            )
        )

        displacement = (
            z - complex_target
        )

        return (
            0.5
            * displacement
            @ complex_hessian
            @ displacement
        )

    print("Neural-network architecture:")

    print(
        "  number of hidden layers:",
        len(hidden_widths),
    )

    print(
        "  hidden widths:",
        hidden_widths,
    )
    print(
        "  activation:",
        args.activation,
    )
    print(
        "  parameter dimension:",
        d,
    )

    print(
        "Quadratic Hessian:"
    )
    print(
        np.asarray(
            quadratic_hessian
        )
    )

    print(
        "Quadratic Hessian eigenvalues:",
        np.asarray(
            quadratic_eigenvalues
        ),
    )

    print(
        "Quadratic target:",
        np.asarray(
            quadratic_target
        ),
    )

else:
    raise ValueError(
        f"Unknown problem: {args.problem}"
    )

@jax.jit
def complex_objective_observable(z):
    # The complex PF objective may itself be complex.
    # Its real part agrees with the real objective on the
    # real parameter subspace.
    return jnp.real(
        complex_loss_fn_flat(z)
    )


grad_loss = jax.jit(
    jax.grad(loss_fn_flat)
)

def get_fig13_flow_substeps():
    """
    Select the number of explicit Euler steps used to
    integrate each local comparison flow for physical time eta.
    """

    if args.fig13_flow_substeps > 0:
        return args.fig13_flow_substeps

    return max(
        1,
        int(
            np.ceil(
                eta / dt
            )
        ),
    )


fig13_flow_substeps = (
    get_fig13_flow_substeps()
)


@jax.jit
def fig13_local_momentum_gradient_flow_endpoint(
    initial_x,
    direction,
):
    """
    Integrate the local momentum-gradient flow for physical
    time eta using explicit Euler.

    direction = -1:
        negative gradient flow

    direction = +1:
        positive gradient flow
    """

    local_dt = jnp.asarray(
        eta / fig13_flow_substeps,
        dtype=initial_x.dtype,
    )

    mu_value = jnp.asarray(
        mu,
        dtype=initial_x.dtype,
    )

    direction_value = jnp.asarray(
        direction,
        dtype=initial_x.dtype,
    )

    def euler_step(
        unused_index,
        current_x,
    ):
        del unused_index

        gradient = grad_loss(
            current_x
        )

        drift = (
            direction_value
            * gradient
            / (1.0 - mu_value)
        )

        return (
            current_x
            + local_dt * drift
        )

    return jax.lax.fori_loop(
        0,
        fig13_flow_substeps,
        euler_step,
        initial_x,
    )

# ==========================================================
# GDM analogue of Rosca Figure 13
# ==========================================================

fig13_hessian_fn = jax.jit(
    jax.hessian(
        loss_fn_flat
    )
)


@jax.jit
def fig13_leading_hessian_pair(x):
    """
    Return the largest algebraic Hessian eigenvalue lambda_0
    and its normalized eigenvector u_0.
    """

    hessian = fig13_hessian_fn(x)

    # Suppress floating-point asymmetry.
    hessian = 0.5 * (
        hessian + hessian.T
    )

    eigenvalues, eigenvectors = (
        jnp.linalg.eigh(
            hessian
        )
    )

    return (
        eigenvalues[-1],
        eigenvectors[:, -1],
    )


@jax.jit
def fig13_momentum_stability_coefficient(x):
    r"""
    Compute the leading momentum-PF stability coefficient

        sc_0(x)
          = log(rho_0)/(eta * lambda_0)
            * <grad f(x), u_0>,

    where rho_0 is the characteristic root connected to rho=1
    at eta*lambda=0.

    The eigenvector sign is chosen so that

        <grad f(x), u_0> >= 0.

    For eta*lambda above the upper branch point, the square-root
    sign must switch so that the selected root reaches -1 at

        eta*lambda = 2*(1+mu).
    """

    lambda_0, u_0 = (
        fig13_leading_hessian_pair(
            x
        )
    )

    gradient = grad_loss(x)

    gradient_projection = (
        jnp.dot(
            gradient,
            u_0,
        )
    )

    # Fix the arbitrary eigenvector sign using Rosca's
    # convention.
    u_0 = jnp.where(
        gradient_projection < 0.0,
        -u_0,
        u_0,
    )

    gradient_projection = jnp.dot(
        gradient,
        u_0,
    )

    real_dtype = x.dtype

    complex_dtype = (
        jnp.complex128
        if real_dtype == jnp.float64
        else jnp.complex64
    )

    eta_value = jnp.asarray(
        eta,
        dtype=real_dtype,
    )

    mu_value = jnp.asarray(
        mu,
        dtype=real_dtype,
    )

    q = eta_value * lambda_0

    characteristic_center = (
        1.0 + mu_value - q
    )

    discriminant = (
        characteristic_center**2
        - 4.0 * mu_value
    )

    complex_discriminant = (
        discriminant.astype(
            complex_dtype
        )
    )

    principal_sqrt = jnp.sqrt(
        complex_discriminant
    )

    # Upper branch point:
    #
    #     q = (1 + sqrt(mu))^2.
    #
    # Continuing the root that begins at rho=1 requires the
    # negative square-root branch after this point.
    upper_branch_point = (
        1.0 + jnp.sqrt(mu_value)
    ) ** 2

    continued_sqrt = jnp.where(
        q > upper_branch_point,
        -principal_sqrt,
        principal_sqrt,
    )

    characteristic_root = 0.5 * (
        characteristic_center.astype(
            complex_dtype
        )
        + continued_sqrt
    )

    small_q = (
        jnp.abs(q) < 1e-8
    )

    safe_q = jnp.where(
        small_q,
        jnp.ones_like(q),
        q,
    )

    coefficient_direct = (
        jnp.log(
            characteristic_root
        )
        / safe_q.astype(
            complex_dtype
        )
    )

    # Limit as q -> 0.
    coefficient_limit = (
        -1.0
        / (1.0 - mu_value)
    ).astype(
        complex_dtype
    )

    coefficient = jnp.where(
        small_q,
        coefficient_limit,
        coefficient_direct,
    )

    stability_coefficient = (
        coefficient
        * gradient_projection.astype(
            complex_dtype
        )
    )

    return (
        lambda_0,
        jnp.real(
            stability_coefficient
        ),
        jnp.imag(
            stability_coefficient
        ),
        jnp.abs(
            stability_coefficient
        ),
        gradient_projection,
        characteristic_root,
    )

# ==========================================================
# Largest algebraic Hessian eigenvalue
# ==========================================================

loss_hessian = jax.jit(
    jax.hessian(
        loss_fn_flat
    )
)


@jax.jit
def largest_hessian_eigenvalue(x):
    """
    Return

        lambda_0(x) = max_i lambda_i(H(x)),

    the largest algebraic Hessian eigenvalue.

    This is different from

        max_i |lambda_i(H(x))|,

    which is the Hessian spectral norm used by the real
    HA-SME domain check.
    """

    hessian = loss_hessian(x)

    # Remove small numerical asymmetries before eigvalsh.
    hessian = 0.5 * (
        hessian + hessian.T
    )

    eigenvalues = jnp.linalg.eigvalsh(
        hessian
    )

    # eigvalsh returns eigenvalues in ascending order.
    return eigenvalues[-1]

# ==========================================================
# Largest positive-curvature Hessian eigenvalue
# ==========================================================

@jax.jit
def largest_hessian_eigenvalue(x):
    """
    Largest algebraic Hessian eigenvalue

        lambda_0(x) = max_i lambda_i(H(x)).

    This differs from the Hessian spectral norm

        ||H(x)||_2 = max_i |lambda_i(H(x))|.

    The former is used for the positive-curvature momentum
    edge of stability. The latter is used for the real
    HA-SME domain condition.
    """

    hessian = jax.hessian(
        loss_fn_flat
    )(x)

    # Suppress small numerical asymmetries.
    hessian = 0.5 * (
        hessian + hessian.T
    )

    eigenvalues = jnp.linalg.eigvalsh(
        hessian
    )

    return eigenvalues[-1]


# ==========================================================
# Momentum HA-SME step-size diagnostic
# ==========================================================

def check_momentum_hasme_stepsize(
    init_param,
    eta,
    mu,
):
    """
    Check the real-valued Momentum HA-SME condition

        eta * ||H(x)||_2 < (1 - sqrt(mu))^2

    at the initial parameter vector.
    """

    init_flat_param, _ = ravel_pytree(init_param)

    hessian = jax.hessian(
        loss_fn_flat
    )(init_flat_param)

    hessian_eigenvalues = jnp.linalg.eigvalsh(
        hessian
    )

    max_abs_eigenvalue = jnp.max(
        jnp.abs(hessian_eigenvalues)
    )

    lambda_0 = hessian_eigenvalues[-1]

    eta_value = jnp.asarray(
        eta,
        dtype=init_flat_param.dtype,
    )

    mu_value = jnp.asarray(
        mu,
        dtype=init_flat_param.dtype,
    )

    scaled_hessian_norm = (
        eta_value * max_abs_eigenvalue
    )

    radius = (
        1.0 - jnp.sqrt(mu_value)
    ) ** 2

    lambda_abs_value = float(
        max_abs_eigenvalue
    )

    lambda_0_value = float(
        lambda_0
    )

    # Real HA-SME boundary uses the spectral norm.
    if lambda_abs_value > 0.0:
        real_pf_eta_boundary = (
            float(radius)
            / lambda_abs_value
        )
    else:
        real_pf_eta_boundary = float("inf")

    # Momentum edge-of-stability boundary uses the largest
    # algebraic positive eigenvalue.
    if lambda_0_value > 0.0:
        momentum_stability_boundary = (
            2.0
            * (1.0 + float(mu_value))
            / lambda_0_value
        )
    else:
        # There is no positive-curvature edge if all Hessian
        # eigenvalues are nonpositive.
        momentum_stability_boundary = float("inf")
    print(
        "  max |Hessian eigenvalue| "
        "(spectral norm):",
        f"{lambda_abs_value:.17e}",
    )

    print(
        "  largest algebraic Hessian eigenvalue "
        "(lambda_0):",
        f"{lambda_0_value:.17e}",
    )

    if lambda_0_value > 0.0:
        initial_edge_ratio = (
            float(eta_value)
            * lambda_0_value
            / (
                2.0
                * (1.0 + float(mu_value))
            )
        )

        print(
            "  initial momentum edge ratio "
            "eta * lambda_0 / [2 * (1 + mu)]:",
            f"{initial_edge_ratio:.17e}",
        )
    else:
        initial_edge_ratio = float("-inf")

        print(
            "  initial momentum edge ratio: undefined "
            "(lambda_0 is not positive)"
        )

    print(
        "  real PF eta boundary "
        "(1 - sqrt(mu))^2 / ||H||_2:",
        f"{real_pf_eta_boundary:.17e}",
    )

    print(
        "  momentum stability boundary "
        "2 * (1 + mu) / lambda_0:",
        f"{momentum_stability_boundary:.17e}",
    )

    print(
        "  selected eta:",
        f"{float(eta_value):.17e}",
    )

    return (
        lambda_abs_value,
        lambda_0_value,
    )


# ==========================================================
# TensorBoard setup
# ==========================================================

if not DEBUG:
    if args.log_dir is not None:
        log_dir = args.log_dir

    else:
        timestamp = datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )

        pretrained_tag = (
            f"pretrained_{args.pretrained_weights}"
            if args.load_pretrained
            else "random_init"
        )

        log_dir = (
            f"seed_{args.seed}_"
            f"logs/"
            f"eta_{eta:g}_"
            f"mu_{mu:g}_"
            f"sigma_{sigma:g}_"
            f"dt_ratio_{args.eta_dt_ratio:g}_"
            f"{pretrained_tag}_"
            f"float64_"
            f"{timestamp}"
        )

    print(
        f"TensorBoard log directory: {log_dir}"
    )

    writer = SummaryWriter(log_dir)

    hparams = {
        "problem": args.problem,
        "eta": eta,
        "dt": dt,
        "eta_dt_ratio": args.eta_dt_ratio,
        "brownian_tol_ratio": (
            args.brownian_tol_ratio
        ),
        "brownian_tol": brownian_tol,
        "sigma": sigma,
        "mu": mu,
        "seed": args.seed,
        "num_steps": num_steps,
        "num_runs": num_runs,
        "sde_batch_size": (
            args.sde_batch_size
        ),
        "final_sde_time": final_sde_time,
        "gpu_enabled": args.gpu,
        "load_pretrained": (
            args.load_pretrained
        ),
        "dtype": "float64",
        "num_hidden_layers": len(hidden_widths),
        "hidden_widths": "x".join(
            str(width)
            for width in hidden_widths
        ),
        "activation": args.activation,
        "initial_model_dimension": int(d),
        "discrete_method": (
            args.discrete_method
        ),
        "reference_name": (
            reference_name
        ),
        "complex_pf": (
            args.complex_pf
        ),
        "stochastic_setting": (
            stochastic_setting
        ),
        "requested_sigma": (
            requested_sigma
        ),
        "effective_sigma": sigma,
        "requested_num_runs": (
            requested_num_runs
        ),
        "effective_num_runs": (
            num_runs
        )
    }

if args.problem == "quadratic":
    hparams.update(
        {
            "quad_h11": args.quad_h11,
            "quad_h12": args.quad_h12,
            "quad_h22": args.quad_h22,
            "quad_target_1": (
                args.quad_target_1
            ),
            "quad_target_2": (
                args.quad_target_2
            ),
            "quad_init_1": (
                args.quad_init_1
            ),
            "quad_init_2": (
                args.quad_init_2
            ),
        }
    )

    writer.add_text(
        "Parameters",
        str(hparams),
    )


# ==========================================================
# Discrete SGDM
# ==========================================================

@jax.jit
def discrete_reference_step(
    x,
    x_prev,
    key,
    eta,
    mu,
    sigma,
):
    grad = grad_loss(x)

    # Explicitly generate noise with the same dtype as x.
    noise = sigma * jax.random.normal(
        key,
        x.shape,
        dtype=x.dtype,
    )

    x_next = (
        x
        - eta * (grad + noise)
        + mu * (x - x_prev)
    )

    return x_next, x

@jax.jit
def gdm_step(
    x,
    x_prev,
    eta,
    mu,
):
    grad = grad_loss(x)

    x_next = (
        x
        - eta * grad
        + mu * (x - x_prev)
    )

    return x_next, x


def run_discrete_reference_trials(
    keys,
    num_steps,
    eta,
    mu,
    sigma,
    init_param,
):
    init_x, _ = ravel_pytree(
        init_param
    )

    eta_value = jnp.asarray(
        eta,
        dtype=init_x.dtype,
    )

    mu_value = jnp.asarray(
        mu,
        dtype=init_x.dtype,
    )

    sigma_value = jnp.asarray(
        sigma,
        dtype=init_x.dtype,
    )

    deterministic = (
        float(sigma) == 0.0
    )

    if deterministic:
        # Use lax.scan instead of tracing a Python loop with
        # one copy of the computation for every optimization
        # step.
        def scan_step(carry, unused_input):
            del unused_input

            x, x_prev = carry

            x_next, old_x = gdm_step(
                x,
                x_prev,
                eta_value,
                mu_value,
            )

            loss = loss_fn_flat(
                x_next
            )

            new_carry = (
                x_next,
                old_x,
            )

            output = (
                loss,
                x_next,
            )

            return new_carry, output

        _, (
            losses,
            trajectory,
        ) = jax.lax.scan(
            scan_step,
            (
                init_x,
                init_x,
            ),
            xs=None,
            length=num_steps,
        )

        # Preserve the common Monte Carlo dimension even
        # though deterministic GDM has exactly one run.
        all_losses = losses[
            None,
            :
        ]

        all_params = trajectory[
            None,
            :,
            :
        ]

    else:
        def run_single_stochastic_trial(
            key,
        ):
            def scan_step(
                carry,
                unused_input,
            ):
                del unused_input

                x, x_prev, current_key = (
                    carry
                )

                current_key, step_key = (
                    jax.random.split(
                        current_key
                    )
                )

                x_next, old_x = (
                    discrete_reference_step(
                        x,
                        x_prev,
                        step_key,
                        eta_value,
                        mu_value,
                        sigma_value,
                    )
                )

                loss = loss_fn_flat(
                    x_next
                )

                new_carry = (
                    x_next,
                    old_x,
                    current_key,
                )

                output = (
                    loss,
                    x_next,
                )

                return new_carry, output

            _, (
                losses,
                trajectory,
            ) = jax.lax.scan(
                scan_step,
                (
                    init_x,
                    init_x,
                    key,
                ),
                xs=None,
                length=num_steps,
            )

            return (
                losses,
                trajectory,
            )

        all_losses, all_params = jax.vmap(
            run_single_stochastic_trial
        )(keys)

    avg_losses = jnp.mean(
        all_losses,
        axis=0,
    )

    return (
        avg_losses,
        all_losses,
        all_params,
    )


# ==========================================================
# SDE trials
# ==========================================================

def run_sde_trials(
    optimizer_class,
    keys,
    num_steps,
    eta,
    mu,
    dt,
    cov_mat_func,
    init_param,
    brownian_tol,
    batch_size,
    flow_objective_fn,
    observable_fn,
    optimizer_kwargs=None,
):
    init_flat_param, _ = ravel_pytree(
        init_param
    )

    # All autonomous SDE models begin at exactly the same
    # parameter vector as SGDM. No initialization correction
    # is applied.
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer = optimizer_class(
        eta,
        mu,
        flow_objective_fn,
        cov_mat_func,
        brownian_tol=brownian_tol,
        **optimizer_kwargs,
    )

    eta_jax = jnp.asarray(
        eta,
        dtype=init_flat_param.dtype,
    )

    dt_jax = jnp.asarray(
        dt,
        dtype=init_flat_param.dtype,
    )

    time_points = (
        eta_jax
        * jnp.arange(
            1,
            num_steps + 1,
            dtype=init_flat_param.dtype,
        )
    )

    end_time = time_points[-1]

    def solve_single_run(key):
        sol = optimizer.solve(
            init_flat_param,
            end_time,
            dt_jax,
            time_points,
            key,
        )

        return sol.ys

    def compute_losses_from_sol(sol_ys):
        return jax.vmap(
            observable_fn
        )(sol_ys)

    all_loss_batches = []
    all_trajectory_batches = []

    num_key_rows = int(
        keys.shape[0]
    )

    for batch_start in range(
        0,
        num_key_rows,
        batch_size,
    ):
        batch_end = min(
            batch_start + batch_size,
            num_key_rows,
        )

        batch_keys = keys[
            batch_start:batch_end
        ]

        # Shape:
        # (batch_size, num_steps, parameter_dimension)
        if int(batch_keys.shape[0]) == 1:
            single_sol_ys = (
                jax.block_until_ready(
                    solve_single_run(
                        batch_keys[0]
                    )
                )
            )

            batch_sol_ys = single_sol_ys[
                None,
                ...,
            ]

        else:
            batch_sol_ys = (
                jax.block_until_ready(
                    jax.vmap(
                        solve_single_run
                    )(batch_keys)
                )
            )

        if not bool(
            jnp.all(
                jnp.isfinite(
                    batch_sol_ys
                )
            )
        ):
            raise FloatingPointError(
                "An SDE trajectory contains NaN or "
                "infinite values in the batch starting at "
                f"run {batch_start + 1}."
            )

        # Shape: (batch_size, num_steps)
        batch_losses = jax.block_until_ready(
            jax.vmap(
                compute_losses_from_sol
            )(batch_sol_ys)
        )

        if not bool(
            jnp.all(
                jnp.isfinite(
                    batch_losses
                )
            )
        ):
            raise FloatingPointError(
                "An SDE loss contains NaN or infinite "
                "values in the batch starting at "
                f"run {batch_start + 1}."
            )

        all_trajectory_batches.append(
            batch_sol_ys
        )

        all_loss_batches.append(
            batch_losses
        )

        print(
            f"  completed SDE runs "
            f"{batch_start + 1}-{batch_end} "
            f"of {num_key_rows}",
            flush=True,
        )

    # These operations must happen after all batches have
    # been collected, not inside the batch loop.
    all_losses = jnp.concatenate(
        all_loss_batches,
        axis=0,
    )

    all_sol_ys = jnp.concatenate(
        all_trajectory_batches,
        axis=0,
    )

    avg_losses = jnp.mean(
        all_losses,
        axis=0,
    )

    return (
        avg_losses,
        all_losses,
        all_sol_ys,
    )

# ==========================================================
# Rosca Figure-10-style local one-step errors for GDM
# ==========================================================


def get_deterministic_flow_specification(algo):
    """
    Return the deterministic continuous model corresponding
    to an algorithm name.

    Every model is run with stochastic=False because this
    diagnostic compares against deterministic GDM.
    """

    optimizer_kwargs = {
        "stochastic": False,
    }

    if algo == COMPLEX_PF_ALGORITHM:
        return (
            ComplexMomentumPF,
            complex_loss_fn_flat,
            complex_objective_observable,
            optimizer_kwargs,
        )

    if algo == "Momentum-HA-SME":
        return (
            MomentumHASME,
            loss_fn_flat,
            loss_fn_flat,
            optimizer_kwargs,
        )

    if algo == "Momentum-SME-1":
        return (
            MomentumFirstOrderFlow,
            loss_fn_flat,
            loss_fn_flat,
            optimizer_kwargs,
        )

    if algo == "Momentum-SME-2":
        return (
            MomentumSecondOrderFlow,
            loss_fn_flat,
            loss_fn_flat,
            optimizer_kwargs,
        )

    raise ValueError(
        f"Unknown continuous model for local prediction: {algo}"
    )


# ==========================================================
# Two-sided momentum-recurrence local errors
# ==========================================================

def run_local_gdm_recurrence_predictions(
    algo,
    reference_params,
    init_flat_param,
    eta,
    mu,
    dt,
    cov_mat_func,
    brownian_tol,
    batch_size,
    root_key,
):
    r"""
    Compute the two-sided local momentum recurrence error.

    For each GDM transition centered at x_{k-1}, compare

        Phi_A^{+eta}(x_{k-1})
        + mu * Phi_A^{-eta}(x_{k-1})

    against

        x_k + mu * x_{k-2}.

    The standard initialization x_{-1} = x_0 is used for the
    first transition.

    The reconstructed prediction of the next GDM state is

        x_hat_k
        = Phi_A^{+eta}(x_{k-1})
        + mu * Phi_A^{-eta}(x_{k-1})
        - mu * x_{k-2}.

    Returned errors are

        ||x_hat_k - x_k||_2

    and the signed reconstructed recurrence loss error

        f(x_k)
        + mu * f(x_{k-2})
        - f(Phi_A^{+eta}(x_{k-1}))
        - mu * f(Phi_A^{-eta}(x_{k-1})).

    The loss error is deliberately signed.
    """

    if args.discrete_method != "gdm":
        raise ValueError(
            "The two-sided recurrence diagnostic requires "
            "--discrete_method gdm."
        )

    reference_params = jnp.asarray(
        reference_params,
        dtype=jnp.float64,
    )

    if reference_params.ndim != 3:
        raise ValueError(
            "Expected reference_params to have shape "
            "(runs, steps, dimension)."
        )

    if int(reference_params.shape[0]) != 1:
        raise ValueError(
            "Deterministic GDM should have exactly one run."
        )

    # Stored trajectory:
    #
    #     x_1, x_2, ..., x_N.
    gdm_stored_states = reference_params[0]

    # Construct:
    #
    #     x_{-1}, x_0, x_1, ..., x_N,
    #
    # using the standard momentum initialization
    #
    #     x_{-1} = x_0.
    extended_gdm_history = jnp.concatenate(
        [
            init_flat_param[None, :],
            init_flat_param[None, :],
            gdm_stored_states,
        ],
        axis=0,
    )

    # For transitions indexed k = 1, ..., N:
    #
    # previous states: x_{-1}, x_0, ..., x_{N-2}
    # center states:   x_0,    x_1, ..., x_{N-1}
    # next states:     x_1,    x_2, ..., x_N
    gdm_previous_states = (
        extended_gdm_history[:-2]
    )

    gdm_center_states = (
        extended_gdm_history[1:-1]
    )

    gdm_next_states = (
        extended_gdm_history[2:]
    )

    expected_num_steps = int(
        gdm_stored_states.shape[0]
    )

    if (
        int(gdm_previous_states.shape[0])
        != expected_num_steps
        or int(gdm_center_states.shape[0])
        != expected_num_steps
        or int(gdm_next_states.shape[0])
        != expected_num_steps
    ):
        raise RuntimeError(
            "The two-sided GDM trajectory arrays are not "
            "aligned correctly."
        )

    mu_value = jnp.asarray(
        mu,
        dtype=init_flat_param.dtype,
    )

    eta_value = jnp.asarray(
        eta,
        dtype=init_flat_param.dtype,
    )

    # Exact discrete recurrence target:
    #
    #     x_k + mu*x_{k-2}.
    discrete_recurrence_targets = (
        gdm_next_states
        + mu_value
        * gdm_previous_states
    )

    # Independently verify the indexing using
    #
    #     x_k + mu*x_{k-2}
    #       = (1+mu)*x_{k-1}
    #         - eta*grad f(x_{k-1}).
    center_gradients = jax.vmap(
        grad_loss
    )(
        gdm_center_states
    )

    analytical_recurrence_targets = (
        (1.0 + mu_value)
        * gdm_center_states
        - eta_value
        * center_gradients
    )

    recurrence_indexing_mismatch = (
        jnp.linalg.norm(
            discrete_recurrence_targets
            - analytical_recurrence_targets,
            axis=-1,
        )
    )

    maximum_indexing_mismatch = float(
        jnp.max(
            recurrence_indexing_mismatch
        )
    )

    print(
        f"\n{algo} recurrence indexing check:"
    )

    print(
        "  max ||x_k + mu*x_{k-2} "
        "- [(1+mu)x_{k-1} - eta*grad f(x_{k-1})]||:",
        f"{maximum_indexing_mismatch:.17e}",
    )

    # This should be close to float64 roundoff.
    if maximum_indexing_mismatch > 1e-10:
        raise RuntimeError(
            "The GDM two-sided recurrence indexing check "
            "failed. The expected equality does not hold."
        )

    (
        optimizer_class,
        flow_objective_fn,
        observable_fn,
        optimizer_kwargs,
    ) = get_deterministic_flow_specification(
        algo
    )

    optimizer = optimizer_class(
        eta,
        mu,
        flow_objective_fn,
        cov_mat_func,
        brownian_tol=brownian_tol,
        **optimizer_kwargs,
    )

    dt_value = jnp.asarray(
        dt,
        dtype=init_flat_param.dtype,
    )

    forward_save_times = (
        eta_value[None]
    )

    backward_save_times = (
        (-eta_value)[None]
    )

    num_recurrence_steps = int(
        gdm_center_states.shape[0]
    )

    local_keys = jax.random.split(
        root_key,
        num_recurrence_steps,
    )

    def solve_forward_and_backward(
        center_state,
        local_key,
    ):
        # Forward flow:
        #
        #     Phi^{+eta}(center_state).
        forward_solution = optimizer.solve(
            center_state,
            eta_value,
            dt_value,
            forward_save_times,
            local_key,
        )

        forward_endpoint = (
            forward_solution.ys[-1]
        )

        # Backward flow:
        #
        #     Phi^{-eta}(center_state).
        #
        # Diffrax requires a negative initial step when
        # integrating from t=0 to t=-eta.
        backward_solution = optimizer.solve(
            center_state,
            -eta_value,
            -dt_value,
            backward_save_times,
            local_key,
        )

        backward_endpoint = (
            backward_solution.ys[-1]
        )

        return (
            forward_endpoint,
            backward_endpoint,
        )

    forward_batches = []
    backward_batches = []

    for batch_start in range(
        0,
        num_recurrence_steps,
        batch_size,
    ):
        batch_end = min(
            batch_start + batch_size,
            num_recurrence_steps,
        )

        batch_centers = (
            gdm_center_states[
                batch_start:batch_end
            ]
        )

        batch_keys = local_keys[
            batch_start:batch_end
        ]

        if int(batch_centers.shape[0]) == 1:
            (
                single_forward,
                single_backward,
            ) = solve_forward_and_backward(
                batch_centers[0],
                batch_keys[0],
            )

            batch_forward = (
                single_forward[None, ...]
            )

            batch_backward = (
                single_backward[None, ...]
            )

        else:
            (
                batch_forward,
                batch_backward,
            ) = jax.vmap(
                solve_forward_and_backward
            )(
                batch_centers,
                batch_keys,
            )

        batch_forward = (
            jax.block_until_ready(
                batch_forward
            )
        )

        batch_backward = (
            jax.block_until_ready(
                batch_backward
            )
        )

        if not bool(
            jnp.all(
                jnp.isfinite(
                    batch_forward
                )
            )
        ):
            raise FloatingPointError(
                f"{algo} produced a non-finite forward "
                f"recurrence prediction in steps "
                f"{batch_start + 1}-{batch_end}."
            )

        if not bool(
            jnp.all(
                jnp.isfinite(
                    batch_backward
                )
            )
        ):
            raise FloatingPointError(
                f"{algo} produced a non-finite backward "
                f"recurrence prediction in steps "
                f"{batch_start + 1}-{batch_end}."
            )

        forward_batches.append(
            batch_forward
        )

        backward_batches.append(
            batch_backward
        )

        print(
            f"  {algo}: completed recurrence predictions "
            f"{batch_start + 1}-{batch_end} "
            f"of {num_recurrence_steps}",
            flush=True,
        )

    forward_endpoints = jnp.concatenate(
        forward_batches,
        axis=0,
    )

    backward_endpoints = jnp.concatenate(
        backward_batches,
        axis=0,
    )

    # Continuous two-sided recurrence combination:
    #
    #     Phi^{+eta}(x_{k-1})
    #       + mu*Phi^{-eta}(x_{k-1}).
    predicted_recurrence_combinations = (
        forward_endpoints
        + mu_value
        * backward_endpoints
    )

    recurrence_residuals = (
        predicted_recurrence_combinations
        - discrete_recurrence_targets
    )

    recurrence_parameter_errors = (
        jnp.linalg.norm(
            recurrence_residuals,
            axis=-1,
        )
    )

    # Reconstruct a prediction of x_k:
    #
    #     x_hat_k
    #       = predicted combination
    #         - mu*x_{k-2}.
    reconstructed_next_states = (
        predicted_recurrence_combinations
        - mu_value
        * gdm_previous_states
    )

    # This must agree with the recurrence-residual norm.
    reconstructed_parameter_errors = (
        jnp.linalg.norm(
            reconstructed_next_states
            - gdm_next_states,
            axis=-1,
        )
    )

    reconstruction_consistency_error = float(
        jnp.max(
            jnp.abs(
                reconstructed_parameter_errors
                - recurrence_parameter_errors
            )
        )
    )

    print(
        f"\n{algo} recurrence reconstruction check:"
    )

    print(
        "  max difference between recurrence-residual norm "
        "and reconstructed-next-state error:",
        f"{reconstruction_consistency_error:.17e}",
    )

    forward_losses = jax.vmap(
        observable_fn
    )(
        forward_endpoints
    )

    backward_losses = jax.vmap(
        observable_fn
    )(
        backward_endpoints
    )

    next_state_losses = jax.vmap(
        loss_fn_flat
    )(
        gdm_next_states
    )

    previous_state_losses = jax.vmap(
        loss_fn_flat
    )(
        gdm_previous_states
    )

    recurrence_loss_errors = (
        next_state_losses
        + mu_value * previous_state_losses
        - jnp.real(forward_losses)
        - mu_value * jnp.real(backward_losses)
    )

    if not bool(
        jnp.all(
            jnp.isfinite(
                recurrence_parameter_errors
            )
        )
    ):
        raise FloatingPointError(
            f"{algo} produced non-finite two-sided "
            "recurrence parameter errors."
        )

    if not bool(
        jnp.all(
            jnp.isfinite(
                recurrence_loss_errors
            )
        )
    ):
        raise FloatingPointError(
            f"{algo} produced non-finite reconstructed "
            "loss errors."
        )

    print(
        f"\n{algo} two-sided recurrence summary:"
    )

    print(
        "  minimum parameter error:",
        f"{float(jnp.min(recurrence_parameter_errors)):.17e}",
    )

    print(
        "  final parameter error:",
        f"{float(recurrence_parameter_errors[-1]):.17e}",
    )

    print(
        "  minimum absolute loss error:",
        f"{float(jnp.min(jnp.abs(recurrence_loss_errors))):.17e}",
    )

    print(
        "  final signed loss error:",
        f"{float(recurrence_loss_errors[-1]):.17e}",
    )

    return {
        "forward_endpoints": (
            forward_endpoints
        ),
        "backward_endpoints": (
            backward_endpoints
        ),
        "predicted_recurrence_combinations": (
            predicted_recurrence_combinations
        ),
        "discrete_recurrence_targets": (
            discrete_recurrence_targets
        ),
        "reconstructed_next_states": (
            reconstructed_next_states
        ),
        "parameter_errors": (
            recurrence_parameter_errors
        ),
        "loss_errors": (
            recurrence_loss_errors
        ),
        "indexing_mismatch": (
            recurrence_indexing_mismatch
        ),
    }




# ==========================================================
# Common initialization
# ==========================================================

# ==========================================================
# Common initialization
# ==========================================================

if args.problem == "quadratic":
    if args.load_pretrained:
        raise ValueError(
            "--load_pretrained is only supported for "
            "--problem nn"
        )

    init_param = jnp.asarray(
        [
            args.quad_init_1,
            args.quad_init_2,
        ],
        dtype=jnp.float64,
    )

else:
    if args.load_pretrained:
        with open(
            f"checkpoints/SGD_weights_"
            f"{args.pretrained_weights}.pkl",
            "rb",
        ) as file:
            init_param = pickle.load(
                file
            )

        init_param = (
            jax.tree_util.tree_map(
                lambda value: jnp.asarray(
                    value,
                    dtype=jnp.float64,
                ),
                init_param,
            )
        )

    else:
        key, subkey = (
            jax.random.split(key)
        )

        init_param = model.init(
            subkey,
            jnp.ones(
                input_shape,
                dtype=jnp.float64,
            ),
        )["params"]

# Verify that float64 is actually being used.
init_flat_param, _ = ravel_pytree(
    init_param
)

print(
    "Initial parameter vector:",
    np.asarray(init_flat_param),
)

print(
    "Parameter dtype:",
    init_flat_param.dtype,
)

print(
    "Training-data dtype:",
    X_train.dtype,
)


# ==========================================================
# HA-SME domain check
# ==========================================================

# ==========================================================
# Initial Hessian diagnostic
# ==========================================================
#
# Compute this diagnostic for both:
#
#   1. real Momentum HA-SME;
#   2. complex Momentum PF.
#
# In complex-PF mode this is diagnostic only. It does not
# enforce the real HA-SME domain and does not run real HA-SME.

run_initial_hessian_diagnostic = (
    "Momentum-HA-SME" in algorithms
    or COMPLEX_PF_ALGORITHM in algorithms
)

initial_hessian_norm = None
initial_lambda_0 = None

if run_initial_hessian_diagnostic:
    (
        initial_hessian_norm,
        initial_lambda_0,
    ) = check_momentum_hasme_stepsize(
        init_param=init_param,
        eta=eta,
        mu=mu,
    )

    if initial_lambda_0 > 0.0:
        initial_edge_ratio = (
            eta
            * initial_lambda_0
            / (
                2.0
                * (1.0 + mu)
            )
        )
    
        initial_above_edge = (
            initial_edge_ratio > 1.0
        )
    
        print(
            "  initial momentum edge ratio "
            "eta * lambda_0 / [2 * (1 + mu)]:",
            f"{initial_edge_ratio:.17e}",
        )
    
        print(
            "  initial state above momentum edge:",
            initial_above_edge,
        )
    
    else:
        initial_edge_ratio = float("nan")
    
        print(
            "  initial momentum edge ratio: undefined "
            "because lambda_0 is not positive."
        )

    print(
        "HESSIAN_DIAGNOSTIC_RESULT="
        f"{initial_hessian_norm:.17e}"
    )
    print(
        "LARGEST_POSITIVE_EIGENVALUE_RESULT="
        f"{initial_lambda_0:.17e}"
    )

    if args.hessian_diagnostic_only:
        print(
            "Initial Hessian diagnostic complete. "
            "Skipping all optimization simulations."
        )

        raise SystemExit(0)


# ==========================================================
# Momentum-DAL analogue of Rosca Figure 19
# ==========================================================

if args.momentum_dal_fig19:
    print(
        "\nRunning the momentum-DAL Figure-19-style "
        "experiment...",
        flush=True,
    )

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    momentum_dal_optimizer = MomentumDALOptimizer(
        momentum=mu,
        f=loss_fn_flat,
        target_ratio=args.dal_target_ratio,
        min_stepsize=args.dal_min_eta,
        max_stepsize=args.dal_max_eta,
        curvature_epsilon=(
            args.dal_curvature_epsilon
        ),
    )

    # Diagnostic tuple returned by MomentumDALOptimizer:
    #
    #   0: used learning rate
    #   1: unclipped/raw learning rate
    #   2: intrinsic momentum drift
    #   3: complete momentum update norm
    #   4: gradient norm
    #   5: ||H g||
    #   6: ||H g_hat||

    @jax.jit
    def run_adaptive_momentum_dal(
        initial_x,
    ):
        def scan_step(
            carry,
            unused_input,
        ):
            del unused_input

            x, x_prev = carry

            (
                x_next,
                diagnostics,
            ) = momentum_dal_optimizer.adaptive_step(
                x,
                x_prev,
            )

            next_loss = loss_fn_flat(
                x_next
            )

            new_carry = (
                x_next,
                x,
            )

            output = (
                next_loss,
                *diagnostics,
            )

            return (
                new_carry,
                output,
            )

        _, outputs = jax.lax.scan(
            scan_step,
            (
                initial_x,
                initial_x,
            ),
            xs=None,
            length=num_steps,
        )

        return outputs

    @jax.jit
    def run_fixed_momentum_gd(
        initial_x,
        fixed_stepsize,
    ):
        fixed_stepsize = jnp.asarray(
            fixed_stepsize,
            dtype=initial_x.dtype,
        )

        def scan_step(
            carry,
            unused_input,
        ):
            del unused_input

            x, x_prev = carry

            (
                x_next,
                diagnostics,
            ) = momentum_dal_optimizer.fixed_step(
                x,
                x_prev,
                fixed_stepsize,
            )

            next_loss = loss_fn_flat(
                x_next
            )

            new_carry = (
                x_next,
                x,
            )

            output = (
                next_loss,
                *diagnostics,
            )

            return (
                new_carry,
                output,
            )

        _, outputs = jax.lax.scan(
            scan_step,
            (
                initial_x,
                initial_x,
            ),
            xs=None,
            length=num_steps,
        )

        return outputs

    def block_and_convert_outputs(
        outputs,
    ):
        outputs = jax.tree_util.tree_map(
            lambda value: (
                jax.block_until_ready(value)
            ),
            outputs,
        )

        (
            losses,
            used_stepsizes,
            raw_stepsizes,
            intrinsic_drifts,
            update_norms,
            gradient_norms,
            hessian_gradient_norms,
            directional_curvatures,
        ) = outputs

        return {
            "loss": np.asarray(
                losses,
                dtype=np.float64,
            ),
            "stepsize": np.asarray(
                used_stepsizes,
                dtype=np.float64,
            ),
            "raw_stepsize": np.asarray(
                raw_stepsizes,
                dtype=np.float64,
            ),
            "intrinsic_drift": np.asarray(
                intrinsic_drifts,
                dtype=np.float64,
            ),
            "update_norm": np.asarray(
                update_norms,
                dtype=np.float64,
            ),
            "gradient_norm": np.asarray(
                gradient_norms,
                dtype=np.float64,
            ),
            "hessian_gradient_norm": np.asarray(
                hessian_gradient_norms,
                dtype=np.float64,
            ),
            "directional_curvature": np.asarray(
                directional_curvatures,
                dtype=np.float64,
            ),
        }

    # ------------------------------------------------------
    # Run adaptive M-DAL
    # ------------------------------------------------------

    adaptive_outputs = (
        run_adaptive_momentum_dal(
            init_flat_param
        )
    )

    momentum_dal_results = (
        block_and_convert_outputs(
            adaptive_outputs
        )
    )

    if not all(
        np.all(np.isfinite(values))
        for values in (
            momentum_dal_results.values()
        )
    ):
        raise FloatingPointError(
            "M-DAL produced non-finite diagnostics."
        )

    print(
        "  finished adaptive M-DAL",
        flush=True,
    )

    # ------------------------------------------------------
    # Run fixed-learning-rate GDM baselines
    # ------------------------------------------------------

    fixed_momentum_results = {}

    # Remove duplicates while preserving the command-line order.
    unique_fixed_etas = []

    for fixed_eta in args.dal_fixed_etas:
        fixed_eta = float(fixed_eta)

        if fixed_eta not in unique_fixed_etas:
            unique_fixed_etas.append(
                fixed_eta
            )

    for fixed_eta in unique_fixed_etas:
        print(
            f"  running fixed GDM with eta={fixed_eta:g}",
            flush=True,
        )

        fixed_outputs = run_fixed_momentum_gd(
            init_flat_param,
            jnp.asarray(
                fixed_eta,
                dtype=init_flat_param.dtype,
            ),
        )

        fixed_result = block_and_convert_outputs(
            fixed_outputs
        )

        if not all(
            np.all(np.isfinite(values))
            for values in fixed_result.values()
        ):
            print(
                "    warning: this fixed-learning-rate run "
                "became non-finite; retaining finite prefixes "
                "in the plot.",
                flush=True,
            )

        fixed_momentum_results[
            fixed_eta
        ] = fixed_result

    # ------------------------------------------------------
    # Print concise summary
    # ------------------------------------------------------

    dal_clip_mask = ~np.isclose(
        momentum_dal_results[
            "stepsize"
        ],
        momentum_dal_results[
            "raw_stepsize"
        ],
        rtol=1e-10,
        atol=1e-14,
    )

    print(
        "\nMomentum-DAL summary:"
    )

    print(
        "  mu:",
        mu,
    )

    print(
        "  target ratio tau:",
        args.dal_target_ratio,
    )

    print(
        "  minimum used eta:",
        f"{np.min(momentum_dal_results['stepsize']):.12e}",
    )

    print(
        "  maximum used eta:",
        f"{np.max(momentum_dal_results['stepsize']):.12e}",
    )

    print(
        "  mean intrinsic drift:",
        f"{np.mean(momentum_dal_results['intrinsic_drift']):.12e}",
    )

    print(
        "  maximum intrinsic drift:",
        f"{np.max(momentum_dal_results['intrinsic_drift']):.12e}",
    )

    print(
        "  steps at which the learning-rate cap/floor was active:",
        int(np.sum(dal_clip_mask)),
        f"out of {num_steps}",
    )

    print(
        "  final M-DAL training loss:",
        f"{momentum_dal_results['loss'][-1]:.12e}",
    )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    dal_plot_dir = Path(
        args.plot_dir
    )

    dal_plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dal_steps = np.arange(
        1,
        num_steps + 1,
        dtype=np.int64,
    )

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(11, 13),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.1,
                1.0,
                1.0,
                1.0,
            ],
        },
    )

    (
        loss_axis,
        stepsize_axis,
        drift_axis,
        update_axis,
    ) = axes

    fixed_colors = plt.cm.viridis(
        np.linspace(
            0.1,
            0.8,
            len(unique_fixed_etas),
        )
    )

    tiny = np.finfo(
        np.float64
    ).tiny

    for color, fixed_eta in zip(
        fixed_colors,
        unique_fixed_etas,
    ):
        result = fixed_momentum_results[
            fixed_eta
        ]

        label = (
            rf"GDM, $\eta={fixed_eta:g}$"
        )

        loss_axis.plot(
            dal_steps,
            np.maximum(
                result["loss"],
                tiny,
            ),
            color=color,
            linewidth=1.7,
            alpha=0.9,
            label=label,
        )

        stepsize_axis.plot(
            dal_steps,
            np.maximum(
                result["stepsize"],
                tiny,
            ),
            color=color,
            linewidth=1.7,
            alpha=0.9,
        )

        drift_axis.plot(
            dal_steps,
            np.maximum(
                result["intrinsic_drift"],
                tiny,
            ),
            color=color,
            linewidth=1.7,
            alpha=0.9,
        )

        update_axis.plot(
            dal_steps,
            np.maximum(
                result["update_norm"],
                tiny,
            ),
            color=color,
            linewidth=1.7,
            alpha=0.9,
        )

    dal_label = (
        r"M-DAL: "
        r"$\eta_k="
        r"\frac{2(1-\mu)^2}"
        r"{(1+\mu)\|H\widehat g\|}$"
    )

    loss_axis.plot(
        dal_steps,
        np.maximum(
            momentum_dal_results["loss"],
            tiny,
        ),
        color="black",
        linewidth=2.8,
        label=dal_label,
    )

    stepsize_axis.plot(
        dal_steps,
        np.maximum(
            momentum_dal_results[
                "stepsize"
            ],
            tiny,
        ),
        color="black",
        linewidth=2.8,
    )

    # Show the unclipped value to make clipping visible.
    stepsize_axis.plot(
        dal_steps,
        np.maximum(
            momentum_dal_results[
                "raw_stepsize"
            ],
            tiny,
        ),
        color="tab:red",
        linewidth=1.3,
        linestyle=":",
        alpha=0.8,
        label="M-DAL raw learning rate",
    )

    drift_axis.plot(
        dal_steps,
        np.maximum(
            momentum_dal_results[
                "intrinsic_drift"
            ],
            tiny,
        ),
        color="black",
        linewidth=2.8,
    )

    update_axis.plot(
        dal_steps,
        np.maximum(
            momentum_dal_results[
                "update_norm"
            ],
            tiny,
        ),
        color="black",
        linewidth=2.8,
    )

    # Cross-entropy and all three diagnostics are nonnegative.
    for axis in axes:
        axis.set_yscale(
            "log"
        )

        axis.grid(
            True,
            which="both",
            alpha=0.3,
        )

    loss_axis.set_ylabel(
        "Training loss"
    )

    loss_axis.set_title(
        "Momentum analogue of Rosca Figure 19\n"
        "Full-batch Iris neural network"
    )

    loss_axis.legend(
        loc="best",
        fontsize=8,
    )

    stepsize_axis.set_ylabel(
        r"Learning rate $\eta_k$"
    )

    stepsize_axis.legend(
        loc="best",
        fontsize=8,
    )

    drift_axis.set_ylabel(
        "Intrinsic momentum drift\n"
        r"$\|\nabla^2f \nabla f\|$"
    )

    update_axis.set_ylabel(
        r"$\|x_{k+1}-x_k\|$"
    )

    update_axis.set_xlabel(
        "GDM iteration"
    )

    figure.tight_layout()

    hidden_width_tag = "x".join(
        str(width)
        for width in hidden_widths
    )

    dal_plot_name = (
        f"momentum_dal_fig19_"
        f"problem_iris_"
        f"widths_{hidden_width_tag}_"
        f"activation_{args.activation}_"
        f"mu_{mu:g}_"
        f"tau_{args.dal_target_ratio:g}_"
        f"steps_{num_steps}_"
        f"seed_{args.seed}.png"
    )

    figure.savefig(
        dal_plot_dir / dal_plot_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # ------------------------------------------------------
    # Save numerical results
    # ------------------------------------------------------

    result_payload = {
        "steps": dal_steps,
        "mu": np.asarray(
            mu,
            dtype=np.float64,
        ),
        "target_ratio": np.asarray(
            args.dal_target_ratio,
            dtype=np.float64,
        ),
    }

    for metric_name, values in (
        momentum_dal_results.items()
    ):
        result_payload[
            f"momentum_dal_{metric_name}"
        ] = values

    for fixed_eta, result in (
        fixed_momentum_results.items()
    ):
        eta_tag = (
            f"{fixed_eta:g}"
            .replace(".", "p")
            .replace("-", "m")
        )

        for metric_name, values in (
            result.items()
        ):
            result_payload[
                f"fixed_eta_{eta_tag}_{metric_name}"
            ] = values

    dal_result_name = (
        f"momentum_dal_fig19_"
        f"problem_iris_"
        f"widths_{hidden_width_tag}_"
        f"activation_{args.activation}_"
        f"mu_{mu:g}_"
        f"tau_{args.dal_target_ratio:g}_"
        f"steps_{num_steps}_"
        f"seed_{args.seed}.npz"
    )

    np.savez(
        dal_plot_dir / dal_result_name,
        **result_payload,
    )

    print(
        "\nMomentum-DAL Figure-19-style plot saved to:",
        dal_plot_dir / dal_plot_name,
    )

    print(
        "Momentum-DAL numerical results saved to:",
        dal_plot_dir / dal_result_name,
    )

    # This flag runs a stand-alone inexpensive experiment.
    # Do not continue into all of the SDE/PF simulations.
    if not DEBUG:
        writer.close()

    raise SystemExit(0)

# ==========================================================
# Run all algorithms
# ==========================================================

# Use independent Monte Carlo streams for the discrete
# reference and each continuous model.
trial_root_key = jax.random.fold_in(
    key,
    1000003,
)

reference_keys = jax.random.split(
    jax.random.fold_in(
        trial_root_key,
        0,
    ),
    num_runs,
)

sde_keys = {
    algo: jax.random.split(
        jax.random.fold_in(
            trial_root_key,
            algo_index + 1,
        ),
        num_runs,
    )
    for algo_index, algo in enumerate(
        algorithms[1:]
    )
}

diff_losses = {}
param_dist = {}
max_loss_diff = {}
max_loss_step = {}
avg_loss_results = {}
mean_trajectories = {}
final_samples = {}


for algo in algorithms:
    print(
        f"Running {algo}...",
        flush=True,
    )

    # ------------------------------------------------------
    # Discrete SGDM reference
    # ------------------------------------------------------
    if algo == REFERENCE_ALGORITHM:
        (
            reference_avg_losses,
            reference_all_losses,
            param_results_reference,
        ) = run_discrete_reference_trials(
            reference_keys,
            num_steps,
            eta,
            mu,
            sigma,
            init_param,
        )

        avg_loss_results[
            REFERENCE_ALGORITHM
        ] = reference_avg_losses

        param_results_reference = jnp.asarray(
            param_results_reference,
            dtype=jnp.float64,
        )

        mean_trajectories[
            REFERENCE_ALGORITHM
        ] = jnp.mean(
            param_results_reference,
            axis=0,
        )

        final_samples[
            REFERENCE_ALGORITHM
        ] = param_results_reference[
            :,
            -1,
            :,
        ]

    # ------------------------------------------------------
    # SDE models
    # ------------------------------------------------------
    else:
        optimizer_kwargs = {
            "stochastic": (
                stochastic_setting
            ),
        }

        complex_result = False

        if algo == COMPLEX_PF_ALGORITHM:
            optimizer_class = (
                ComplexMomentumPF
            )

            flow_objective_fn = (
                complex_loss_fn_flat
            )

            observable_fn = (
                complex_objective_observable
            )

            optimizer_kwargs = {
                "stochastic": (
                    stochastic_setting
                ),
            }

            complex_result = True

        elif algo == "Momentum-HA-SME":
            optimizer_class = (
                MomentumHASME
            )

            flow_objective_fn = (
                loss_fn_flat
            )

            observable_fn = (
                loss_fn_flat
            )

        elif algo == "Momentum-SME-1":
            optimizer_class = (
                MomentumFirstOrderFlow
            )

            flow_objective_fn = (
                loss_fn_flat
            )

            observable_fn = (
                loss_fn_flat
            )

        elif algo == "Momentum-SME-2":
            optimizer_class = (
                MomentumSecondOrderFlow
            )

            flow_objective_fn = (
                loss_fn_flat
            )

            observable_fn = (
                loss_fn_flat
            )

        else:
            raise ValueError(
                f"Unknown algorithm: {algo}"
            )

        (
            avg_losses,
            all_losses,
            param_results,
        ) = run_sde_trials(
            optimizer_class=optimizer_class,
            keys=sde_keys[algo],
            num_steps=num_steps,
            eta=eta,
            mu=mu,
            dt=dt,
            batch_size=args.sde_batch_size,
            cov_mat_func=cov_mat_func,
            init_param=init_param,
            brownian_tol=brownian_tol,
            flow_objective_fn=(
                flow_objective_fn
            ),
            observable_fn=observable_fn,
            optimizer_kwargs=(
                optimizer_kwargs
            ),
        )

        avg_loss_results[
            algo
        ] = avg_losses

        if complex_result:
            param_results = jnp.asarray(
                param_results,
                dtype=jnp.complex128,
            )
        else:
            param_results = jnp.asarray(
                param_results,
                dtype=jnp.float64,
            )

        mean_trajectories[
            algo
        ] = jnp.mean(
            param_results,
            axis=0,
        )

        final_samples[
            algo
        ] = param_results[
            :,
            -1,
            :,
        ]

        diff_losses[algo] = jnp.abs(
            avg_losses
            - reference_avg_losses
        )

        max_loss_diff[algo] = jnp.max(
            diff_losses[algo]
        )

        max_index = int(
            jnp.argmax(
                diff_losses[algo]
            )
        )

        max_loss_step[algo] = (
            max_index + 1
        )

        mean_flow_params = jnp.mean(
            param_results,
            axis=0,
        )

        mean_reference_params = jnp.mean(
            param_results_reference,
            axis=0,
        )

        # This norm correctly includes both real and imaginary
        # parts when mean_flow_params is complex.
        param_dist[algo] = jnp.linalg.norm(
            mean_flow_params
            - mean_reference_params,
            axis=-1,
        )

    print(
        f"Running {algo} finished.",
        flush=True,
    )

# ==========================================================
# Compute GDM Figure-13-style diagnostics
# ==========================================================

fig13_gdm_results = None

if args.fig13_gdm:
    print(
        "\nComputing GDM Figure-13-style diagnostics...",
        flush=True,
    )

    # The reference trajectory contains:
    #
    #     x_1, x_2, ..., x_N.
    #
    # Construct:
    #
    #     x_0, x_1, ..., x_N.
    full_gdm_state_history = jnp.concatenate(
        [
            init_flat_param[None, :],
            param_results_reference[0],
        ],
        axis=0,
    )

    # For each transition:
    #
    #     x_k -> x_{k+1},
    #
    # initialize both local flows at x_k.
    gdm_start_states = (
        full_gdm_state_history[:-1]
    )

    gdm_target_states = (
        full_gdm_state_history[1:]
    )

    fig13_steps = np.arange(
        1,
        num_steps + 1,
        dtype=np.int64,
    )

    start_losses = []
    target_losses = []

    descent_endpoint_losses = []
    ascent_endpoint_losses = []

    start_lambda_0_values = []

    stability_real_values = []
    stability_imaginary_values = []
    stability_absolute_values = []

    gradient_projection_values = []
    characteristic_root_values = []

    for transition_index in range(
        num_steps
    ):
        start_x = (
            gdm_start_states[
                transition_index
            ]
        )

        target_x = (
            gdm_target_states[
                transition_index
            ]
        )

        # Local descending momentum-gradient flow.
        descent_endpoint = (
            jax.block_until_ready(
                fig13_local_momentum_gradient_flow_endpoint(
                    start_x,
                    -1.0,
                )
            )
        )

        # Local positive momentum-gradient flow.
        ascent_endpoint = (
            jax.block_until_ready(
                fig13_local_momentum_gradient_flow_endpoint(
                    start_x,
                    +1.0,
                )
            )
        )

        (
            start_lambda_0,
            stability_real,
            stability_imaginary,
            stability_absolute,
            gradient_projection,
            characteristic_root,
        ) = jax.block_until_ready(
            fig13_momentum_stability_coefficient(
                start_x
            )
        )

        start_loss = float(
            jax.block_until_ready(
                loss_fn_flat(
                    start_x
                )
            )
        )

        target_loss = float(
            jax.block_until_ready(
                loss_fn_flat(
                    target_x
                )
            )
        )

        descent_loss = float(
            jax.block_until_ready(
                loss_fn_flat(
                    descent_endpoint
                )
            )
        )

        ascent_loss = float(
            jax.block_until_ready(
                loss_fn_flat(
                    ascent_endpoint
                )
            )
        )

        start_losses.append(
            start_loss
        )

        target_losses.append(
            target_loss
        )

        descent_endpoint_losses.append(
            descent_loss
        )

        ascent_endpoint_losses.append(
            ascent_loss
        )

        stability_real_values.append(
            float(stability_real)
        )

        stability_imaginary_values.append(
            float(stability_imaginary)
        )

        stability_absolute_values.append(
            float(stability_absolute)
        )

        gradient_projection_values.append(
            float(gradient_projection)
        )

        characteristic_root_values.append(
            complex(
                characteristic_root
            )
        )

        print(
            f"  transition {transition_index + 1}/"
            f"{num_steps}: "
            f"lambda_0(x_k)="
            f"{float(start_lambda_0):.6e}, "
            f"Re(sc_0)="
            f"{float(stability_real):.6e}",
            flush=True,
        )

    fig13_lambda_edge = (
        2.0
        * (1.0 + mu)
        / eta
    )

    fig13_gdm_results = {
        "steps": fig13_steps,

        "start_loss": np.asarray(
            start_losses,
            dtype=np.float64,
        ),

        "gdm_loss": np.asarray(
            target_losses,
            dtype=np.float64,
        ),

        "descent_loss": np.asarray(
            descent_endpoint_losses,
            dtype=np.float64,
        ),

        "ascent_loss": np.asarray(
            ascent_endpoint_losses,
            dtype=np.float64,
        ),

        "start_lambda_0": np.asarray(
            start_lambda_0_values,
            dtype=np.float64,
        ),

        "stability_real": np.asarray(
            stability_real_values,
            dtype=np.float64,
        ),

        "stability_imaginary": np.asarray(
            stability_imaginary_values,
            dtype=np.float64,
        ),

        "stability_absolute": np.asarray(
            stability_absolute_values,
            dtype=np.float64,
        ),

        "gradient_projection": np.asarray(
            gradient_projection_values,
            dtype=np.float64,
        ),

        "characteristic_root": np.asarray(
            characteristic_root_values,
            dtype=np.complex128,
        ),

        "gdm_lambda_0": np.asarray(
            start_lambda_0_values,
            dtype=np.float64,
        ),

        "lambda_edge": float(
            2.0 * (1.0 + mu) / eta
        ),
    }

    fig13_gdm_results[
        "above_edge"
    ] = (
        fig13_gdm_results[
            "start_lambda_0"
        ]
        > fig13_lambda_edge
    )

    fig13_gdm_results[
        "gdm_loss_increased"
    ] = (
        fig13_gdm_results[
            "gdm_loss"
        ]
        > fig13_gdm_results[
            "start_loss"
        ]
    )


# ==========================================================
# Hessian lambda_0 along the GDM trajectory
# ==========================================================

gdm_lambda_0_history = None
gdm_edge_ratio_history = None
gdm_above_edge_mask = None
gdm_hessian_step_values = None
gdm_lambda_edge = None

if args.discrete_method == "gdm":
    # param_results_reference has shape:
    #
    #     (1, num_steps, parameter_dimension)
    #
    # and contains:
    #
    #     x_1, x_2, ..., x_N.
    #
    # Add x_0 to construct the full GDM trajectory.
    full_gdm_state_history = jnp.concatenate(
        [
            init_flat_param[None, :],
            param_results_reference[0],
        ],
        axis=0,
    )

    gdm_hessian_step_values = np.arange(
        0,
        num_steps + 1,
        dtype=np.int64,
    )

    lambda_0_values = []

    print(
        "\nComputing lambda_0 along the GDM trajectory...",
        flush=True,
    )

    for state_index in range(
        int(
            full_gdm_state_history.shape[0]
        )
    ):
        lambda_0_value = float(
            jax.block_until_ready(
                largest_hessian_eigenvalue(
                    full_gdm_state_history[
                        state_index
                    ]
                )
            )
        )

        if not np.isfinite(
            lambda_0_value
        ):
            raise FloatingPointError(
                "A non-finite largest Hessian eigenvalue "
                f"was found at GDM state x_{state_index}."
            )

        lambda_0_values.append(
            lambda_0_value
        )

        print(
            f"  x_{state_index}: "
            f"lambda_0={lambda_0_value:.12e}",
            flush=True,
        )

    gdm_lambda_0_history = np.asarray(
        lambda_0_values,
        dtype=np.float64,
    )

    # Positive-curvature GDM edge:
    #
    #     eta * lambda_0 = 2 * (1 + mu).
    gdm_lambda_edge = (
        2.0
        * (1.0 + mu)
        / eta
    )

    gdm_edge_ratio_history = (
        eta
        * gdm_lambda_0_history
        / (
            2.0
            * (1.0 + mu)
        )
    )

    gdm_above_edge_mask = (
        gdm_lambda_0_history
        > gdm_lambda_edge
    )

    above_edge_steps = (
        gdm_hessian_step_values[
            gdm_above_edge_mask
        ]
    )

    print(
        "\nGDM lambda_0 summary:"
    )

    print(
        "  momentum edge lambda:",
        f"{gdm_lambda_edge:.12e}",
    )

    print(
        "  minimum lambda_0:",
        f"{np.min(gdm_lambda_0_history):.12e}",
    )

    print(
        "  maximum lambda_0:",
        f"{np.max(gdm_lambda_0_history):.12e}",
    )

    print(
        "  minimum edge ratio:",
        f"{np.min(gdm_edge_ratio_history):.12e}",
    )

    print(
        "  maximum edge ratio:",
        f"{np.max(gdm_edge_ratio_history):.12e}",
    )

    print(
        "  states above the momentum edge:",
        above_edge_steps.tolist(),
    )

# ==========================================================
# Compute two-sided momentum-recurrence local errors
# ==========================================================

recurrence_local_results = {}

if args.recurrence_local_errors:
    print(
        "\nComputing two-sided GDM recurrence predictions...",
        flush=True,
    )

    recurrence_root_key = jax.random.fold_in(
        trial_root_key,
        202313,
    )

    for recurrence_algo_index, algo in enumerate(
        algorithms[1:]
    ):
        print(
            f"Running recurrence predictions for {algo}...",
            flush=True,
        )

        algo_recurrence_key = jax.random.fold_in(
            recurrence_root_key,
            recurrence_algo_index,
        )

        recurrence_local_results[algo] = (
            run_local_gdm_recurrence_predictions(
                algo=algo,
                reference_params=(
                    param_results_reference
                ),
                init_flat_param=(
                    init_flat_param
                ),
                eta=eta,
                mu=mu,
                dt=dt,
                cov_mat_func=(
                    cov_mat_func
                ),
                brownian_tol=(
                    brownian_tol
                ),
                batch_size=(
                    args.sde_batch_size
                ),
                root_key=(
                    algo_recurrence_key
                ),
            )
        )

        print(
            f"Finished recurrence predictions for {algo}.",
            flush=True,
        )


# ==========================================================
# Verify that all SDE results were computed
# ==========================================================

expected_sde_algorithms = algorithms[1:]

missing_algorithms = [
    algo
    for algo in expected_sde_algorithms
    if algo not in diff_losses
]

if not args.figure13_only:
    if missing_algorithms:
        raise RuntimeError(
            "Weak errors were not computed for the following models: "
            f"{missing_algorithms}"
        )

print(
    "\nAlgorithms with computed weak errors:",
    list(diff_losses.keys()),
)


# ==========================================================
# Print maximal weak errors
# ==========================================================

print(
    "\nMaximum weak loss errors:"
)

for algo in expected_sde_algorithms:
    error_value = float(
        max_loss_diff[algo]
    )

    error_step = max_loss_step[algo]
    error_time = eta * error_step

    print(
        f"{algo}: "
        f"{error_value:.12e} "
        f"(step {error_step}, "
        f"time {error_time:.12e})"
    )

# ==========================================================
# Pointwise weak-error plot
# ==========================================================

plot_dir = Path(
    args.plot_dir
)

plot_dir.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# Plot the GDM analogue of Rosca Figure 13
# ==========================================================

if fig13_gdm_results is not None:
    fig13_start = (
        args.fig13_start_step
    )

    fig13_end = (
        num_steps
        if args.fig13_end_step is None
        else min(
            args.fig13_end_step,
            num_steps,
        )
    )

    if fig13_start > num_steps:
        raise ValueError(
            "--fig13_start_step exceeds --num_steps."
        )

    fig13_slice = slice(
        fig13_start - 1,
        fig13_end,
    )

    plot_steps = (
        fig13_gdm_results[
            "steps"
        ][fig13_slice]
    )

    above_edge = (
        fig13_gdm_results[
            "above_edge"
        ][fig13_slice]
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.1,
                1.1,
                0.9,
            ],
        },
    )

    lambda_axis = axes[0]
    loss_axis = axes[1]
    stability_axis = axes[2]

    # ------------------------------------------------------
    # Largest Hessian eigenvalue
    # ------------------------------------------------------

    lambda_axis.plot(
        plot_steps,
        gdm_lambda_0_history[1:][fig13_slice],
        color="black",
        linewidth=2.4,
        label=(
            r"GDM: $\lambda_0(x_k)$"
        ),
    )

    lambda_axis.axhline(
        gdm_lambda_edge,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=(
            r"$\lambda_{\mathrm{edge}}"
            r"=2(1+\mu)/\eta$"
        ),
    )

    lambda_axis.set_ylabel(
        r"$\lambda_0$"
    )

    lambda_axis.set_title(
        "Stability coefficients and training loss along GDM"
    )

    lambda_axis.grid(
        True,
        alpha=0.3,
    )

    lambda_axis.legend(
        loc="best",
        fontsize=9,
    )

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    loss_axis.plot(
        plot_steps,
        fig13_gdm_results[
            "gdm_loss"
        ][fig13_slice],
        color="black",
        linewidth=2.4,
        label=r"GDM: $f(x_{k+1})$",
    )

    loss_axis.plot(
        plot_steps,
        fig13_gdm_results[
            "descent_loss"
        ][fig13_slice],
        color="tab:blue",
        linewidth=2,
        label=(
            r"Negative gradient flow: "
            r"$f(\Phi^-_\eta(x_k))$"
        ),
    )

    # loss_axis.plot(
    #     plot_steps,
    #     fig13_gdm_results[
    #         "ascent_loss"
    #     ][fig13_slice],
    #     color="tab:red",
    #     linewidth=2,
    #     label=(
    #         r"Positive gradient flow: "
    #         r"$f(\Phi^+_\eta(x_k))$"
    #     ),
    # )

    loss_axis.set_ylabel(
        "Training loss"
    )

    loss_axis.grid(
        True,
        alpha=0.3,
    )

    loss_axis.legend(
        loc="best",
        fontsize=9,
    )

    # ------------------------------------------------------
    # Leading stability coefficient
    # ------------------------------------------------------

    stability_axis.plot(
        plot_steps,
        fig13_gdm_results[
            "stability_real"
        ][fig13_slice],
        color="tab:purple",
        linewidth=2.4,
        label=(
            r"$\operatorname{Re}[\mathrm{sc}_0(x_k)]$"
        ),
    )

    stability_axis.axhline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=1.5,
    )

    stability_axis.set_xlabel(
        "GDM step"
    )

    stability_axis.set_ylabel(
        r"$\operatorname{Re}[\mathrm{sc}_0]$"
    )

    stability_axis.grid(
        True,
        alpha=0.3,
    )

    stability_axis.legend(
        loc="best",
    )

    figure.tight_layout()

    fig13_plot_name = (
        f"gdm_rosca_fig13_"
        f"problem_{args.problem}_"
        f"mu_{mu:g}_"
        f"eta_{eta:g}_"
        f"steps_{num_steps}_"
        f"plot_steps_{fig13_start}_"
        f"to_{fig13_end}_"
        f"seed_{args.seed}.png"
    )

    figure.savefig(
        plot_dir / fig13_plot_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "\nGDM Figure-13-style plot saved to:",
        plot_dir / fig13_plot_name,
    )
# ==========================================================
# Largest Hessian eigenvalue along GDM
# ==========================================================

if gdm_lambda_0_history is not None:
    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    axis.plot(
        gdm_hessian_step_values,
        gdm_lambda_0_history,
        color="tab:blue",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=(
            r"$\lambda_0(x_k)"
            r"=\lambda_{\max}(\nabla^2 f(x_k))$"
        ),
    )

    # Momentum edge:
    #
    #     lambda_0 = 2(1 + mu) / eta.
    axis.axhline(
        gdm_lambda_edge,
        color="black",
        linestyle="--",
        linewidth=2,
        label=(
            r"$2(1+\mu)/\eta$"
            " (momentum edge)"
        ),
    )

    # Highlight areas above the edge.
    axis.fill_between(
        gdm_hessian_step_values,
        gdm_lambda_edge,
        gdm_lambda_0_history,
        where=gdm_above_edge_mask,
        interpolate=True,
        color="tab:red",
        alpha=0.22,
        label=(
            r"$\eta\lambda_0(x_k)"
            r">2(1+\mu)$"
        ),
    )

    axis.set_xlabel(
        "GDM step"
    )

    axis.set_ylabel(
        r"Largest Hessian eigenvalue "
        r"$\lambda_0(x_k)$"
    )

    axis.set_title(
        "Largest Hessian eigenvalue along GDM"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    lambda_0_plot_name = (
        f"lambda_0_along_gdm_"
        f"problem_{args.problem}_"
        f"mu_{mu:g}_"
        f"eta_{eta:g}_"
        f"steps_{num_steps}_"
        f"seed_{args.seed}.png"
    )

    figure.savefig(
        plot_dir / lambda_0_plot_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "\nGDM lambda_0 plot saved to:",
        plot_dir / lambda_0_plot_name,
    )

# Filename tag describing the neural-network configuration.
nn_plot_tag = ""

if args.problem == "nn":
    hidden_width_tag = "x".join(
        str(width)
        for width in hidden_widths
    )

    nn_plot_tag = (
        f"widths_{hidden_width_tag}_"
        f"activation_{args.activation}_"
    )

else:
    hidden_width_tag = "quadratic"

step_values = np.arange(
    1,
    num_steps + 1,
    dtype=np.int64,
)

# ==========================================================
# Two-sided GDM recurrence-error plot
# ==========================================================

if recurrence_local_results:
    recurrence_plot_start = (
        args.local_error_start_step
    )

    recurrence_plot_end = (
        num_steps
        if args.local_error_end_step is None
        else min(
            args.local_error_end_step,
            num_steps,
        )
    )

    if recurrence_plot_start > num_steps:
        raise ValueError(
            "--local_error_start_step exceeds --num_steps."
        )

    recurrence_plot_slice = slice(
        recurrence_plot_start - 1,
        recurrence_plot_end,
    )

    recurrence_plot_steps = (
        step_values[
            recurrence_plot_slice
        ]
    )

    recurrence_styles = {
        COMPLEX_PF_ALGORITHM: {
            "color": "tab:purple",
            "linestyle": "-",
        },
        "Momentum-HA-SME": {
            "color": "tab:blue",
            "linestyle": "-",
        },
        "Momentum-SME-1": {
            "color": "tab:orange",
            "linestyle": "--",
        },
        "Momentum-SME-2": {
            "color": "tab:green",
            "linestyle": "-.",
        },
    }

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        sharex=True,
    )

    parameter_axis = axes[0]
    loss_axis = axes[1]

    maximum_absolute_loss_error = 0.0

    for algo in algorithms[1:]:
        parameter_values = np.asarray(
            recurrence_local_results[
                algo
            ][
                "parameter_errors"
            ],
            dtype=np.float64,
        )[recurrence_plot_slice]

        loss_values = np.asarray(
            recurrence_local_results[
                algo
            ][
                "loss_errors"
            ],
            dtype=np.float64,
        )[recurrence_plot_slice]

        safe_parameter_values = np.maximum(
            parameter_values,
            np.finfo(
                np.float64
            ).tiny,
        )

        maximum_absolute_loss_error = max(
            maximum_absolute_loss_error,
            float(
                np.max(
                    np.abs(
                        loss_values
                    )
                )
            ),
        )

        style = recurrence_styles.get(
            algo,
            {
                "linestyle": "-",
            },
        )

        parameter_axis.plot(
            recurrence_plot_steps,
            safe_parameter_values,
            label=display_names[algo],
            linewidth=2,
            **style,
        )

        loss_axis.plot(
            recurrence_plot_steps,
            loss_values,
            label=display_names[algo],
            linewidth=2,
            **style,
        )

    # The raw recurrence residual is nonnegative and can span
    # several orders of magnitude.
    parameter_axis.set_yscale(
        "log"
    )

    parameter_axis.set_xlabel(
        "GDM iteration"
    )

    parameter_axis.set_ylabel(
        "Two-sided momentum recurrence error\n"
        r"$\|"
        r"\Phi_A^\eta(x_{k-1})"
        r"+\mu\Phi_A^{-\eta}(x_{k-1})"
        r"-x_k-\mu x_{k-2}"
        r"\|_2$"
    )

    parameter_axis.set_title(
        "Two-sided recurrence parameter error"
    )

    parameter_axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    parameter_axis.legend()

    loss_axis.axhline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=1.2,
    )

    # Preserve signs while revealing small loss errors.
    symlog_threshold = max(
        maximum_absolute_loss_error
        * 1e-6,
        1e-15,
    )

    loss_axis.set_yscale(
        "symlog",
        linthresh=symlog_threshold,
    )

    loss_axis.set_xlabel(
        "GDM iteration"
    )

    loss_axis.set_ylabel(
        "Signed reconstructed recurrence loss error\n"
        r"$f(x_k)+\mu f(x_{k-2})"
        r"-f(\mathbf{\Phi}_A^{+\eta}(x_{k-1}))"
        r"-\mu f(\mathbf{\Phi}_A^{-\eta}(x_{k-1}))$"
    )

    loss_axis.set_title(
        "Signed reconstructed recurrence loss error"
    )

    loss_axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    loss_axis.legend()

    figure.suptitle(
        "Two-sided local momentum-recurrence errors for GDM",
        fontsize=13,
    )

    figure.tight_layout()

    recurrence_plot_name = (
        f"gdm_two_sided_recurrence_errors_"
        f"problem_{args.problem}_"
        f"{nn_plot_tag}"
        f"mu_{mu:g}_"
        f"eta_{eta:g}_"
        f"steps_{num_steps}_"
        f"plot_steps_{recurrence_plot_start}_"
        f"to_{recurrence_plot_end}_"
        f"ratio_{args.eta_dt_ratio:g}_"
        f"seed_{args.seed}.png"
    )

    figure.savefig(
        plot_dir / recurrence_plot_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "\nTwo-sided recurrence-error plot saved to:",
        plot_dir / recurrence_plot_name,
    )

plt.figure(
    figsize=(8, 5)
)

for algo in algorithms[1:]:
    plot_values = np.asarray(
        diff_losses[algo],
        dtype=np.float64,
    )

    # Logarithmic axes cannot display exact zero.
    plot_values = np.maximum(
        plot_values,
        np.finfo(np.float64).tiny,
    )

    plt.plot(
        step_values,
        plot_values,
        label=display_names[algo],
        linewidth=2,
    )

    maximum_index = (
        max_loss_step[algo] - 1
    )

    plt.scatter(
        step_values[maximum_index],
        plot_values[maximum_index],
        s=35,
        zorder=5,
    )

plt.yscale(
    "log"
)

plt.xlabel(
    f"{reference_short_name} step"
)

plt.ylabel(
    r"$|\widehat{\mathbb{E}}[f(X_{k\eta})]"
    r"-\widehat{\mathbb{E}}[f(x_k)]|$"
)

plt.title(
    "Pointwise weak approximation error"
)

plt.legend()

plt.grid(
    True,
    which="both",
    alpha=0.3,
)

plt.tight_layout()

pointwise_plot_name = (
    f"objective_approximation_error_"
    f"method_{args.discrete_method}_"
    f"problem_{args.problem}_"
    f"{nn_plot_tag}"
    f"widths_{hidden_width_tag}_"
    f"activation_{args.activation}_"
    f"mu_{mu:g}_"
    f"eta_{eta:g}_"
    f"sigma_{sigma:g}_"
    f"steps_{num_steps}_"
    f"runs_{num_runs}_"
    f"ratio_{args.eta_dt_ratio:g}_"
    f"seed_{args.seed}.png"
)

plt.savefig(
    plot_dir / pointwise_plot_name,
    dpi=300,
)

plt.close()


# ==========================================================
# Signed difference between HA-SME and SME-2 weak errors
# ==========================================================
#
# The individual weak errors remain absolute:
#
#   e_HA(k)
#     = |E[f(X_HA)] - E[f(x_SGDM)]|
#
#   e_SME2(k)
#     = |E[f(X_SME2)] - E[f(x_SGDM)]|
#
# We plot their signed difference:
#
#   Delta(k) = e_HA(k) - e_SME2(k)
#
# Interpretation:
#
#   Delta(k) < 0  -> HA-SME has smaller weak error
#   Delta(k) > 0  -> SME-2 has smaller weak error

ha_algorithm = "Momentum-HA-SME"
sme2_algorithm = "Momentum-SME-2"

if (
    ha_algorithm in diff_losses
    and sme2_algorithm in diff_losses
):
    ha_weak_error = np.asarray(
        diff_losses[ha_algorithm],
        dtype=np.float64,
    )

    sme2_weak_error = np.asarray(
        diff_losses[sme2_algorithm],
        dtype=np.float64,
    )

    # Do not take the absolute value of this subtraction.
    weak_error_difference = (
        ha_weak_error
        - sme2_weak_error
    )

    ha_better = (
        weak_error_difference < 0.0
    )

    sme2_better = (
        weak_error_difference > 0.0
    )

    # Use a symmetric logarithmic scale because the values
    # can be positive or negative and may span several
    # orders of magnitude.
    maximum_absolute_difference = np.max(
        np.abs(
            weak_error_difference
        )
    )

    symlog_threshold = max(
        maximum_absolute_difference * 1e-4,
        1e-12,
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    # Main signed-difference curve.
    plt.plot(
        step_values,
        weak_error_difference,
        color="black",
        linewidth=2.2,
        label=(
            r"$e_{\rm HA}(k)"
            r"-e_{\rm SME2}(k)$"
        ),
    )

    # Blue shading means HA-SME has the lower error.
    plt.fill_between(
        step_values,
        weak_error_difference,
        0.0,
        where=ha_better,
        interpolate=True,
        color="tab:blue",
        alpha=0.25,
        label="HA-SME has lower weak error",
    )

    # Green shading means SME-2 has the lower error.
    plt.fill_between(
        step_values,
        weak_error_difference,
        0.0,
        where=sme2_better,
        interpolate=True,
        color="tab:green",
        alpha=0.25,
        label="SME-2 has lower weak error",
    )

    # Equality reference line.
    plt.axhline(
        0.0,
        color="gray",
        linestyle="--",
        linewidth=1.5,
    )

    plt.yscale(
        "symlog",
        linthresh=symlog_threshold,
    )

    plt.xlabel(
        f"{reference_short_name} step"
    )

    plt.ylabel(
        r"$e_{\rm HA}(k)-e_{\rm SME2}(k)$"
    )

    plt.title(
        "Difference between HA-SME and SME-2 weak errors"
    )

    plt.legend()

    plt.grid(
        True,
        which="both",
        alpha=0.3,
    )

    plt.tight_layout()

    if args.problem == "nn":
        model_tag = (
            f"widths_{hidden_width_tag}_"
            f"activation_{args.activation}_"
        )
    else:
        model_tag = ""

    weak_error_difference_plot_name = (
        f"ha_minus_sme2_objective_error_"
        f"method_{args.discrete_method}_"
        f"problem_{args.problem}_"
        f"{nn_plot_tag}"
        f"{model_tag}"
        f"mu_{mu:g}_"
        f"eta_{eta:g}_"
        f"sigma_{sigma:g}_"
        f"steps_{num_steps}_"
        f"runs_{num_runs}_"
        f"ratio_{args.eta_dt_ratio:g}_"
        f"seed_{args.seed}.png"
    )

    plt.savefig(
        plot_dir
        / weak_error_difference_plot_name,
        dpi=300,
    )

    plt.close()

    # ------------------------------------------------------
    # Print which model dominates over which step intervals
    # ------------------------------------------------------

    def get_step_intervals(mask, steps):
        """
        Convert a Boolean mask into inclusive step intervals.

        Example:
            [True, True, False, True]
        becomes:
            [(1, 2), (4, 4)]
        """

        mask = np.asarray(
            mask,
            dtype=bool,
        )

        if mask.size == 0:
            return []

        padded_mask = np.concatenate(
            [
                np.asarray([False]),
                mask,
                np.asarray([False]),
            ]
        )

        changes = np.diff(
            padded_mask.astype(
                np.int32
            )
        )

        start_indices = np.where(
            changes == 1
        )[0]

        end_indices = (
            np.where(
                changes == -1
            )[0]
            - 1
        )

        return [
            (
                int(steps[start_index]),
                int(steps[end_index]),
            )
            for start_index, end_index in zip(
                start_indices,
                end_indices,
            )
        ]

    ha_better_intervals = get_step_intervals(
        ha_better,
        step_values,
    )

    sme2_better_intervals = get_step_intervals(
        sme2_better,
        step_values,
    )

    print(
        "\nHA-SME versus SME-2 weak-error comparison:"
    )

    print(
        "  Difference plotted as:"
    )

    print(
        "    HA-SME weak error - SME-2 weak error"
    )

    print(
        "  Negative values: HA-SME is better"
    )

    print(
        "  Positive values: SME-2 is better"
    )

    print(
        "  Steps where HA-SME has lower error:",
        int(np.sum(ha_better)),
        f"out of {num_steps}",
    )

    print(
        "  HA-SME dominance intervals:",
        ha_better_intervals,
    )

    print(
        "  Steps where SME-2 has lower error:",
        int(np.sum(sme2_better)),
        f"out of {num_steps}",
    )

    print(
        "  SME-2 dominance intervals:",
        sme2_better_intervals,
    )

    print(
        "  Minimum difference "
        "(strongest HA-SME advantage):",
        f"{float(np.min(weak_error_difference)):.12e}",
        "at step",
        int(
            step_values[
                np.argmin(
                    weak_error_difference
                )
            ]
        ),
    )

    print(
        "  Maximum difference "
        "(strongest SME-2 advantage):",
        f"{float(np.max(weak_error_difference)):.12e}",
        "at step",
        int(
            step_values[
                np.argmax(
                    weak_error_difference
                )
            ]
        ),
    )

    print(
        "  Plot saved to:",
        plot_dir
        / weak_error_difference_plot_name,
    )

else:
    print(
        "\nSkipping HA-SME minus SME-2 weak-error plot: "
        "one or both models were not run."
    )

# ==========================================================
# Two-dimensional trajectory plot
# ==========================================================

if args.problem == "quadratic":
    plt.figure(
        figsize=(7, 6)
    )

    trajectory_styles = {
        REFERENCE_ALGORITHM: {
            "color": "black",
            "linestyle": "-",
            "linewidth": 2.5,
        },
        COMPLEX_PF_ALGORITHM: {
            "color": "tab:purple",
            "linestyle": ":",
            "linewidth": 2.5,
        },
        "Momentum-HA-SME": {
            "color": "tab:blue",
            "linestyle": "-",
            "linewidth": 2.0,
        },
        "Momentum-SME-1": {
            "color": "tab:orange",
            "linestyle": "--",
            "linewidth": 2.0,
        },
        "Momentum-SME-2": {
            "color": "tab:green",
            "linestyle": "-.",
            "linewidth": 2.0,
        },
    }

    for algo in algorithms:
        trajectory = np.asarray(
            mean_trajectories[algo]
        )

        if np.iscomplexobj(
            trajectory
        ):
            # Plot the real projection in parameter space. The
            # imaginary norm is shown in a separate plot.
            trajectory_to_plot = np.real(
                trajectory
            )
        else:
            trajectory_to_plot = trajectory

        # Include the common initial point.
        full_trajectory = np.vstack(
            [
                np.asarray(
                    init_flat_param
                )[None, :],
                trajectory_to_plot,
            ]
        )

        plt.plot(
            full_trajectory[:, 0],
            full_trajectory[:, 1],
            label=display_names[algo],
            **trajectory_styles[algo],
        )

    plt.scatter(
        float(init_flat_param[0]),
        float(init_flat_param[1]),
        color="purple",
        marker="o",
        s=70,
        label="initial point",
        zorder=6,
    )

    plt.scatter(
        float(quadratic_target[0]),
        float(quadratic_target[1]),
        color="black",
        marker="*",
        s=160,
        label="minimizer",
        zorder=7,
    )

    plt.xlabel(
        r"$\theta_1$"
    )

    plt.ylabel(
        r"$\theta_2$"
    )

    plt.title(
        "Mean stochastic optimization trajectories"
    )

    plt.legend()
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.axis("equal")
    plt.tight_layout()

    trajectory_plot_name = (
        f"mean_trajectory_"
        f"method_{args.discrete_method}_"
        f"problem_{args.problem}_"
        f"{nn_plot_tag}"
        f"mu_{mu:g}_"
        f"eta_{eta:g}_"
        f"sigma_{sigma:g}_"
        f"steps_{num_steps}_"
        f"runs_{num_runs}_"
        f"ratio_{args.eta_dt_ratio:g}_"
        f"seed_{args.seed}.png"
    )

    plt.savefig(
        plot_dir / trajectory_plot_name,
        dpi=300,
    )

    plt.close()


# ==========================================================
# Mean parameter approximation error
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

for algo in algorithms[1:]:

    values = np.asarray(
        param_dist[algo],
        dtype=np.float64,
    )

    values = np.maximum(
        values,
        np.finfo(np.float64).tiny,
    )

    plt.plot(
        step_values,
        values,
        label=display_names[algo],
        linewidth=2,
    )

plt.yscale(
    "log"
)

plt.xlabel(
    f"{reference_short_name} step"
)

if stochastic_setting:
    parameter_error_title = (
        "Mean parameter error relative to SGDM"
    )

    parameter_error_ylabel = (
        r"$\|\widehat{\mathbb{E}}[X(k\eta)]"
        r"-\widehat{\mathbb{E}}[x_k]\|_2$"
    )

else:
    parameter_error_title = (
        "Global trajectory error relative to GDM"
    )

    parameter_error_ylabel = (
        r"$\|X(k\eta)-x_k\|_2$"
    )

plt.ylabel(
    parameter_error_ylabel
)

plt.title(
    parameter_error_title
)

plt.legend()
plt.grid(
    True,
    which="both",
    alpha=0.3,
)
plt.tight_layout()

parameter_plot_name = (
    f"parameter_error_"
    f"method_{args.discrete_method}_"
    f"problem_{args.problem}_"
    f"{nn_plot_tag}"
    f"mu_{mu:g}_"
    f"eta_{eta:g}_"
    f"sigma_{sigma:g}_"
    f"steps_{num_steps}_"
    f"runs_{num_runs}_"
    f"ratio_{args.eta_dt_ratio:g}_"
    f"seed_{args.seed}.png"
)

plt.savefig(
    plot_dir / parameter_plot_name,
    dpi=300,
)

plt.close()
# ==========================================================
# TensorBoard logging
# ==========================================================

if not DEBUG:
    for step in range(num_steps):
        writer.add_scalars(
            "Loss_Diff",
            {
                algo: np.asarray(
                    diff_losses[algo][step]
                )
                for algo in algorithms[1:]
            },
            step + 1,
        )

        writer.add_scalars(
            "Mean_Param_Error",
            {
                algo: np.asarray(
                    param_dist[algo][step]
                )
                for algo in algorithms[1:]
            },
            step + 1,
        )

    writer.add_scalars(
        "Max_Loss_Diff",
        {
            algo: np.asarray(
                max_loss_diff[algo]
            )
            for algo in algorithms[1:]
        },
        0,
    )

    writer.close()