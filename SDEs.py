import jax
import jax.numpy as jnp
import jax.random as jrandom
from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
import equinox as eqx
DEBUG = False
if DEBUG:
    from jax import config
    config.update("jax_disable_jit", True)
    print("Debugging...")

def symmetric_psd_sqrt(matrix, tol=1e-10):
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = jnp.linalg.eigh(matrix)

    # Only intended to remove small negative roundoff errors.
    eigvals = jnp.where(
        eigvals > -tol,
        jnp.maximum(eigvals, 0.0),
        jnp.nan,
    )

    return (
        eigvecs
        @ jnp.diag(jnp.sqrt(eigvals))
        @ eigvecs.T
    )


class MomentumHASME:
    def __init__(
        self,
        stepsize,
        momentum,
        f,
        cov_mat_func,
        brownian_tol=1e-6,
        stochastic=True,
    ):
        if stepsize <= 0.0:
            raise ValueError(
                "stepsize must be positive."
            )

        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "momentum must satisfy 0 <= momentum < 1."
            )

        if brownian_tol <= 0.0:
            raise ValueError(
                "brownian_tol must be positive."
            )

        self.stepsize = stepsize
        self.momentum = momentum
        self.f = f
        self.grad = jax.grad(f)
        self.hessian = jax.hessian(f)
        self.cov_mat_func = cov_mat_func
        self.brownian_tol = brownian_tol
        self.stochastic = bool(
            stochastic
        )

    def compute_drift_and_diffusion(
        self,
        t,
        x,
        args,
    ):
        del t, args

        eta = jnp.asarray(
            self.stepsize,
            dtype=x.dtype,
        )

        mu = jnp.asarray(
            self.momentum,
            dtype=x.dtype,
        )

        grad = self.grad(x)
        hessian = self.hessian(x)

        # Symmetrize to suppress small asymmetries caused by
        # floating-point roundoff.
        hessian = 0.5 * (
            hessian + hessian.T
        )

        eigvals, U = jnp.linalg.eigh(hessian)

        # --------------------------------------------------
        # Runtime HA-SME domain check
        # --------------------------------------------------
        #
        # The real analytic HA-SME branch requires
        #
        #   eta * ||H(x)||_2
        #       < (1 - sqrt(mu))^2
        #
        # at every state where the coefficients are
        # evaluated, not only at the initial state.

        scaled_hessian_norm = (
            eta * jnp.max(jnp.abs(eigvals))
        )

        allowed_radius = (
            1.0 - jnp.sqrt(mu)
        ) ** 2

        invalid_domain = (
            ~jnp.isfinite(scaled_hessian_norm)
            | ~jnp.isfinite(allowed_radius)
            | (
                scaled_hessian_norm
                >= allowed_radius
            )
        )

        eigvals = eqx.error_if(
            eigvals,
            invalid_domain,
            (
                "Momentum HA-SME left its real-valued "
                "domain. Required at every evaluated state: "
                "eta * ||H(x)||_2 < "
                "(1 - sqrt(mu))^2. Reduce eta."
            ),
        )
        # q_i = eta * lambda_i
        q = eta * eigvals

        a = 1.0 + mu - q

        discriminant = (
            a**2 - 4.0 * mu
        )

        # The runtime domain check guarantees a nonnegative
        # discriminant mathematically. maximum only removes
        # tiny negative roundoff values.
        sqrt_discriminant = jnp.sqrt(
            jnp.maximum(
                discriminant,
                jnp.asarray(
                    0.0,
                    dtype=x.dtype,
                ),
            )
        )

        # Larger root:
        #
        # r_i = f(mu, eta * lambda_i)
        r = 0.5 * (
            a + sqrt_discriminant
        )

        # --------------------------------------------------
        # HA-SME drift
        # --------------------------------------------------
        #
        # b(x) = U diag(log(r_i) / q_i) U^T grad f(x).
        #
        # At q_i = 0, use the Taylor expansion
        #
        # log(r_i) / q_i
        #   = -1/(1-mu)
        #     - q_i(1+mu)/(2(1-mu)^3)
        #     + O(q_i^2).
        small_q = (
            jnp.abs(q) < 1e-6
        )

        drift_ratio_taylor = (
            -1.0 / (1.0 - mu)
            - q
            * (1.0 + mu)
            / (
                2.0
                * (1.0 - mu) ** 3
            )
        )

        # Use a safe denominator so the inactive branch of
        # jnp.where does not contain an explicit 0/0.
        safe_q = jnp.where(
            small_q,
            jnp.ones_like(q),
            q,
        )

        drift_ratio_direct = (
            jnp.log(r) / safe_q
        )

        drift_ratio = jnp.where(
            small_q,
            drift_ratio_taylor,
            drift_ratio_direct,
        )

        grad_in_basis = U.T @ grad

        drift = U @ (
            drift_ratio * grad_in_basis
        )

        # A deterministic GDM comparison is an ODE. Do not
        # construct or factorize a diffusion covariance.
        if not self.stochastic:
            zero_diffusion = jnp.zeros(
                (
                    x.shape[0],
                    x.shape[0],
                ),
                dtype=x.dtype,
            )

            return (
                drift,
                zero_diffusion,
            )

        # --------------------------------------------------
        # HA-SME covariance
        # --------------------------------------------------
        sigma = self.cov_mat_func(x)

        # Remove small numerical asymmetries in Sigma.
        sigma = 0.5 * (
            sigma + sigma.T
        )

        sigma_hat = (
            U.T @ sigma @ U
        )

        # --------------------------------------------------
        # k -> infinity limit of the Section-IV piecewise
        # HA-SME covariance kernel.
        #
        # q_i = eta * lambda_i
        # r_i = f_mu(q_i)
        #
        # G_inf(q_i, q_j)
        #   = -(1-mu^2) log(r_i r_j)
        #     / [
        #         (1-mu)(1-mu^2)(q_i+q_j)
        #         + mu(q_i^2+q_j^2)
        #         - (1+mu^2)q_i q_j
        #       ].
        #
        # Its removable value at q_i=q_j=0 is
        #
        #     G_inf(0, 0) = 1 / (1-mu)^2.
        # --------------------------------------------------

        q_i = q[:, None]
        q_j = q[None, :]

        # Computing log(r_i) + log(r_j) is slightly more
        # stable than forming r_i * r_j first.
        log_r = jnp.log(r)

        log_root_product = (
            log_r[:, None]
            + log_r[None, :]
        )

        denominator = (
            (1.0 - mu)
            * (1.0 - mu**2)
            * (q_i + q_j)
            + mu * (q_i**2 + q_j**2)
            - (1.0 + mu**2) * q_i * q_j
        )

        numerator = (
            -(1.0 - mu**2)
            * log_root_product
        )

        # The formula is 0/0 at the origin. Use its
        # continuous Section-IV value there.
        near_origin = (
            jnp.maximum(
                jnp.abs(q_i),
                jnp.abs(q_j),
            )
            < 1e-6
        )

        # Avoid evaluating an explicit 0/0 in the inactive
        # branch of jnp.where.
        safe_denominator = jnp.where(
            near_origin,
            jnp.ones_like(denominator),
            denominator,
        )

        kernel_direct = (
            numerator / safe_denominator
        )

        kernel_at_origin = (
            jnp.ones_like(kernel_direct)
            / (1.0 - mu) ** 2
        )

        kernel = jnp.where(
            near_origin,
            kernel_at_origin,
            kernel_direct,
        )

        # The mathematical kernel is symmetric; suppress
        # floating-point asymmetry before using it.
        kernel = 0.5 * (
            kernel + kernel.T
        )
        
        s_matrix = (
            eta * sigma_hat * kernel
        )

        diffusion_cov = (
            U @ s_matrix @ U.T
        )

        # Suppress floating-point asymmetry before taking the
        # symmetric PSD square root.
        diffusion_cov = 0.5 * (
            diffusion_cov
            + diffusion_cov.T
        )

        diffusion = symmetric_psd_sqrt(
            diffusion_cov
        )

        return drift, diffusion

    def drift(
        self,
        t,
        x,
        args,
    ):
        drift, _ = (
            self.compute_drift_and_diffusion(
                t,
                x,
                args,
            )
        )

        return drift

    def diffusion(
        self,
        t,
        x,
        args,
    ):
        _, diffusion = (
            self.compute_drift_and_diffusion(
                t,
                x,
                args,
            )
        )

        return diffusion

    def solve(
        self,
        init_x,
        t,
        dt0,
        ts,
        key,
        t0=0,
    ):
        if self.stochastic:
            brownian_motion = (
                VirtualBrownianTree(
                    t0,
                    t,
                    tol=self.brownian_tol,
                    shape=init_x.shape,
                    key=key,
                )
            )

            terms = MultiTerm(
                ODETerm(
                    self.drift
                ),
                ControlTerm(
                    self.diffusion,
                    brownian_motion,
                ),
            )

        else:
            # Real deterministic HA-SME.
            terms = ODETerm(
                self.drift
            )

        return diffeqsolve(
            terms,
            Euler(),
            t0,
            t,
            dt0,
            y0=init_x,
            saveat=SaveAt(
                ts=ts
            ),
            max_steps=None,
        )

def complex_symmetric_matrix_sqrt(matrix):
    """
    Principal matrix square root of a complex symmetric matrix.

    For a diagonalizable matrix

        A = V diag(lambda_i) V^{-1},

    this constructs

        sqrt(A) = V diag(sqrt(lambda_i)) V^{-1}.

    The result is symmetrized numerically. This is an
    experimental complex factorization, not a real PSD
    square root.
    """

    matrix = 0.5 * (
        matrix + matrix.T
    )

    eigenvalues, eigenvectors = jnp.linalg.eig(
        matrix
    )

    inverse_eigenvectors = jnp.linalg.inv(
        eigenvectors
    )

    root = (
        eigenvectors
        @ jnp.diag(
            jnp.sqrt(eigenvalues)
        )
        @ inverse_eigenvectors
    )

    return 0.5 * (
        root + root.T
    )


class ComplexMomentumPF:
    """
    Experimental complex principal flow for momentum methods.

    Deterministic mode:
        stochastic=False

    Stochastic mode:
        stochastic=True

    The supplied objective must map a complex parameter vector
    to a complex scalar and must be holomorphic in the region
    visited by the solver.

    Conventions:
      - principal complex square root;
      - principal complex logarithm;
      - general eigendecomposition H = V Lambda V^{-1};
      - real Brownian motion multiplied by a complex diffusion
        matrix in stochastic mode.

    The stochastic complex extension is experimental.
    """

    def __init__(
        self,
        stepsize,
        momentum,
        f,
        cov_mat_func,
        brownian_tol=1e-6,
        stochastic=True,
    ):
        if stepsize <= 0.0:
            raise ValueError(
                "stepsize must be positive."
            )

        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "momentum must satisfy 0 <= momentum < 1."
            )

        if brownian_tol <= 0.0:
            raise ValueError(
                "brownian_tol must be positive."
            )

        self.stepsize = stepsize
        self.momentum = momentum
        self.f = f
        self.cov_mat_func = cov_mat_func
        self.brownian_tol = brownian_tol
        self.stochastic = bool(
            stochastic
        )

        # f is complex-valued and holomorphic.
        self.grad = jax.grad(
            f,
            holomorphic=True,
        )

        # Holomorphic Jacobian of the complex gradient.
        self.hessian = jax.jacfwd(
            self.grad,
            holomorphic=True,
        )

    def compute_drift_and_diffusion(
        self,
        t,
        x,
        args,
    ):
        del t, args

        if not jnp.issubdtype(
            x.dtype,
            jnp.complexfloating,
        ):
            raise TypeError(
                "ComplexMomentumHASME requires a complex state."
            )

        real_dtype = (
            jnp.float64
            if x.dtype == jnp.complex128
            else jnp.float32
        )

        eta = jnp.asarray(
            self.stepsize,
            dtype=real_dtype,
        )

        mu = jnp.asarray(
            self.momentum,
            dtype=real_dtype,
        )

        gradient = self.grad(x)

        hessian = self.hessian(x)

        # A holomorphic Hessian is complex symmetric rather
        # than Hermitian.
        hessian = 0.5 * (
            hessian + hessian.T
        )

        # General complex eigendecomposition:
        #
        #     H = V diag(lambda) V^{-1}.
        #
        # Unlike the real symmetric case, V is not generally
        # orthogonal.
        eigenvalues, eigenvectors = jnp.linalg.eig(
            hessian
        )

        inverse_eigenvectors = jnp.linalg.inv(
            eigenvectors
        )

        # q_i = eta * lambda_i.
        q = (
            eta.astype(x.dtype)
            * eigenvalues
        )

        a = (
            1.0
            + mu.astype(x.dtype)
            - q
        )

        discriminant = (
            a**2
            - 4.0 * mu.astype(x.dtype)
        )

        # Do not clip the discriminant. Negative real values
        # naturally produce complex square roots.
        sqrt_discriminant = jnp.sqrt(
            discriminant
        )
        
        root = 0.5 * (
            a + sqrt_discriminant
        )

        # No positive floor is applied. If root crosses zero,
        # the logarithm is genuinely singular.
        log_root = jnp.log(
            root
        )

        # --------------------------------------------------
        # Complex drift
        # --------------------------------------------------

        small_q = (
            jnp.abs(q) < 1e-8
        )

        safe_q = jnp.where(
            small_q,
            jnp.ones_like(q),
            q,
        )

        drift_ratio_direct = (
            log_root / safe_q
        )

        drift_ratio_taylor = (
            -1.0 / (1.0 - mu)
            - q
            * (1.0 + mu)
            / (
                2.0
                * (1.0 - mu) ** 3
            )
        )

        drift_ratio = jnp.where(
            small_q,
            drift_ratio_taylor,
            drift_ratio_direct,
        )

        gradient_in_basis = (
            inverse_eigenvectors
            @ gradient
        )

        drift = (
            eigenvectors
            @ (
                drift_ratio
                * gradient_in_basis
            )
        )

        # In deterministic mode the complex PF is an ODE.
        # Do not evaluate the covariance or its complex matrix
        # square root.
        if not self.stochastic:
            zero_diffusion = jnp.zeros(
                (
                    x.shape[0],
                    x.shape[0],
                ),
                dtype=x.dtype,
            )

            return (
                drift,
                zero_diffusion,
            )

        # --------------------------------------------------
        # Complex diffusion covariance
        # --------------------------------------------------

        covariance = self.cov_mat_func(
            x
        ).astype(x.dtype)

        covariance = 0.5 * (
            covariance + covariance.T
        )

        # Generalization of U^T Sigma U to a non-orthogonal
        # eigenbasis:
        #
        #     Sigma_hat = V^{-1} Sigma V^{-T}.
        covariance_in_basis = (
            inverse_eigenvectors
            @ covariance
            @ inverse_eigenvectors.T
        )

        z = (
            root[:, None]
            * root[None, :]
        )

        denominator = (
            z
            - mu**2 / z
            + mu**2
            - 1.0
        )

        near_one = (
            jnp.abs(z - 1.0) < 1e-8
        )

        safe_denominator = jnp.where(
            near_one,
            jnp.ones_like(denominator),
            denominator,
        )

        kernel_direct = (
            jnp.log(z)
            / safe_denominator
        )

        kernel_limit = (
            jnp.ones_like(z)
            / (1.0 + mu**2)
        )

        kernel = jnp.where(
            near_one,
            kernel_limit,
            kernel_direct,
        )

        covariance_basis_result = (
            eta.astype(x.dtype)
            * covariance_in_basis
            * kernel
        )

        diffusion_covariance = (
            eigenvectors
            @ covariance_basis_result
            @ eigenvectors.T
        )

        diffusion_covariance = 0.5 * (
            diffusion_covariance
            + diffusion_covariance.T
        )

        diffusion = complex_symmetric_matrix_sqrt(
            diffusion_covariance
        )

        return (
            drift,
            diffusion,
        )

    def drift(
        self,
        t,
        x,
        args,
    ):
        drift, _ = (
            self.compute_drift_and_diffusion(
                t,
                x,
                args,
            )
        )

        return drift

    def diffusion(
        self,
        t,
        x,
        args,
    ):
        _, diffusion = (
            self.compute_drift_and_diffusion(
                t,
                x,
                args,
            )
        )

        return diffusion

    def solve(
        self,
        init_x,
        t,
        dt0,
        ts,
        key,
        t0=0,
    ):
        complex_dtype = (
            jnp.complex128
            if init_x.dtype == jnp.float64
            else jnp.complex64
        )

        complex_init = init_x.astype(
            complex_dtype
        )

        if self.stochastic:
            # Brownian motion is real. Multiplication by the
            # complex diffusion matrix produces a complex state.
            brownian_motion = VirtualBrownianTree(
                t0,
                t,
                tol=self.brownian_tol,
                shape=init_x.shape,
                key=key,
            )

            terms = MultiTerm(
                ODETerm(
                    self.drift
                ),
                ControlTerm(
                    self.diffusion,
                    brownian_motion,
                ),
            )

        else:
            # Deterministic complex principal flow.
            terms = ODETerm(
                self.drift
            )

        return diffeqsolve(
            terms,
            Euler(),
            t0,
            t,
            dt0,
            y0=complex_init,
            saveat=SaveAt(
                ts=ts
            ),
            max_steps=None,
        )


# Backward-compatible name.
ComplexMomentumHASME = (
    ComplexMomentumPF
)

class MomentumFirstOrderFlow:
    def __init__(
        self,
        stepsize,
        momentum,
        f,
        cov_mat_func,
        brownian_tol=1e-6,
        stochastic=True,
    ):
        if stepsize <= 0.0:
            raise ValueError(
                "stepsize must be positive."
            )

        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "momentum must satisfy 0 <= momentum < 1."
            )

        if brownian_tol <= 0.0:
            raise ValueError(
                "brownian_tol must be positive."
            )

        self.stepsize = stepsize
        self.momentum = momentum
        self.f = f
        self.grad = jax.grad(f)
        self.cov_mat_func = cov_mat_func
        self.brownian_tol = brownian_tol
        self.stochastic = bool(
            stochastic
        )

    def drift(self, t, x, args):
        mu = self.momentum
        return -self.grad(x) / (1.0 - mu)


    def diffusion(self, t, x, args):
        mu = self.momentum
        diffusion_cov = (
            self.stepsize
            * self.cov_mat_func(x)
            / (1.0 - mu)**2
        )
        return symmetric_psd_sqrt(diffusion_cov)

    def solve(
        self,
        init_x,
        t,
        dt0,
        ts,
        key,
        t0=0,
    ):
        if self.stochastic:
            brownian_motion = (
                VirtualBrownianTree(
                    t0,
                    t,
                    tol=self.brownian_tol,
                    shape=init_x.shape,
                    key=key,
                )
            )

            terms = MultiTerm(
                ODETerm(
                    self.drift
                ),
                ControlTerm(
                    self.diffusion,
                    brownian_motion,
                ),
            )

        else:
            # Real deterministic SME flow.
            terms = ODETerm(
                self.drift
            )

        return diffeqsolve(
            terms,
            Euler(),
            t0,
            t,
            dt0,
            y0=init_x,
            saveat=SaveAt(
                ts=ts
            ),
            max_steps=None,
        )

class MomentumSecondOrderFlow(MomentumFirstOrderFlow):
    def __init__(
        self,
        stepsize,
        momentum,
        f,
        cov_mat_func,
        brownian_tol=1e-6,
        stochastic=True,
    ):
        super().__init__(
            stepsize,
            momentum,
            f,
            cov_mat_func,
            brownian_tol=brownian_tol,
            stochastic=stochastic,
        )

        self.hessian = jax.hessian(f)

    def drift(self, t, x, args):
        eta = self.stepsize
        mu = self.momentum

        grad = self.grad(x)
        hessian = self.hessian(x)

        correction = (
            eta
            * (1.0 + mu)
            / (2.0 * (1.0 - mu)**3)
        )

        return (
            -grad / (1.0 - mu)
            - correction * (hessian @ grad)
        )

class LiConstantMomentumSME1:
    r"""
    First-order constant-momentum phase-space SME from
    Section 4.2.1 of Li, Tai and E.

    Discrete SGDM convention:

        v_{k+1}
            = mu * v_k
              - eta * stochastic_gradient(x_k)

        x_{k+1}
            = x_k + v_{k+1}

    Continuous model:

        dV
            = [-(1-mu)/eta * V - grad f(X)] dt
              + sqrt(eta) * Sigma(X)^(1/2) dW

        dX
            = V / eta dt

    The state is ordered as

        y = concatenate([V, X]).

    One SGDM step corresponds to continuous time eta:

        t_k = k * eta.
    """

    def __init__(
        self,
        stepsize,
        momentum,
        f,
        cov_mat_func,
        brownian_tol=1e-6,
        stochastic=True,
    ):
        if stepsize <= 0.0:
            raise ValueError(
                "stepsize must be positive."
            )

        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "momentum must satisfy 0 <= momentum < 1."
            )

        if brownian_tol <= 0.0:
            raise ValueError(
                "brownian_tol must be positive."
            )

        self.stepsize = float(
            stepsize
        )

        self.momentum = float(
            momentum
        )

        self.f = f

        self.grad = jax.grad(
            f
        )

        self.cov_mat_func = (
            cov_mat_func
        )

        self.brownian_tol = (
            brownian_tol
        )

        self.stochastic = bool(
            stochastic
        )

    @staticmethod
    def position_from_state(states):
        """
        Extract X from states whose last dimension is [V, X].
        """
        parameter_dimension = (
            states.shape[-1] // 2
        )

        return states[
            ...,
            parameter_dimension:,
        ]

    def drift(
        self,
        t,
        y,
        args,
    ):
        del t, args

        parameter_dimension = (
            y.shape[0] // 2
        )

        velocity = y[
            :parameter_dimension
        ]

        position = y[
            parameter_dimension:
        ]

        eta = jnp.asarray(
            self.stepsize,
            dtype=y.dtype,
        )

        mu = jnp.asarray(
            self.momentum,
            dtype=y.dtype,
        )

        gradient = self.grad(
            position
        )

        velocity_drift = (
            -(
                1.0 - mu
            )
            * velocity
            / eta
            - gradient
        )

        position_drift = (
            velocity / eta
        )

        return jnp.concatenate(
            [
                velocity_drift,
                position_drift,
            ],
            axis=0,
        )

    def diffusion(
        self,
        t,
        y,
        args,
    ):
        del t, args

        parameter_dimension = (
            y.shape[0] // 2
        )

        position = y[
            parameter_dimension:
        ]

        eta = jnp.asarray(
            self.stepsize,
            dtype=y.dtype,
        )

        covariance = (
            self.cov_mat_func(
                position
            )
        )

        # Suppress floating-point asymmetry.
        covariance = 0.5 * (
            covariance
            + covariance.T
        )

        covariance_root = (
            symmetric_psd_sqrt(
                covariance
            )
        )

        velocity_diffusion = (
            jnp.sqrt(eta)
            * covariance_root
        )

        # There is no direct Brownian forcing in X.
        position_diffusion = (
            jnp.zeros_like(
                velocity_diffusion
            )
        )

        # Shape: (2*d, d).
        return jnp.concatenate(
            [
                velocity_diffusion,
                position_diffusion,
            ],
            axis=0,
        )

    def solve(
        self,
        init_x,
        t,
        dt0,
        ts,
        key,
        t0=0,
    ):
        # The code initializes x_{-1} = x_0, so the initial
        # momentum increment is zero.
        init_velocity = jnp.zeros_like(
            init_x
        )

        init_state = jnp.concatenate(
            [
                init_velocity,
                init_x,
            ],
            axis=0,
        )

        if self.stochastic:
            # Only V is driven directly by Brownian motion,
            # so the Brownian dimension is d rather than 2*d.
            brownian_motion = (
                VirtualBrownianTree(
                    t0,
                    t,
                    tol=self.brownian_tol,
                    shape=init_x.shape,
                    key=key,
                )
            )

            terms = MultiTerm(
                ODETerm(
                    self.drift
                ),
                ControlTerm(
                    self.diffusion,
                    brownian_motion,
                ),
            )

        else:
            terms = ODETerm(
                self.drift
            )

        return diffeqsolve(
            terms,
            Euler(),
            t0,
            t,
            dt0,
            y0=init_state,
            saveat=SaveAt(
                ts=ts
            ),
            max_steps=None,
        )

class MomentumDALOptimizer:
    r"""
    Deterministic full-batch GD with momentum and a
    momentum-adjusted Drift Adaptive Learning Rate.

    Update:
        x_{k+1}
          = x_k
            - eta_k * grad f(x_k)
            + mu * (x_k - x_{k-1}).

    Adaptive learning rate:
        eta_k
          = 2 * target_ratio * (1-mu)^2
            / ((1+mu) * ||H(x_k) g_hat(x_k)||).

    Intrinsic momentum drift:
        d_k
          = (1+mu) / (2*(1-mu)^3)
            * eta_k^2
            * ||H(x_k) grad f(x_k)||.

    The returned stepsize is clipped for numerical safety. Therefore,
    when clipping is active, the target drift-to-signal ratio is not
    necessarily attained exactly.
    """

    def __init__(
        self,
        momentum,
        f,
        target_ratio=1.0,
        min_stepsize=1e-8,
        max_stepsize=0.5,
        curvature_epsilon=1e-12,
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "momentum must satisfy 0 <= momentum < 1."
            )

        if target_ratio <= 0.0:
            raise ValueError(
                "target_ratio must be positive."
            )

        if min_stepsize <= 0.0:
            raise ValueError(
                "min_stepsize must be positive."
            )

        if max_stepsize <= 0.0:
            raise ValueError(
                "max_stepsize must be positive."
            )

        if max_stepsize < min_stepsize:
            raise ValueError(
                "max_stepsize must be at least min_stepsize."
            )

        if curvature_epsilon <= 0.0:
            raise ValueError(
                "curvature_epsilon must be positive."
            )

        self.momentum = float(momentum)
        self.f = f
        self.grad = jax.grad(f)

        self.target_ratio = float(target_ratio)
        self.min_stepsize = float(min_stepsize)
        self.max_stepsize = float(max_stepsize)
        self.curvature_epsilon = float(
            curvature_epsilon
        )

    def geometry(self, x):
        r"""
        Compute the local quantities required by M-DAL.

        Only one Hessian-vector product is used:

            H g_hat = JVP(grad f, g_hat).

        Since
            H g = ||g|| H g_hat,
        we obtain
            ||H g|| = ||g|| ||H g_hat||
        without a second HVP.
        """
        gradient = self.grad(x)

        gradient_norm = jnp.linalg.norm(
            gradient
        )

        epsilon = jnp.asarray(
            self.curvature_epsilon,
            dtype=x.dtype,
        )

        safe_gradient_norm = jnp.maximum(
            gradient_norm,
            epsilon,
        )

        unit_gradient = (
            gradient
            / safe_gradient_norm
        )

        # Forward-over-reverse Hessian-vector product.
        hessian_unit_gradient = jax.jvp(
            self.grad,
            (x,),
            (unit_gradient,),
        )[1]

        directional_curvature = jnp.linalg.norm(
            hessian_unit_gradient
        )

        # Exact when ||g|| > 0. At a stationary point both sides
        # are interpreted as zero.
        hessian_gradient_norm = (
            gradient_norm
            * directional_curvature
        )

        return (
            gradient,
            gradient_norm,
            directional_curvature,
            hessian_gradient_norm,
        )

    def adaptive_stepsize(
        self,
        directional_curvature,
        dtype,
    ):
        r"""
        Momentum-adjusted DAL:

            eta
              = 2*tau*(1-mu)^2
                / ((1+mu)*||H g_hat||).
        """
        mu = jnp.asarray(
            self.momentum,
            dtype=dtype,
        )

        target_ratio = jnp.asarray(
            self.target_ratio,
            dtype=dtype,
        )

        epsilon = jnp.asarray(
            self.curvature_epsilon,
            dtype=dtype,
        )

        safe_curvature = jnp.maximum(
            directional_curvature,
            epsilon,
        )

        raw_stepsize = (
            2.0
            * target_ratio
            * (1.0 - mu) ** 2
            / (
                (1.0 + mu)
                * safe_curvature
            )
        )

        used_stepsize = jnp.clip(
            raw_stepsize,
            jnp.asarray(
                self.min_stepsize,
                dtype=dtype,
            ),
            jnp.asarray(
                self.max_stepsize,
                dtype=dtype,
            ),
        )

        return (
            used_stepsize,
            raw_stepsize,
        )

    def _take_step(
        self,
        x,
        x_prev,
        stepsize,
        raw_stepsize,
        gradient,
        gradient_norm,
        directional_curvature,
        hessian_gradient_norm,
    ):
        mu = jnp.asarray(
            self.momentum,
            dtype=x.dtype,
        )

        stepsize = jnp.asarray(
            stepsize,
            dtype=x.dtype,
        )

        momentum_increment = (
            mu * (x - x_prev)
        )

        gradient_increment = (
            -stepsize * gradient
        )

        total_increment = (
            momentum_increment
            + gradient_increment
        )

        x_next = (
            x + total_increment
        )

        drift_prefactor = (
            (1.0 + mu)
            / (
                2.0
                * (1.0 - mu) ** 3
            )
        )

        intrinsic_drift = (
            drift_prefactor
            * stepsize**2
            * hessian_gradient_norm
        )

        update_norm = jnp.linalg.norm(
            total_increment
        )

        diagnostics = (
            stepsize,
            raw_stepsize,
            intrinsic_drift,
            update_norm,
            gradient_norm,
            hessian_gradient_norm,
            directional_curvature,
        )

        return (
            x_next,
            diagnostics,
        )

    def adaptive_step(
        self,
        x,
        x_prev,
    ):
        (
            gradient,
            gradient_norm,
            directional_curvature,
            hessian_gradient_norm,
        ) = self.geometry(x)

        (
            stepsize,
            raw_stepsize,
        ) = self.adaptive_stepsize(
            directional_curvature,
            x.dtype,
        )

        return self._take_step(
            x=x,
            x_prev=x_prev,
            stepsize=stepsize,
            raw_stepsize=raw_stepsize,
            gradient=gradient,
            gradient_norm=gradient_norm,
            directional_curvature=(
                directional_curvature
            ),
            hessian_gradient_norm=(
                hessian_gradient_norm
            ),
        )

    def fixed_step(
        self,
        x,
        x_prev,
        stepsize,
    ):
        r"""
        Fixed-learning-rate GDM step, with the same diagnostic
        quantities as adaptive M-DAL.
        """
        (
            gradient,
            gradient_norm,
            directional_curvature,
            hessian_gradient_norm,
        ) = self.geometry(x)

        stepsize = jnp.asarray(
            stepsize,
            dtype=x.dtype,
        )

        return self._take_step(
            x=x,
            x_prev=x_prev,
            stepsize=stepsize,
            raw_stepsize=stepsize,
            gradient=gradient,
            gradient_norm=gradient_norm,
            directional_curvature=(
                directional_curvature
            ),
            hessian_gradient_norm=(
                hessian_gradient_norm
            ),
        )