"""Vasicek short-rate model: exact simulation and closed-form bond prices.

    dr(t) = a * (b - r(t)) dt + sigma * dW(t)

a     : speed of mean reversion
b     : long-term mean level of the short rate
sigma : volatility of the short rate
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Vasicek:
    a: float
    b: float
    sigma: float
    r0: float

    def simulate(self, times: np.ndarray, n_paths: int, seed: int = 42) -> np.ndarray:
        """Simulate short-rate paths on the grid `times` (times[0] must be 0).

        Uses the exact Gaussian transition of the Ornstein-Uhlenbeck process,
        so there is no discretisation bias whatever the time step.
        Antithetic variates halve the Monte Carlo noise for free.
        Returns an array of shape (n_paths, len(times)).
        """
        rng = np.random.default_rng(seed)
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, len(times) - 1))
        z = np.concatenate([z, -z])  # antithetic pairs
        return self.simulate_from_normals(times, z)[:n_paths]

    def simulate_from_normals(self, times: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Exact OU recursion driven by a given matrix of N(0,1) draws.

        z has shape (n_paths, len(times) - 1). Passing the same z to two
        models gives common random numbers (needed for stable bump Greeks).
        """
        rates = np.empty((z.shape[0], len(times)))
        rates[:, 0] = self.r0
        for k in range(1, len(times)):
            dt = times[k] - times[k - 1]
            decay = np.exp(-self.a * dt)
            std = self.sigma * np.sqrt((1.0 - decay**2) / (2.0 * self.a))
            rates[:, k] = rates[:, k - 1] * decay + self.b * (1.0 - decay) + std * z[:, k - 1]
        return rates

    def first_step_moments(self, dt: float) -> tuple[float, float]:
        """Mean and std of r(dt) given r(0) = r0 (used by the likelihood ratio)."""
        decay = np.exp(-self.a * dt)
        mean = self.r0 * decay + self.b * (1.0 - decay)
        std = self.sigma * np.sqrt((1.0 - decay**2) / (2.0 * self.a))
        return mean, std

    def B(self, tau):
        return (1.0 - np.exp(-self.a * np.asarray(tau))) / self.a

    def zcb(self, t: float, T: float, r) -> np.ndarray:
        """Price at time t of a zero-coupon bond paying 1 at T, given r(t) = r.

        Closed form: P(t,T) = A(tau) * exp(-B(tau) * r), tau = T - t.
        """
        tau = T - t
        a, b, s = self.a, self.b, self.sigma
        B = (1.0 - np.exp(-a * tau)) / a
        log_A = (B - tau) * (a * a * b - 0.5 * s * s) / (a * a) - s * s * B * B / (4.0 * a)
        return np.exp(log_A - B * np.asarray(r))
