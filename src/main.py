#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
import scipy.fft as sfft
import matplotlib.pyplot as plt


# Initial conditions
class wave_eq:

    def __init__(self, N: int = 64, c: float = 1.0):
        # Parameters
        L = 2 * np.pi # Domain size
        self.dx = L / N 

        # Defining x-domain
        self.x = np.linspace(0, L, N)
        self.c = c # Wave speed
        self.dt = 0.001
        self.T = 10.0 # Total duration
        self.steps = int(self.T / self.dt)

        # Wavenumber
        self.k = sfft.fftfreq(N, d = self.dx) * 2 * np.pi

    def solve(self) -> np.ndarray:
        """
        Solving the 1D wave equation using Fourier-Galerkin method
        """
        # Initial conditions
        u0 = np.sin(self.x)
        v0 = np.zeros_like(self.x)

        # Fourier transform of initial conditions
        u_hat = sfft.fft(u0)
        v_hat = sfft.fft(v0)

        # u(x, t) at time step n - 1
        u_hat_prev = u_hat
        u_hat_cur = u_hat + self.dt * v_hat - (self.c * self.dt * self.k)**2 / 2 * u_hat 

        # Updating at each time step
        for _ in range(self.steps):
            u_hat_next = 2 * u_hat_cur - u_hat_prev - (self.c * self.dt * self.k)**2 * u_hat_cur
            u_hat_prev = u_hat_cur
            u_hat_cur = u_hat_next
        
        # Using the inverse Fourier transform
        u = np.real(sfft.ifft(u_hat_cur))

        return u



if __name__ == '__main__':
    u = wave_eq().solve()

    plt.figure()
    plt.plot(range(u.shape[0]), u)
    plt.show()




