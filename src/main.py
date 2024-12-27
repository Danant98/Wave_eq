#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
import scipy.fft as sfft
import matplotlib.pyplot as plt


# Initial conditions
class wave_eq:

    def __init__(self, 
                 Nx: int = 256, 
                 Lx: float = 10.0,
                 T: float = 1.5,
                 dt: float = 1e-2,
                 cx: float = 1.0,
                 dims: str = '1D'
                 ) -> None:
        # Storing dims
        self.dims = dims

        if dims == '1D':
            # Defining spatial domain
            self.x = np.linspace(0, Lx, Nx)

            # Defining time domain
            self.dt = dt
            self.t = np.arange(0, T, dt)

            # Setting size of domain
            self.Lx = Lx
            self.cx = cx

            # Wavenumbers
            self.kx = sfft.fftfreq(Nx, d = Lx / Nx) * 2 * np.pi

            # Computing Laplace operator in Fourier space
            self.Lhat = 2 - ((self.dt**2) * (cx**2) * (self.kx**2))

            # Initializing u(x, t)
            self.u = np.zeros((len(self.x), len(self.t)))


    def u0(self, l: float, amp: float = 1.0) -> None:
        """
        Initial condition; Gaussian distribution
        """
        if self.dims == '1D':
            return amp * np.exp(-(self.x - (self.Lx / 2))**2 ) / (2 * l**2)

    def solve(self) -> np.ndarray:
        """
        Solving wave equation using Fourier-Galerkin in space and central difference in time
        Computing first time step using Taylor expansion
        """
        # Initial conditions; u(x, 0) and u'(x, 0)
        self.u[:, 0] = self.u0(0.03 * self.Lx)
        self.v0 = np.zeros(self.x.shape[0])

        # Fourier transfrom of u(x, 0)
        uhat = sfft.fft(self.u[:, 0])
        vhat = sfft.fft(self.v0)
        
        # Computing first time step using forward difference in time
        uhat1 = uhat + self.dt * vhat - 0.5 * (self.cx**2) * (self.dt**2) * (self.kx**2) * uhat

        # Looping over time
        for n in range(1, len(self.t) - 1):
            uhat_new = self.Lhat * uhat1 - uhat
            self.u[:, n + 1] = np.real(sfft.ifft(uhat_new))
            uhat = uhat1
            uhat1 = uhat_new

        return self.u

    def animate(self) -> None:
        """
        Animate solution
        """
        if self.dims == '1D':
            plt.figure()
            for i in range(self.u.shape[1]):
                plt.clf()
                plt.plot(self.x, self.u[:, i])
                plt.xlabel(r'$x$')
                plt.ylabel(r'$u(x, t)$')
                plt.pause(1E-7)


if __name__ == '__main__':
    wq = wave_eq()
    wq.solve()
    wq.animate()