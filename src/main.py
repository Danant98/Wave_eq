#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
import scipy.fft as sfft
import matplotlib.pyplot as plt


# Initial conditions
class wave_eq:

    def __init__(self, 
                 Nx: int = 64, 
                 Ny: int = 64,
                 Lx: float = 15.0,
                 Ly: float = 15.0,
                 T: float = 1.5,
                 dt: float = 1e-2,
                 c: float = 1.0,
                 dims: str = '1D'
                 ) -> None:
        # Storing dims
        self.dims = dims

        if dims == '1D':
            assert c * dt / (Lx / Nx) <= 1, 'CLF condition violated'
            
            # Defining spatial domain
            self.x = np.linspace(0, Lx, Nx)

            # Defining time domain
            self.dt = dt
            self.t = np.arange(0, T, dt)

            # Setting size of domain and wave speed
            self.Lx = Lx
            self.c = c

            # Wavenumber
            self.kx = sfft.fftfreq(Nx, d = Lx / Nx) * 2 * np.pi

            # Computing Laplace operator in Fourier space
            self.Lhat = 2 - ((self.dt**2) * (c**2) * (self.kx**2))

            # Initializing u(x, t)
            self.u = np.zeros((len(self.x), len(self.t)))

        elif dims == '2D':
            assert c * dt <= min(Lx / Nx, Ly / Ny), 'CLF condition violated'

            # Defining spatial domain
            self.x = np.linspace(0, Lx, Nx)
            self.y = np.linspace(0, Ly, Ny)

            # Defining time domain
            self.t = np.arange(0, T, dt)
            self.dt = dt

            # Setting size of domain and wave speed 
            self.Lx = Lx
            self.Ly = Ly
            self.c = c

            # Wavenumbers
            kx = np.fft.fftfreq(Nx, d = Lx / Nx) * 2 * np.pi
            ky = np.fft.fftfreq(Ny, d = Ly / Ny) * 2 * np.pi
            self.kx, self.ky = np.meshgrid(kx, ky)
            self.k_squared = self.kx**2 + self.ky**2

            # Computing Laplace operator in Fourier space
            self.Lhat = 2 - ((self.dt**2) * (c**2) * self.k_squared)

            # Initializing u(x, y, t)
            self.u = np.zeros((Nx, Ny, len(self.t)))

    def u0(self, l: float, amp: float = 1.0) -> None:
        """
        Initial condition; Gaussian distribution
        """
        if self.dims == '1D':
            return amp * np.exp(-((self.x - self.Lx / 2)**2) / (2 * l**2))
        elif self.dims == '2D':
            X, Y = np.meshgrid(self.x, self.y)
            return amp * np.exp(-( (X - (self.Lx / 2))**2 + (Y - (self.Ly / 2))**2 ) / (2 * l**2) )


    def solve(self) -> np.ndarray:
        """
        Solving wave equation using Fourier-Galerkin in space and central difference in time
        Computing first time step using Taylor expansion
        """
        if self.dims == '1D':
            # Initial conditions; u(x, 0) and u'(x, 0)
            self.u[:, 0] = self.u0(0.03 * self.Lx)
            self.v0 = np.zeros(self.x.shape[0])

            # Fourier transfrom of u(x, 0)
            uhat = sfft.fft(self.u[:, 0])
            vhat = sfft.fft(self.v0)
            
            # Computing first time step using forward difference in time
            uhat1 = uhat + self.dt * vhat - 0.5 * (self.c**2) * (self.dt**2) * (self.kx**2) * uhat

            # Looping over time
            for n in range(1, len(self.t) - 1):
                uhat_new = self.Lhat * uhat1 - uhat
                self.u[:, n + 1] = np.real(sfft.ifft(uhat_new))
                uhat = uhat1
                uhat1 = uhat_new

        elif self.dims == '2D':
            # Initial conditions
            self.u[:, :, 0] = self.u0(0.03 * self.Lx)
            self.v0 = np.zeros_like(self.u[:, :, 0])

            # Fourier transform of initial conditions
            uhat0 = sfft.fft2(self.u[:, :, 0])
            vhat0 = sfft.fft2(self.v0)

            # Computing the first time step using Taylor expansion
            uhat = uhat0 + self.dt * vhat0 - 0.5 * (self.c**2) * (self.dt**2) * (self.k_squared) * uhat0
            self.u[:, :, 1] = np.real(sfft.ifft2(uhat))

            # Looping over time
            for n in range(1, len(self.t) - 1):
                uhat_new = self.Lhat * uhat - uhat0
                self.u[:, :, n + 1] = np.real(sfft.ifft2(uhat_new))
                uhat0 = uhat
                uhat = uhat_new

        return self.u

    def animate(self) -> None:
        """
        Animate solution
        """
        if self.dims == '1D':
            plt.figure()
            vmin, vmax = np.min(self.u), np.max(self.u)
            for i in range(self.u.shape[1]):
                plt.clf()
                plt.plot(self.x, self.u[:, i])
                plt.xlabel(r'$x$')
                plt.ylabel(r'$u(x, t)$')
                plt.ylim([vmin, vmax])
                plt.pause(1E-7)
        elif self.dims == '2D':
            plt.figure()
            vmin, vmax = np.min(self.u), np.max(self.u)
            for i in range(len(self.t)):
                plt.clf()
                mappable = plt.contourf(self.x, self.y, self.u[:, :, i], levels = 20, vmin = vmin, vmax = vmax)
                plt.colorbar(mappable, label = r'$u(x, y)$')
                plt.xlabel(r'$x$')
                plt.ylabel(r'$y$')
                plt.title(f'Time: {self.t[i]:.3f}')
                plt.pause(1e-7)            
        plt.show()


if __name__ == '__main__':
    wq = wave_eq(dims = '2D')
    wq.solve()
    wq.animate()