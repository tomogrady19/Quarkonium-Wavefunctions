"""Meson Class"""
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import simps

class Meson:

    mass_dict = {'charm': 1.34, 'bottom': 4.7}  # quark masses (GeV)
    u0 = [0, 1]  # initial condition
    ac = 0  # convergence threshold

    def __init__(self, quark1='charm', quark2 ='charm'):
        if quark1 == quark2 == 'charm':
            state_mass = 3.068  # experimental value of (n, l) = (1, 0) state (GeV)
            self.a = 0.4
            self.rs = np.arange(0.0001, 15, 0.01)  # range (1/GeV)
        elif quark1 == quark2 == 'bottom':
            state_mass = 9.3987  # experimental value of (n, l) = (1, 0) state (GeV)
            self.a = 0.28
            self.rs = np.arange(0.0001, 9, 0.01)  # range (1/GeV)
        elif all(q in ['charm', 'bottom'] for q in [quark1, quark2]):
            state_mass = 6.276  # experimental value of (n, l) = (1, 0) state (GeV)
            self.a = 0.34
            self.rs = np.arange(0.0001, 15, 0.01)  # range (1/GeV)

        self.m1 = self.mass_dict[quark1]  # mass of quark1 (GeV)
        self.m2 = self.mass_dict[quark2]  # mass of quark1 (GeV)
        self.mu = 1/(1/self.m1 + 1/self.m2)  # reduced mass (GeV)
        self.E = state_mass - self.m1 - self.m2  # Energy of state (GeV)

        self.n = 1  # principal quantum number
        self.l = 0  # angular momentum quantum number

    def radial_equation(self, u0, r):
        """second order differential equation for radial wavefunction"""
        [u, du_dr] = u0
        if r > 0:
            d2u_dr2 = (self.l * (self.l + 1) / r ** 2 - 2 * self.mu * (self.E - self.b * r + 4 * self.a / 3 / r)) * u
        else:
            d2u_dr2 = 0  # radial wavefunction only makes sense for positive values of r
        return np.array([du_dr, d2u_dr2])

    def wf_b(self, b):
        # solve for the radial wavefunction given a certain value of beta (string tension) (GeV^2)
        self.b = b
        return odeint(self.radial_equation, self.u0, self.rs)[:, 0]  # extract radial wavefunction

    def wf_E(self, E):
        # solve for the radial wavefunction given a certain energy
        self.E = E
        return odeint(self.radial_equation, self.u0, self.rs)[:, 0]  # extract radial wavefunction

    def TN_b(self, b):  # turns and nodes for different beta (string tension) values
        x_values = self.wf_b(b)  # selects all x values of radial wavefunctions for different beta values
        return [turns(x_values), nodes(x_values)]

    def TN_E(self, E):  # turns and nodes for different energy values
        x_values = self.wf_E(E)
        return [turns(x_values), nodes(x_values)]

    def solve_beta(self, A, C):
        """Find the value of beta (b) (string tension) that gives a normalizable wavefunction by matching nodes and turning points."""
        B = 0.5 * (A + C)  # A, B and C are our first three guesses

        # initial evaluations
        A_tn = self.TN_b(A)
        B_tn = self.TN_b(B)
        C_tn = self.TN_b(C)

        while A_tn != B_tn or B_tn != C_tn:
            if abs(B - A) <= self.ac or abs(C - B) <= self.ac:  # ac is not None and
                break
            if A_tn != B_tn:  # if true beta value is between A and B, search in this range
                C = B
                C_tn = B_tn
            elif C_tn != B_tn:  # if true beta value is between B and C, search in this range
                A = B
                A_tn = B_tn
            B = 0.5 * (A + C)
            B_tn = self.TN_b(B)
        self.b = B

    def solve_energy(self, A, C):
        """Find the value of energy (E) that gives a normalizable wavefunction by matching nodes and turning points."""
        B = 0.5 * (A + C)

        # initial evaluations
        A_tn = self.TN_E(A)
        B_tn = self.TN_E(B)
        C_tn = self.TN_E(C)

        while A_tn != B_tn or B_tn != C_tn:
            if abs(B - A) <= self.ac or abs(C - B) <= self.ac:  # ac is not None and
                break
            if A_tn != B_tn:  # if true beta value is between A and B, search in this range
                C = B
                C_tn = B_tn
            elif C_tn != B_tn:  # if true beta value is between B and C, search in this range
                A = B
                A_tn = B_tn
            B = 0.5 * (A + C)
            B_tn = self.TN_E(B)
        return B

    def solve(self, E_range, n, l):

        self.n = n
        self.l = l

        E = self.solve_energy(*E_range)
        ys = self.wf_E(E)
        prob_density = np.square(ys)

        ys /= simps(prob_density, self.rs) ** 0.5  # normalise
        return ys


def turns(X):  # calculating the number of turning points
    N = 0
    for i in range(len(X) - 1):
        if ((X[i - 1] < X[i] and X[i + 1] < X[i])  # local maximum
                or X[i - 1] > X[i] and X[i + 1] > X[i]):  # local minimum
            N += 1
    return N


def nodes(X):  # calculating the number of nodes (interceptions of the x axis)
    M = 0
    for i in range(len(X) - 1):
        if X[i] * X[i + 1] <= 0:
            M += 1
    return M