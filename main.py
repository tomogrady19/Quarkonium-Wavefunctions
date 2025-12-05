from matplotlib import pyplot as plt
from meson import Meson


def plot_wf(meson, n, l, E_range):
    x = meson.rs
    y = meson.solve(E_range, n, l)
    plt.plot(x, y, label='n=%i, l=%i' % (meson.n, meson.l))


def plot_charmonium():
    meson = Meson('charm', 'charm')
    meson.solve_beta(0.19, 0.2)
    E_ranges = [[0.3, 0.4], [0.8, 0.9], [1, 1.1], [1.5, 1.6]]
    nl_list = [[1, 0], [1, 1], [2, 0], [3, 0]]

    for i, nl in enumerate(nl_list):
        plot_wf(meson, *nl, E_ranges[i])

    plt.ylabel(r'$\mathregular{u_{nl}}(r)$')
    plt.xlabel(r'$r$ / $\mathregular{GeV^{-1}}$')
    plt.title('Charmonium Radial Wavefunction')
    plt.legend()
    plt.grid()
    plt.show()


def plot_bottomonium():
    meson = Meson('bottom', 'bottom')
    meson.solve_beta(0.2, 0.3)
    E_ranges = [[0.12, 0.13], [0.5, 0.6], [0.6, 0.7], [1, 1.1]]
    nl_list = [[1, 0], [1, 1], [2, 0], [3, 0]]

    for i, nl in enumerate(nl_list):
        plot_wf(meson, *nl, E_ranges[i])

    plt.ylabel(r'$\mathregular{u_{nl}}(r)$')
    plt.xlabel(r'$r$ / $\mathregular{GeV^{-1}}$')
    plt.title('Bottomonium Radial Wavefunction')
    plt.legend()
    plt.grid()
    plt.show()


def plot_bc_meson():
    meson = Meson('bottom', 'charm')
    meson.solve_beta(0.1, 0.2)
    E_ranges = [[0.2, 0.3], [0.6, 0.7], [1.1, 1.2]]
    nl_list = [[1, 0], [1, 1], [2, 0]]

    for i, nl in enumerate(nl_list):
        plot_wf(meson, *nl, E_ranges[i])

    plt.ylabel(r'$\mathregular{u_{nl}}(r)$')
    plt.xlabel(r'$r$ / $\mathregular{GeV^{-1}}$')
    plt.title('Bc Meson Radial Wavefunction')
    plt.legend()
    plt.grid()
    plt.show()

plot_charmonium()
plot_bottomonium()
plot_bc_meson()

