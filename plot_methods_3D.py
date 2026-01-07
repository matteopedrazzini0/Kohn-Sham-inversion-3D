import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 14,        # grandezza testo di default
    "axes.titlesize": 16,   # titoli dei subplot
    "axes.labelsize": 16,   # label degli assi
    "xtick.labelsize": 14,  # ticks x
    "ytick.labelsize": 14,  # ticks y
    "legend.fontsize": 14,  # legenda
    "figure.titlesize": 20  # titolo figura
})


def plot_orbital_3D(phi_th, phi_calc, grid, filename):
    """
    Plot 3D orbitals: target, calculated, and difference.
    Each row=rho_target, rho_calc, diff
    Each column=slice along x, y, z axes (central slice)
    """
    x, y, z=grid
    nx, ny, nz=len(x), len(y), len(z)

    ix, iy, iz=nx//2, ny//2, nz//2

    diff=phi_th-phi_calc

    fig, axes=plt.subplots(3, 3, figsize=(15, 12))

    # -----------------------
    # RIGA 0 = rho_target
    im00=axes[0,0].imshow(phi_th[ix,:,:].T, origin='lower', extent=[y[0], y[-1], z[0], z[-1]], aspect='auto', cmap='viridis')
    axes[0,0].set_title(f"Target, cut x={x[ix]} fm")
    fig.colorbar(im00, ax=axes[0, 0])
    im01=axes[0,1].imshow(phi_th[:,iy,:].T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], aspect='auto', cmap='viridis')
    axes[0,1].set_title(f"Target, cut y={y[iy]} fm")
    fig.colorbar(im01, ax=axes[0, 1])
    im02=axes[0,2].imshow(phi_th[:,:,iz].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', cmap='viridis')
    axes[0,2].set_title(f"Target, cut z={z[iz]} fm")
    fig.colorbar(im02, ax=axes[0, 2])

    # -----------------------
    # RIGA 1 = rho_calc
    im10=axes[1,0].imshow(phi_calc[ix,:,:].T, origin='lower', extent=[y[0], y[-1], z[0], z[-1]], aspect='auto', cmap='viridis')
    axes[1,0].set_title(f"Calculated, cut x={x[ix]} fm")
    fig.colorbar(im10, ax=axes[1, 0])
    im11=axes[1,1].imshow(phi_calc[:,iy,:].T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], aspect='auto', cmap='viridis')
    axes[1,1].set_title(f"Calculated, slice y={y[iy]} fm")
    fig.colorbar(im11, ax=axes[1, 1])
    im12=axes[1,2].imshow(phi_calc[:,:,iz].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', cmap='viridis')
    axes[1,2].set_title(f"Calculated, cut z={z[iz]} fm")
    fig.colorbar(im12, ax=axes[1, 2])

    # -----------------------
    # RIGA 2 = differenza
    im20=axes[2,0].imshow(diff[ix,:,:].T, origin='lower', extent=[y[0], y[-1], z[0], z[-1]], aspect='auto', cmap='coolwarm')
    axes[2,0].set_title(f"Difference, x={x[ix]} fm")
    fig.colorbar(im20, ax=axes[2, 0])
    im21=axes[2,1].imshow(diff[:,iy,:].T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], aspect='auto', cmap='coolwarm')
    axes[2,1].set_title(f"Difference, y={y[iy]} fm")
    fig.colorbar(im21, ax=axes[2, 1])
    im22=axes[2,2].imshow(diff[:,:,iz].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', cmap='coolwarm')
    axes[2,2].set_title(f"Difference, z={z[iz]} fm")
    fig.colorbar(im22, ax=axes[2, 2])

    labels = [
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]"),
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]"),
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]")
    ]

    for ax, (xl, yl) in zip(axes.flat, labels):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)




def plot_density_3D(rho_target, rho_calc, grid, filename="/Users/matteopedrazzini/Desktop/SIM/3D_CODE/test_3D/3D_density.png"):
    """
    Plot 3D densities: target, calculated, and difference.
    Each row=rho_target, rho_calc, diff
    Each column=slice along x, y, z axes (central slice)
    """
    x, y, z=grid
    nx, ny, nz=len(x), len(y), len(z)

    ix, iy, iz=nx//2, ny//2, nz//2

    diff=rho_target-rho_calc

    fig, axes=plt.subplots(3, 3, figsize=(15, 12))

    # -----------------------
    # RIGA 0 = rho_target
    im00=axes[0,0].imshow(rho_target[ix,:,:].T, origin='lower', extent=[y[0], y[-1], z[0], z[-1]], aspect='auto', cmap='viridis')
    axes[0,0].set_title(f"ρ Target [fm$^{{-3}}$], cut x={x[ix]}")
    fig.colorbar(im00, ax=axes[0, 0])
    im01=axes[0,1].imshow(rho_target[:,iy,:].T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], aspect='auto', cmap='viridis')
    axes[0,1].set_title(f"ρ Target [fm$^{{-3}}$], cut y={y[iy]}")
    fig.colorbar(im01, ax=axes[0, 1])
    im02=axes[0,2].imshow(rho_target[:,:,iz].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', cmap='viridis')
    axes[0,2].set_title(f"ρ Target [fm$^{{-3}}$], cut z={z[iz]}")
    fig.colorbar(im02, ax=axes[0, 2])

    # -----------------------
    # RIGA 1 = rho_calc
    im10=axes[1,0].imshow(rho_calc[ix,:,:].T, origin='lower', extent=[y[0], y[-1], z[0], z[-1]], aspect='auto', cmap='viridis')
    axes[1,0].set_title(f"ρ Calculated [fm$^{{-3}}$], cut x={x[ix]}")
    fig.colorbar(im10, ax=axes[1, 0])
    im11=axes[1,1].imshow(rho_calc[:,iy,:].T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], aspect='auto', cmap='viridis')
    axes[1,1].set_title(f"ρ Calculated [fm$^{{-3}}$], cut y={y[iy]}")
    fig.colorbar(im11, ax=axes[1, 1])
    im12=axes[1,2].imshow(rho_calc[:,:,iz].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', cmap='viridis')
    axes[1,2].set_title(f"ρ Calculated [fm$^{{-3}}$], cut z={z[iz]}")
    fig.colorbar(im12, ax=axes[1, 2])

    # -----------------------
    # RIGA 2 = differenza
    im20=axes[2,0].imshow(diff[ix,:,:].T, origin='lower', extent=[y[0], y[-1], z[0], z[-1]], aspect='auto', cmap='coolwarm')
    axes[2,0].set_title(rf"$\Delta\rho$ [fm$^{{-3}}$], x={x[ix]}")
    fig.colorbar(im20, ax=axes[2, 0])
    im21=axes[2,1].imshow(diff[:,iy,:].T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], aspect='auto', cmap='coolwarm')
    axes[2,1].set_title(rf"$\Delta\rho$ [fm$^{{-3}}$], y={y[iy]}")
    fig.colorbar(im21, ax=axes[2, 1])
    im22=axes[2,2].imshow(diff[:,:,iz].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', cmap='coolwarm')
    axes[2,2].set_title(rf"$\Delta\rho$ [fm$^{{-3}}$], z={z[iz]}")
    fig.colorbar(im22, ax=axes[2, 2])

    labels = [
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]"),
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]"),
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]")
    ]

    for ax, (xl, yl) in zip(axes.flat, labels):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def plot_potential_3D_no_mod_grid(V_target, V_calc, grid, filename):
    """
    Plot 3D densities: target, calculated, and difference.
    Each row=rho_target, rho_calc, diff
    Each column=slice along x, y, z axes (central slice)
    """
    x, y, z=grid
    nx, ny, nz=len(x), len(y), len(z)

    ix, iy, iz=nx//2, ny//2, nz//2

    diff=V_calc-V_target

    fig, axes=plt.subplots(3, 3, figsize=(15, 12))

    # -----------------------
    # RIGA 0 = rho_target
    im00=axes[0,0].imshow(V_target[ix,:,:].T, origin='lower', extent=[y[0], y[-1], z[0], z[-1]], aspect='auto', cmap='plasma')
    axes[0,0].set_title(f"V Target [MeV], x={x[ix]} fm")
    fig.colorbar(im00, ax=axes[0, 0])
    im01=axes[0,1].imshow(V_target[:,iy,:].T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], aspect='auto', cmap='plasma')
    axes[0,1].set_title(f"V Target [MeV], y={y[iy]} fm")
    fig.colorbar(im01, ax=axes[0, 1])
    im02=axes[0,2].imshow(V_target[:,:,iz].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', cmap='plasma')
    axes[0,2].set_title(f"V Target [MeV], z={z[iz]} fm")
    fig.colorbar(im02, ax=axes[0, 2])

    # -----------------------
    # RIGA 1 = rho_calc
    im10=axes[1,0].imshow(V_calc[ix,:,:].T, origin='lower', extent=[y[0], y[-1], z[0], z[-1]], aspect='auto', cmap='plasma')
    axes[1,0].set_title(f"V Calculated [MeV],x={x[ix]}")
    fig.colorbar(im10, ax=axes[1, 0])
    im11=axes[1,1].imshow(V_calc[:,iy,:].T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], aspect='auto', cmap='plasma')
    axes[1,1].set_title(f"V Calculated [MeV],y={y[iy]}")
    fig.colorbar(im11, ax=axes[1, 1])
    im12=axes[1,2].imshow(V_calc[:,:,iz].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', cmap='plasma')
    axes[1,2].set_title(f"V Calculated [MeV],z={z[iz]}")
    fig.colorbar(im12, ax=axes[1, 2])

    # -----------------------
    # RIGA 2 = differenza
    im20=axes[2,0].imshow(diff[ix,:,:].T, origin='lower', extent=[y[0], y[-1], z[0], z[-1]], aspect='auto', cmap='coolwarm')
    axes[2,0].set_title(f"Difference [MeV],x={x[ix]} fm")
    fig.colorbar(im20, ax=axes[2, 0])
    im21=axes[2,1].imshow(diff[:,iy,:].T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], aspect='auto', cmap='coolwarm')
    axes[2,1].set_title(f"Difference [MeV],y={y[iy]} fm")
    fig.colorbar(im21, ax=axes[2, 1])
    im22=axes[2,2].imshow(diff[:,:,iz].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', cmap='coolwarm')
    axes[2,2].set_title(f"Difference [MeV],z={z[iz]} fm")
    fig.colorbar(im22, ax=axes[2, 2])

    labels = [
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]"),
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]"),
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]")
    ]

    for ax, (xl, yl) in zip(axes.flat, labels):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)



def plot_potential_3D(V_target, V_calc, grid, filename="/Users/matteopedrazzini/Desktop/SIM/3D_CODE/test_3D/3D_potential.png"):
    """
    Plot 3D potentials: target, calculated, difference.
    Each row=target, calculated, difference
    Each column=slice along x, y, z axes (central slice)
    """
    x, y, z=grid
    nx, ny, nz=len(x), len(y), len(z)

    ix, iy, iz=nx//2, ny//2, nz//2

    diff=V_calc-V_target

    ix_min=np.searchsorted(x, -2)
    ix_max=np.searchsorted(x, 2)

    iy_min=np.searchsorted(y, -2)
    iy_max=np.searchsorted(y, 2)

    iz_min=np.searchsorted(z, -2)
    iz_max=np.searchsorted(z, 2)

    fig, axes=plt.subplots(3, 3, figsize=(15, 12))

    # -----------------------
    # RIGA 0 = V_target
    im00=axes[0,0].imshow(V_target[ix, iy_min:iy_max, iz_min:iz_max].T, origin='lower', extent=[y[iy_min], y[iy_max-1], z[iz_min], z[iz_max-1]], aspect='auto', cmap='plasma')
    axes[0,0].set_title(f"V_Target [MeV], cut x={x[ix]} fm")
    fig.colorbar(im00, ax=axes[0, 0])
    im01=axes[0,1].imshow(V_target[ix_min:ix_max, iy, iz_min:iz_max].T, origin='lower', extent=[x[ix_min], x[ix_max-1], z[iz_min], z[iz_max-1]], aspect='auto', cmap='plasma')
    axes[0,1].set_title(f"V_Target [MeV], cut y={y[iy]} fm")
    fig.colorbar(im01, ax=axes[0, 1])
    im02=axes[0,2].imshow(V_target[ix_min:ix_max, iy_min:iy_max, iz].T, origin='lower', extent=[x[ix_min], x[ix_max-1], y[iy_min], y[iy_max-1]], aspect='auto', cmap='plasma')
    axes[0,2].set_title(f"V_Target [MeV], cut z={z[iz]} fm")
    fig.colorbar(im02, ax=axes[0, 2])

    # -----------------------
    # RIGA 1 = V_calc
    im10=axes[1,0].imshow(V_calc[ix, iy_min:iy_max, iz_min:iz_max].T, origin='lower', extent=[y[iy_min], y[iy_max-1], z[iz_min], z[iz_max-1]], aspect='auto', cmap='plasma')
    axes[1,0].set_title(f"V_Calculated [MeV], cut x={x[ix]} fm")
    fig.colorbar(im10, ax=axes[1, 0])
    im11=axes[1,1].imshow(V_calc[ix_min:ix_max, iy, iz_min:iz_max].T, origin='lower', extent=[x[ix_min], x[ix_max-1], z[iz_min], z[iz_max-1]], aspect='auto', cmap='plasma')
    axes[1,1].set_title(f"V_Calculated [MeV], cut y={y[iy]} fm")
    fig.colorbar(im11, ax=axes[1, 1])
    im12=axes[1,2].imshow(V_calc[ix_min:ix_max, iy_min:iy_max, iz].T, origin='lower', extent=[x[ix_min], x[ix_max-1], y[iy_min], y[iy_max-1]], aspect='auto', cmap='plasma')
    axes[1,2].set_title(f"V_Calculated [MeV], cut z={z[iz]} fm")
    fig.colorbar(im12, ax=axes[1, 2])

    # -----------------------
    # RIGA 2 = differenza
    im20=axes[2,0].imshow(diff[ix, iy_min:iy_max, iz_min:iz_max].T, origin='lower', extent=[y[iy_min], y[iy_max-1], z[iz_min], z[iz_max-1]], aspect='auto', cmap='coolwarm')
    axes[2,0].set_title(f"Difference [MeV], cut x={x[ix]} fm")
    fig.colorbar(im20, ax=axes[2, 0])
    im21=axes[2,1].imshow(diff[ix_min:ix_max, iy, iz_min:iz_max].T, origin='lower', extent=[x[ix_min], x[ix_max-1], z[iz_min], z[iz_max-1]], aspect='auto', cmap='coolwarm')
    axes[2,1].set_title(f"Difference [MeV], cut y={y[iy]} fm")
    fig.colorbar(im21, ax=axes[2, 1])
    im22=axes[2,2].imshow(diff[ix_min:ix_max, iy_min:iy_max, iz].T, origin='lower', extent=[x[ix_min], x[ix_max-1], y[iy_min], y[iy_max-1]], aspect='auto', cmap='coolwarm')
    axes[2,2].set_title(f"Difference [MeV], cut z={z[iz]} fm")
    fig.colorbar(im22, ax=axes[2, 2])

    labels = [
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]"),
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]"),
    ("y [fm]", "z [fm]"), ("x [fm]", "z [fm]"), ("x [fm]", "y [fm]")
    ]

    for ax, (xl, yl) in zip(axes.flat, labels):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def plot_potential(grid, V_analytic, V_ipopt, filename):
    diff=V_ipopt-V_analytic

    fig, axes=plt.subplots(3, 1, figsize=(8, 12), sharex=False)

    axes[0].plot(grid, V_analytic, 'b', label='Target')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel(r'$V(x) [MeV]$')
    axes[0].set_title("potenziale analitico")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(grid, V_ipopt, 'r', label='Calculated')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel(r'$V_{ipopt}(x) [MeV]$')
    axes[1].legend()
    axes[1].set_title("potenziale calcolato")
    axes[1].grid(True)

    axes[2].plot(grid, diff, 'k', label='Difference')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel(r'$\Delta V(x) [MeV]$')
    axes[2].set_title("ΔV")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)



def plot_potential_single_panel(grid, V_analytic, V_ipopt, filename):
    diff = V_ipopt - V_analytic

    plt.figure(figsize=(8,6))
    plt.plot(grid, V_analytic, 'b', label='V analitico')
    plt.plot(grid, V_ipopt, 'r', label='V calcolato')
    plt.plot(grid, diff, 'k--', label='ΔV')

    plt.xlabel('x')
    plt.ylabel('Potenziale')
    plt.title('Confronto potenziale')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()