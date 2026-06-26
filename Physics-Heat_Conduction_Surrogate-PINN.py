"""
Physics-Informed Neural Network (PINN) for 2D Steady-State Heat Conduction
on a rectangular domain with a circular hole.

PDE:  k * (∂²T/∂x² + ∂²T/∂y²) + Q(x,y) = 0    in Ω \ hole
BCs:
  T(0, y) = T_LEFT   (left wall, Dirichlet)
  T(1, y) = T_RIGHT  (right wall, Dirichlet)
  ∂T/∂y = 0  at y=0 and y=1  (top/bottom, Neumann/insulated)
  ∂T/∂n = 0  on hole boundary (insulated hole, Neumann)

k = 1.0 (given)
Q = Gaussian heat source centered inside the domain
"""

import matplotlib
matplotlib.use('TkAgg')  # change to 'Qt5Agg' if you have PyQt5 instead

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Problem parameters ───────────────────────────────────────────────────────
T_LEFT  = 100.0
T_RIGHT =  20.0
K       =   1.0

HOLE_CX, HOLE_CY, HOLE_R = 0.5, 0.5, 0.15

SRC_CX, SRC_CY = 0.65, 0.30
SRC_AMP  = 300.0
SRC_SIG  =  0.08

def Q_source(x, y):
    r2 = (x - SRC_CX)**2 + (y - SRC_CY)**2
    return SRC_AMP * torch.exp(-r2 / (2 * SRC_SIG**2))

def in_hole(x, y):
    return ((x - HOLE_CX)**2 + (y - HOLE_CY)**2) < HOLE_R**2

# ── Neural network ───────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, layers=(2, 64, 64, 64, 64, 1)):
        super().__init__()
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

# ── Collocation point samplers ───────────────────────────────────────────────
def sample_interior(n):
    pts = []
    while len(pts) < n:
        batch = torch.rand(n * 3, 2)
        mask = ~in_hole(batch[:, 0], batch[:, 1])
        pts.append(batch[mask])
    pts = torch.cat(pts, dim=0)[:n]
    return pts[:, 0:1].requires_grad_(True), pts[:, 1:2].requires_grad_(True)

def sample_left_bc(n):
    return torch.zeros(n, 1).requires_grad_(True), torch.rand(n, 1).requires_grad_(True)

def sample_right_bc(n):
    return torch.ones(n, 1).requires_grad_(True), torch.rand(n, 1).requires_grad_(True)

def sample_top_bottom_bc(n):
    x = torch.rand(n, 1)
    y = torch.cat([torch.zeros(n//2, 1), torch.ones(n - n//2, 1)], dim=0)
    return x.requires_grad_(True), y.requires_grad_(True)

def sample_hole_bc(n):
    theta = torch.rand(n, 1) * 2 * np.pi
    x = HOLE_CX + HOLE_R * torch.cos(theta)
    y = HOLE_CY + HOLE_R * torch.sin(theta)
    return x.requires_grad_(True), y.requires_grad_(True)

# ── Gradient helpers ─────────────────────────────────────────────────────────
def grad(u, v, create_graph=True):
    return torch.autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                               create_graph=create_graph, retain_graph=True)[0]

# ── Loss functions ───────────────────────────────────────────────────────────
def pde_loss(model, x, y):
    T = model(x, y)
    Txx = grad(grad(T, x), x)
    Tyy = grad(grad(T, y), y)
    return (K * (Txx + Tyy) + Q_source(x, y)).pow(2).mean()

def dirichlet_loss(model, x, y, T_val):
    return (model(x, y) - T_val).pow(2).mean()

def neumann_loss_y(model, x, y):
    T = model(x, y)
    return grad(T, y).pow(2).mean()

def neumann_hole_loss(model, x_h, y_h):
    T  = model(x_h, y_h)
    Tx = grad(T, x_h)
    Ty = grad(T, y_h)
    nx = (x_h - HOLE_CX) / HOLE_R
    ny = (y_h - HOLE_CY) / HOLE_R
    return (Tx * nx + Ty * ny).pow(2).mean()

# ── Live loss plot ────────────────────────────────────────────────────────────
def setup_live_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.tick_params(colors="#aaa", labelsize=9)
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    ax.title.set_color("#eee")

    line_total, = ax.semilogy([], [], color="#7af", lw=1.5, label="Total")
    line_pde,   = ax.semilogy([], [], color="#f87", lw=1.0, label="PDE",       alpha=0.85)
    line_dir,   = ax.semilogy([], [], color="#7f7", lw=1.0, label="Dirichlet", alpha=0.85)
    line_neu,   = ax.semilogy([], [], color="#fa7", lw=1.0, label="Neumann",   alpha=0.85)

    ax.set_xlim(0, 1)
    ax.set_ylim(1e-6, 10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("PINN training — live loss", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor="#ccc")
    ax.grid(True, color="#2a2a2a", linewidth=0.5)
    fig.tight_layout()
    plt.pause(0.01)
    return fig, ax, (line_total, line_pde, line_dir, line_neu)


def update_live_plot(fig, ax, lines, history, epoch, n_epochs, update_every):
    if epoch % update_every != 0 and epoch != 1:
        return
    line_total, line_pde, line_dir, line_neu = lines
    epochs = np.arange(1, len(history["total"]) + 1)
    line_total.set_data(epochs, history["total"])
    line_pde.set_data(epochs,   history["pde"])
    line_dir.set_data(epochs,   history["dir"])
    line_neu.set_data(epochs,   history["neu"])
    ax.set_xlim(1, n_epochs)
    all_vals = history["total"] + history["pde"] + history["dir"] + history["neu"]
    ax.set_ylim(max(min(all_vals) * 0.5, 1e-8), max(all_vals) * 2)
    ax.set_title(f"PINN training — epoch {epoch}/{n_epochs}  "
                 f"(loss={history['total'][-1]:.3e})", fontsize=10)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


# ── Training ─────────────────────────────────────────────────────────────────
def train(n_epochs=8000,
          n_interior=2000,
          n_bc=400,
          lr=1e-3,
          w_pde=1.0,
          w_dir=10.0,
          w_neu=1.0,
          live_plot=True,
          update_every=2):          # <-- single consistent name

    model = PINN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=3000, gamma=0.5)
    history = {"total": [], "pde": [], "dir": [], "neu": []}

    if live_plot:
        fig_live, ax_live, lines_live = setup_live_plot()

    print(f"\nTraining PINN for {n_epochs} epochs...")
    print(f"{'Epoch':>8} {'Total':>12} {'PDE':>12} {'Dirichlet':>12} {'Neumann':>12}")
    print("-" * 60)

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        xi, yi         = [t.to(device) for t in sample_interior(n_interior)]
        xl, yl         = [t.to(device) for t in sample_left_bc(n_bc)]
        xr, yr         = [t.to(device) for t in sample_right_bc(n_bc)]
        xtb, ytb       = [t.to(device) for t in sample_top_bottom_bc(n_bc)]
        x_hole, y_hole = [t.to(device) for t in sample_hole_bc(n_bc)]

        l_pde   = pde_loss(model, xi, yi)
        l_dir   = dirichlet_loss(model, xl, yl, T_LEFT) + dirichlet_loss(model, xr, yr, T_RIGHT)
        l_neu   = neumann_loss_y(model, xtb, ytb) + neumann_hole_loss(model, x_hole, y_hole)
        loss    = w_pde * l_pde + w_dir * l_dir + w_neu * l_neu

        loss.backward()
        optimiser.step()
        scheduler.step()

        history["total"].append(loss.item())
        history["pde"].append(l_pde.item())
        history["dir"].append(l_dir.item())
        history["neu"].append(l_neu.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>8d} {loss.item():>12.4e} {l_pde.item():>12.4e} "
                  f"{l_dir.item():>12.4e} {l_neu.item():>12.4e}")

        if live_plot:
            update_live_plot(fig_live, ax_live, lines_live,
                             history, epoch, n_epochs, update_every)  # <-- uses update_every

    if live_plot:
        update_live_plot(fig_live, ax_live, lines_live,
                         history, epoch, n_epochs, update_every=10)
        plt.ioff()
        fig_live.savefig("pinn_loss_live.png", dpi=150, bbox_inches="tight",
                         facecolor=fig_live.get_facecolor())
        print("Live loss plot saved to pinn_loss_live.png")

    print("\nTraining complete.")
    return model, history

# ── Visualisation ─────────────────────────────────────────────────────────────
def visualise(model, history, resolution=200):
    model.eval()
    xs = np.linspace(0, 1, resolution)
    ys = np.linspace(0, 1, resolution)
    XX, YY = np.meshgrid(xs, ys)

    x_flat = torch.tensor(XX.ravel(), dtype=torch.float32).unsqueeze(1).to(device)
    y_flat = torch.tensor(YY.ravel(), dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        T_pred = model(x_flat, y_flat).cpu().numpy().reshape(resolution, resolution)

    hole_mask = (XX - HOLE_CX)**2 + (YY - HOLE_CY)**2 < HOLE_R**2
    T_plot = np.where(hole_mask, np.nan, T_pred)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f0f0f")
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                          left=0.07, right=0.93, top=0.93, bottom=0.07)
    ax_T    = fig.add_subplot(gs[0, :])
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_line = fig.add_subplot(gs[1, 1])

    for ax in [ax_T, ax_loss, ax_line]:
        ax.set_facecolor("#1a1a1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        ax.tick_params(colors="#aaa", labelsize=9)
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")

    im = ax_T.imshow(T_plot, origin="lower", extent=[0,1,0,1],
                     cmap="inferno", aspect="equal",
                     vmin=np.nanmin(T_plot), vmax=np.nanmax(T_plot))
    cbar = fig.colorbar(im, ax=ax_T, fraction=0.025, pad=0.01)
    cbar.set_label("Temperature (°)", color="#ccc", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#aaa", labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaa")

    ax_T.add_patch(plt.Circle((HOLE_CX, HOLE_CY), HOLE_R,
                               fill=False, edgecolor="white", lw=1.5, linestyle="--"))
    ax_T.text(HOLE_CX, HOLE_CY, "hole", ha="center", va="center", color="white", fontsize=8)
    ax_T.add_patch(plt.Circle((SRC_CX, SRC_CY), 2*SRC_SIG,
                               fill=False, edgecolor="#ffdd55", lw=1.2, linestyle=":"))
    ax_T.text(SRC_CX, SRC_CY + 2*SRC_SIG + 0.03, "Q source",
              ha="center", color="#ffdd55", fontsize=8)
    ax_T.axvline(0, color="#4af", lw=2, alpha=0.7)
    ax_T.axvline(1, color="#f84", lw=2, alpha=0.7)
    ax_T.text(0.01, 0.5, f"T={T_LEFT:.0f}°", color="#4af", fontsize=9, va="center", rotation=90)
    ax_T.text(0.98, 0.5, f"T={T_RIGHT:.0f}°", color="#f84", fontsize=9,
              va="center", rotation=90, ha="right")
    ax_T.set_title("PINN — Predicted Temperature Field  (k=1, steady-state)", fontsize=11)
    ax_T.set_xlabel("x"); ax_T.set_ylabel("y")

    epochs = np.arange(1, len(history["total"]) + 1)
    ax_loss.semilogy(epochs, history["total"], color="#7af", lw=1.5, label="Total")
    ax_loss.semilogy(epochs, history["pde"],   color="#f87", lw=1,   label="PDE",       alpha=0.8)
    ax_loss.semilogy(epochs, history["dir"],   color="#7f7", lw=1,   label="Dirichlet", alpha=0.8)
    ax_loss.semilogy(epochs, history["neu"],   color="#fa7", lw=1,   label="Neumann",   alpha=0.8)
    ax_loss.set_title("Training loss history", fontsize=10)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8, framealpha=0.2, labelcolor="#ccc")
    ax_loss.grid(True, color="#333", linewidth=0.5)

    mid_row = resolution // 2
    T_mid = T_plot[mid_row, :]
    valid = ~np.isnan(T_mid)
    ax_line.plot(xs[valid], T_mid[valid], color="#7af", lw=2)
    ax_line.axhline(T_LEFT,  color="#4af", lw=1, linestyle="--", alpha=0.6, label=f"T_L={T_LEFT:.0f}°")
    ax_line.axhline(T_RIGHT, color="#f84", lw=1, linestyle="--", alpha=0.6, label=f"T_R={T_RIGHT:.0f}°")
    ax_line.set_title("Temperature profile at y = 0.5", fontsize=10)
    ax_line.set_xlabel("x"); ax_line.set_ylabel("T(x, 0.5)")
    ax_line.legend(fontsize=8, framealpha=0.2, labelcolor="#ccc")
    ax_line.grid(True, color="#333", linewidth=0.5)

    plt.savefig("pinn_heat_result.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Plot saved to pinn_heat_result.png")
    plt.show()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, history = train(
        n_epochs=100,
        n_interior=2000,
        n_bc=400,
        lr=1e-3,
        w_pde=1.0,
        w_dir=10.0,
        w_neu=1.0,
        live_plot=True,
        update_every=2,
    )
    visualise(model, history, resolution=200)