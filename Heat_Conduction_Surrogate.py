import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import meshio
import pyvista as pv
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================================================
# Parameters for DeepONet Training and HyperParameter Tuning  
# =========================================================
Num_Neurons =128
LearningRate = 1e-3
Num_Epochs = 500
# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# SORT KEY
# =========================================================
def numeric_sort_key(name):
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1

# =========================================================
# TEMPERATURE EXTRACTION
# =========================================================
def extract_temperature(mesh):
    keys = mesh.point_data.keys()
    candidates = ["T", "temperature", "Temperature", "TEMP", "temp", "NT11"]

    for c in candidates:
        if c in keys:
            return np.asarray(mesh.point_data[c], dtype=np.float32)

    if len(keys) > 0:
        return np.asarray(mesh.point_data[list(keys)[0]], dtype=np.float32)

    raise KeyError(f"No temperature field found: {list(keys)}")


# =========================================================
# Six (6) PARAMS LOADER
# =========================================================
def load_params(json_path):
    with open(json_path, "r") as f:
        p = json.load(f)

    def g(keys):
        for k in keys:
            if k in p:
                return p[k]
        raise KeyError(f"Missing keys: {keys}")

    return np.array([
        g(["L", "length", "Lx"]),
        g(["B", "breadth", "Ly"]),
        g(["R", "radius", "r"]),
        g(["q", "heat_source", "qvol"]),
        g(["T_left", "T_L", "Tleft"]),
        g(["T_right", "T_R", "Tright"]),
    ], dtype=np.float32)


# =========================================================
# MESH COORDS
# =========================================================
def load_mesh_coords(mesh_path):
    mesh = meshio.read(mesh_path)
    return mesh.points[:, :2].astype(np.float32)


# =========================================================
# DATASET
# =========================================================
class HeatDataset(Dataset):
    def __init__(self, root="data"):
        self.samples = []

        root_path = os.path.join(os.path.dirname(__file__), root)
        print("Scanning:", root_path)

        folders = sorted(os.listdir(root_path), key=numeric_sort_key)

        for s in folders:
            p = os.path.join(root_path, s)
            if not os.path.isdir(p):
                continue

            files = os.listdir(p)

            if any(f.endswith(".msh") for f in files) and \
               any(f.endswith(".vtp") for f in files) and \
               any(f.endswith(".json") for f in files):
                self.samples.append(p)

        print("Total samples:", len(self.samples))
        if len(self.samples) == 0:
            raise RuntimeError("Dataset empty")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p = self.samples[i]

        msh = [f for f in os.listdir(p) if f.endswith(".msh")][0]
        vtp = [f for f in os.listdir(p) if f.endswith(".vtp")][0]
        jsn = [f for f in os.listdir(p) if f.endswith(".json")][0]

        coords = load_mesh_coords(os.path.join(p, msh))
        params = load_params(os.path.join(p, jsn))

        mesh = pv.read(os.path.join(p, vtp))
        T = extract_temperature(mesh)

        return (
            torch.tensor(coords, dtype=torch.float32),
            torch.tensor(params, dtype=torch.float32),
            torch.tensor(T, dtype=torch.float32),
        )


# =========================================================
# MODEL
# =========================================================
class Branch(nn.Module):
    def __init__(self, in_dim=6, w=Num_Neurons):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, w),
            nn.ReLU(),
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, w),
        )

    def forward(self, x):
        return self.net(x)


class Trunk(nn.Module):
    def __init__(self, in_dim=2, w=Num_Neurons):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, w),
            nn.ReLU(),
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, w),
        )

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self, w=Num_Neurons):
        super().__init__()
        self.branch = Branch(w=w)
        self.trunk = Trunk(w=w)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, coords, params):
        b = self.branch(params)
        t = self.trunk(coords)
        return torch.einsum("bw,bnw->bn", b, t) + self.bias


def train(model, loader, device, epochs=5, lr=1e-3, max_points=4096):

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss_hist = []
    rel_hist = []

    # ============================
    # LIVE PLOTTING SETUP
    # ============================
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    fig.subplots_adjust(hspace=0.5)

    line1, = ax[0].plot([], [], label="Loss")
    ax[0].set_title("Loss vs Epochs")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)
    ax[0].legend()

    line2, = ax[1].plot([], [], label="Rel L2")
    ax[1].set_title("Relative L2 vs Epochs")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Rel L2")
    ax[1].grid(True)
    ax[1].legend()

    for ep in range(epochs):
        model.train()

        total_loss = 0
        total_rel = 0
        n = 0

        for coords, params, T in tqdm(loader, desc=f"Epoch {ep+1}"):

            coords = coords.squeeze(0)
            params = params.squeeze(0)
            T = T.squeeze(0)

            if coords.shape[0] > max_points:
                idx = torch.randperm(coords.shape[0])[:max_points]
                coords = coords[idx]
                T = T[idx]

            coords = coords.to(device).unsqueeze(0)
            params = params.to(device).unsqueeze(0)
            T = T.to(device).unsqueeze(0)

            pred = model(coords, params)

            loss = loss_fn(pred, T)
            rel = torch.norm(pred - T) / (torch.norm(T) + 1e-12)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_rel += rel.item()
            n += 1

        # ============================
        # STORE HISTORY
        # ============================
        loss_hist.append(total_loss / n)
        rel_hist.append(total_rel / n)

        print(f"Epoch {ep+1} | Loss: {loss_hist[-1]:.6e} | Rel L2: {rel_hist[-1]:.6e}")

        # ============================
        # LIVE UPDATE PLOT
        # ============================
        x = list(range(len(loss_hist)))

        line1.set_xdata(x)
        line1.set_ydata(loss_hist)

        line2.set_xdata(x)
        line2.set_ydata(rel_hist)

        for a in ax:
            a.relim()
            a.autoscale_view()

        plt.pause(0.01)
        plt.draw()

    plt.ioff()
    plt.show()

    return loss_hist, rel_hist


# =========================================================
# EVALUATION
# =========================================================
def predict_all(model, dataset, device):

    model.eval()
    model.to(device)

    print("\n========== EVALUATION ==========\n")

    for i, path in enumerate(dataset.samples):

        msh = [f for f in os.listdir(path) if f.endswith(".msh")][0]
        vtp = [f for f in os.listdir(path) if f.endswith(".vtp")][0]
        jsn = [f for f in os.listdir(path) if f.endswith(".json")][0]

        coords = load_mesh_coords(os.path.join(path, msh))
        params = load_params(os.path.join(path, jsn))

        mesh = pv.read(os.path.join(path, vtp))
        T_true = extract_temperature(mesh)

        coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)
        params_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            T_pred = model(coords_t, params_t).cpu().numpy().squeeze()

        T_true = T_true.astype(np.float32)
        T_pred = T_pred.astype(np.float32)

        error = T_pred - T_true

        print(f"\nSample {i+1}")
        print("T_true[:10]:", T_true[:10])
        print("T_pred[:10]:", T_pred[:10])

        print("\n--- Stats ---")
        print(f"Min/Max True: {T_true.min():.4f}/{T_true.max():.4f}")
        print(f"Min/Max Pred: {T_pred.min():.4f}/{T_pred.max():.4f}")
        print(f"MAE: {np.mean(np.abs(error)):.6e}")
        print(f"RMSE: {np.sqrt(np.mean(error**2)):.6e}")

        mesh.point_data["T_true"] = T_true
        mesh.point_data["T_pred"] = T_pred
        mesh.point_data["error"] = error

        out_path = os.path.join(path, "prediction.vtp")
        mesh.save(out_path)

        print("Saved:", out_path)


# =========================================================
# INFERENCE  
# =========================================================
def infer_from_sample_folder(model, folder_path, device, out_name="inference_new_case.vtp"):

    model.eval()
    model.to(device)

    msh_file = [f for f in os.listdir(folder_path) if f.endswith(".msh")][0]
    msh_path = os.path.join(folder_path, msh_file)

    mesh = meshio.read(msh_path)
    coords = mesh.points[:, :2].astype(np.float32)

    coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)

    params = np.array([1.2, 0.8, 0.15, 5e5, 300.0, 350.0], dtype=np.float32)
    params_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        T_pred = model(coords_t, params_t).cpu().numpy().squeeze()

    # convert to PyVista
    mesh_vtk = pv.read(msh_path)
    mesh_vtk.point_data["T_pred"] = T_pred
    mesh_vtk.set_active_scalars("T_pred")

    # Convert to surface so .vtp is valid
    mesh_vtk = mesh_vtk.extract_surface()

    out_path = os.path.join(folder_path, out_name)
    mesh_vtk.save(out_path)

    print("\n========== INFERENCE ==========")
    print("Mesh used:", msh_file)
    print("Saved:", out_path)

    return T_pred


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    dataset = HeatDataset("data")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = DeepONet(w=Num_Neurons)

    loss_hist, rel_hist = train(model, loader, device=device, epochs=Num_Epochs, lr=LearningRate)

    plt.figure()
    plt.plot(rel_hist)
    plt.title("Rel L2 vs Epochs")
    plt.grid()
    #plt.show()

    plt.figure()
    plt.plot(loss_hist)
    plt.title("Loss vs Epochs")
    plt.grid()
    #plt.show()

    predict_all(model, dataset, device)

    # =====================================================
    # CUSTOM INFERENCE
    # =====================================================
    print("\n========== CUSTOM INFERENCE ==========")

    infer_folder = os.path.join(os.path.dirname(__file__), "Sample_inference")

    infer_from_sample_folder(
        model=model,
        folder_path=infer_folder,
        device=device,
        out_name="inference_new_case.vtp"
    )