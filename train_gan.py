
import os
import sys
import json
import time
import glob
import math
import shutil
import struct
import random
import textwrap
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ── Colour helpers (reused from train.py) ──────────────────────────────────
RESET="\033[0m";BOLD="\033[1m";CYAN="\033[96m";GREEN="\033[92m"
YELLOW="\033[93m";RED="\033[91m";MAGENTA="\033[95m";GREY="\033[90m"
WHITE="\033[97m"
def _c(t,c): return f"{c}{t}{RESET}"
def _hdr(title):
    w=72
    print("╔"+"═"*(w-2)+"╗")
    print("║"+_c(title.center(w-2),BOLD+CYAN)+"║")
    print("╚"+"═"*(w-2)+"╝")
def _sec(t): print(f"\n{GREY}{'─'*28}{RESET} {BOLD}{YELLOW}{t}{RESET} {GREY}{'─'*28}{RESET}")
def _kv(k,v,col=WHITE,w=30): print(f"  {GREY}{k:<{w}}{RESET}{col}{v}{RESET}")
def _ok(msg): print(f"  {GREEN}✔{RESET} {msg}")
def _info(msg): print(f"  {CYAN}ℹ{RESET} {msg}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
GAN_OUTPUT_DIR = "gan_output"          # where your trained GAN saved images
VR_OUTPUT_DIR  = "vr_output"          # all VR artefacts go here
MESH_DIR       = os.path.join(VR_OUTPUT_DIR, "meshes")
SCENE_DIR      = os.path.join(VR_OUTPUT_DIR, "scenes")
METRICS_DIR    = os.path.join(VR_OUTPUT_DIR, "metrics")
FINAL_DIR      = os.path.join(VR_OUTPUT_DIR, "final")   # ← deliverable folder

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_IMAGES_PER_CLASS = 4      # how many GAN images to process per class
MESH_RESOLUTION      = 128    # grid resolution of displacement mesh (NxN)
DEPTH_SCALE          = 0.15   # depth exaggeration factor for 3-D effect
POINT_CLOUD_SAMPLES  = 4096   # points sampled per image for metrics
TARGET_FPS           = 72     # target VR frame rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Collect GAN images
# ══════════════════════════════════════════════════════════════════════════════
def collect_gan_images(gan_dir=GAN_OUTPUT_DIR):
    """Return {class_name: [image_path, …]} from the GAN output folder."""
    _sec("Step 1 — Collecting GAN-Generated Images")
    catalog = {}
    if not os.path.isdir(gan_dir):
        _info(f"'{gan_dir}' not found — generating synthetic demo images instead.")
        return _make_demo_images()

    for cls in sorted(os.listdir(gan_dir)):
        cls_path = os.path.join(gan_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = [
            os.path.join(cls_path, f)
            for f in sorted(os.listdir(cls_path))
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
        if imgs:
            catalog[cls] = imgs[:MAX_IMAGES_PER_CLASS]

    if not catalog:
        _info("No images found in gan_output — using synthetic demo images.")
        return _make_demo_images()

    total = sum(len(v) for v in catalog.values())
    _ok(f"Found {total} images across {len(catalog)} classes: {list(catalog.keys())}")
    return catalog


def _make_demo_images():
    """Create synthetic colourful craft-like demo images when no GAN output exists."""
    demo_dir = os.path.join(VR_OUTPUT_DIR, "_demo_gan")
    classes  = ["pottery", "weaving", "metalwork", "woodcraft"]
    catalog  = {}
    for cls in classes:
        cls_dir = os.path.join(demo_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        paths = []
        for i in range(MAX_IMAGES_PER_CLASS):
            np.random.seed(i + hash(cls) % 1000)
            h = w = 256
            img = np.zeros((h, w, 3), dtype=np.uint8)
            # layered random patterns that vaguely mimic craft textures
            for _ in range(6):
                cx, cy = np.random.randint(40, 216, 2)
                r      = np.random.randint(20, 80)
                col    = np.random.randint(80, 255, 3).tolist()
                cv2.circle(img, (cx, cy), r, col, -1)
            img = cv2.GaussianBlur(img, (15, 15), 0)
            noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
            img   = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            path  = os.path.join(cls_dir, f"epoch_progress_{(i+1)*10}.png")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            paths.append(path)
        catalog[cls] = paths
    _ok(f"Created synthetic demo images for classes: {classes}")
    return catalog


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Depth Estimation (MiDaS or fallback)
# ══════════════════════════════════════════════════════════════════════════════
_midas_model  = None
_midas_transform = None

def _load_midas():
    global _midas_model, _midas_transform
    if _midas_model is not None:
        return True
    try:
        _midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                      trust_repo=True).to(DEVICE).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                          trust_repo=True)
        _midas_transform = midas_transforms.small_transform
        _ok("MiDaS depth model loaded.")
        return True
    except Exception as e:
        _info(f"MiDaS unavailable ({e}) — using gradient-based pseudo-depth.")
        return False


def estimate_depth(img_np_rgb: np.ndarray) -> np.ndarray:
    """
    Returns a [0,1] depth map (float32, H×W).
    Uses MiDaS if available, otherwise a Sobel-edge + Gaussian blend heuristic.
    """
    if _load_midas():
        try:
            inp = _midas_transform(img_np_rgb).to(DEVICE)
            with torch.no_grad():
                pred = _midas_model(inp)
                pred = F.interpolate(pred.unsqueeze(1),
                                     size=img_np_rgb.shape[:2],
                                     mode="bilinear",
                                     align_corners=False).squeeze()
            d = pred.cpu().numpy().astype(np.float32)
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            return d
        except Exception:
            pass

    # Fallback: Sobel edges + low-freq luminance
    gray = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    sy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    edges = np.sqrt(sx**2 + sy**2)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    blur  = gaussian_filter(gray, sigma=15)
    depth = 0.6 * (1.0 - blur) + 0.4 * edges
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Generate OBJ Mesh (displacement mesh)
# ══════════════════════════════════════════════════════════════════════════════
def depth_to_obj(depth_map: np.ndarray,
                 texture_name: str,
                 resolution: int = MESH_RESOLUTION,
                 z_scale: float = DEPTH_SCALE) -> str:
    """
    Build a simple displacement-mesh OBJ string from a depth map.
    UV coords map to the original texture image.
    """
    H, W  = depth_map.shape
    res   = resolution
    # sample grid
    ys = np.linspace(0, H - 1, res).astype(int)
    xs = np.linspace(0, W - 1, res).astype(int)
    grid_d = depth_map[np.ix_(ys, xs)]   # res×res

    lines = [f"# Craft Workshop Displacement Mesh — {texture_name}",
             f"mtllib {os.path.splitext(texture_name)[0]}.mtl", ""]

    # vertices
    for iy, y in enumerate(np.linspace(0, 1, res)):
        for ix, x in enumerate(np.linspace(0, 1, res)):
            z = float(grid_d[iy, ix]) * z_scale
            lines.append(f"v {x:.6f} {1.0-y:.6f} {z:.6f}")

    lines.append("")
    # texture coords
    for iy in range(res):
        for ix in range(res):
            u = ix / (res - 1)
            v = 1.0 - iy / (res - 1)
            lines.append(f"vt {u:.6f} {v:.6f}")

    lines.append("")
    lines.append(f"usemtl mat_{os.path.splitext(texture_name)[0]}")
    lines.append("s 1")

    # faces (two triangles per quad)
    def vi(iy, ix): return iy * res + ix + 1   # 1-indexed

    for iy in range(res - 1):
        for ix in range(res - 1):
            a = vi(iy,   ix);   b = vi(iy,   ix+1)
            c = vi(iy+1, ix+1); d = vi(iy+1, ix)
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
            lines.append(f"f {a}/{a} {c}/{c} {d}/{d}")

    return "\n".join(lines)


def write_mtl(mtl_path: str, texture_filename: str):
    name = os.path.splitext(os.path.basename(mtl_path))[0]
    mtl  = textwrap.dedent(f"""\
        newmtl mat_{name}
        Ka 1.000 1.000 1.000
        Kd 1.000 1.000 1.000
        Ks 0.000 0.000 0.000
        d 1.0
        illum 2
        map_Kd {texture_filename}
    """)
    with open(mtl_path, "w") as f:
        f.write(mtl)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build Three.js VR HTML Scene
# ══════════════════════════════════════════════════════════════════════════════
def build_vr_html(scene_entries: list, out_path: str):
    """
    scene_entries: list of dicts with keys:
        class_name, image_b64, depth_array (H×W float32), obj_filename
    Produces a self-contained Three.js + WebXR HTML file.
    """

    # Encode depth maps as compact JSON arrays (down-sampled for JS)
    def _depth_js(d):
        ds = cv2.resize(d, (64, 64), interpolation=cv2.INTER_AREA)
        return json.dumps(ds.flatten().tolist())

    entries_js = json.dumps([
        {"class": e["class_name"],
         "depth64": json.loads(_depth_js(e["depth"])),
         "color":   e.get("avg_color", "#a07850")}
        for e in scene_entries
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Craft Workshop — VR Experience</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0a0812;font-family:'Courier New',monospace;overflow:hidden}}
  #canvas{{width:100vw;height:100vh;display:block}}
  #hud{{position:fixed;top:16px;left:16px;color:#e8d5a3;font-size:13px;
        background:rgba(10,8,18,.72);border:1px solid #5a3e28;border-radius:6px;
        padding:12px 16px;max-width:280px;pointer-events:none;z-index:10}}
  #hud h2{{font-size:15px;color:#f0c060;margin-bottom:6px;letter-spacing:.08em}}
  #hud .metric{{display:flex;justify-content:space-between;gap:16px;
                border-top:1px solid #3a2a18;margin-top:4px;padding-top:4px}}
  #hud .val{{color:#7affb0}}
  #controls{{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);
             display:flex;gap:10px;z-index:10}}
  .btn{{background:#1e1428;color:#e8d5a3;border:1px solid #5a3e28;border-radius:4px;
        padding:8px 18px;cursor:pointer;font-family:inherit;font-size:13px;
        transition:background .2s}}
  .btn:hover{{background:#3a2a18}}
  #loader{{position:fixed;inset:0;background:#0a0812;display:flex;align-items:center;
           justify-content:center;flex-direction:column;gap:12px;color:#e8d5a3;
           font-size:16px;z-index:100}}
  .spinner{{width:40px;height:40px;border:3px solid #3a2a18;
            border-top-color:#f0c060;border-radius:50%;animation:spin 1s linear infinite}}
  @keyframes spin{{to{{transform:rotate(360deg)}}}}
</style>
</head>
<body>
<div id="loader"><div class="spinner"></div><span>Initialising VR Scene…</span></div>
<canvas id="canvas"></canvas>

<div id="hud">
  <h2>⚒ Craft Workshop VR</h2>
  <div class="metric"><span>Active Class</span><span class="val" id="h-class">—</span></div>
  <div class="metric"><span>FPS</span><span class="val" id="h-fps">—</span></div>
  <div class="metric"><span>Draw Calls</span><span class="val" id="h-dc">—</span></div>
  <div class="metric"><span>Triangles</span><span class="val" id="h-tri">—</span></div>
  <div class="metric"><span>Depth Quality</span><span class="val" id="h-dq">—</span></div>
  <div class="metric"><span>VR Ready</span><span class="val" id="h-vr">Checking…</span></div>
</div>

<div id="controls">
  <button class="btn" onclick="prevClass()">◀ Prev</button>
  <button class="btn" onclick="toggleAnim()">⏯ Animate</button>
  <button class="btn" onclick="nextClass()">Next ▶</button>
  <button class="btn" onclick="enterVR()">🥽 Enter VR</button>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── Scene data injected from Python ──────────────────────────────────────
const ENTRIES = {entries_js};

// ── Three.js setup ────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({{canvas:document.getElementById('canvas'),antialias:true}});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth,innerHeight);
renderer.xr.enabled = true;

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x0a0812);
scene.fog = new THREE.FogExp2(0x0a0812, 0.04);

const camera = new THREE.PerspectiveCamera(75, innerWidth/innerHeight, 0.01, 100);
camera.position.set(0, 1.6, 3);

// Lighting
const amb  = new THREE.AmbientLight(0xfff0e0, 0.4);
const dir  = new THREE.DirectionalLight(0xffe8c0, 1.2);
dir.position.set(2, 4, 3);
const pt   = new THREE.PointLight(0xff9940, 0.8, 8);
pt.position.set(-1, 2, 1);
scene.add(amb, dir, pt);

// Floor grid
const grid = new THREE.GridHelper(20, 40, 0x3a2a18, 0x1e1428);
grid.position.y = -0.01;
scene.add(grid);

// ── Build displacement meshes from depth arrays ────────────────────────
const MESH_RES = 64;
const meshes   = [];
let   currentIdx = 0;
let   animating  = true;

function buildMesh(entry) {{
  const geo  = new THREE.PlaneGeometry(3, 3, MESH_RES-1, MESH_RES-1);
  const pos  = geo.attributes.position;
  const dep  = entry.depth64;  // 64*64 flat array

  for (let i = 0; i < pos.count; i++) {{
    const z = dep[i] * 0.6;    // depth exaggeration
    pos.setZ(i, z);
  }}
  geo.computeVertexNormals();

  // Colour vertices by depth
  const cols = new Float32Array(pos.count * 3);
  const c1   = new THREE.Color(entry.color);
  const c2   = new THREE.Color(0xfff8e0);
  for (let i = 0; i < pos.count; i++) {{
    const t   = dep[i];
    const col = c1.clone().lerp(c2, t);
    cols[i*3]=col.r; cols[i*3+1]=col.g; cols[i*3+2]=col.b;
  }}
  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));

  const mat  = new THREE.MeshPhongMaterial({{
    vertexColors: true, side: THREE.DoubleSide,
    shininess: 40, wireframe: false
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y  = 0.3;
  mesh.userData.class = entry.class;
  mesh.visible = false;
  scene.add(mesh);

  // Wireframe overlay
  const wfmat  = new THREE.MeshBasicMaterial({{color:0x5a3e28,wireframe:true,transparent:true,opacity:0.15}});
  const wfmesh = new THREE.Mesh(geo, wfmat);
  mesh.add(wfmesh);

  return mesh;
}}

ENTRIES.forEach(e => meshes.push(buildMesh(e)));
if (meshes.length) meshes[0].visible = true;

function showClass(idx) {{
  meshes.forEach((m,i) => m.visible = (i===idx));
  document.getElementById('h-class').textContent =
    ENTRIES[idx] ? ENTRIES[idx].class : '—';
}}
function nextClass() {{ currentIdx=(currentIdx+1)%meshes.length; showClass(currentIdx); }}
function prevClass() {{ currentIdx=(currentIdx-1+meshes.length)%meshes.length; showClass(currentIdx); }}
function toggleAnim() {{ animating=!animating; }}

// VR
function enterVR() {{
  if (navigator.xr) {{
    navigator.xr.isSessionSupported('immersive-vr').then(ok => {{
      if (ok) renderer.xr.getSession() || renderer.xr.requestSession('immersive-vr');
      else alert('WebXR immersive-vr not supported on this device/browser.');
    }});
  }} else {{ alert('WebXR not available.'); }}
}}

// Check VR capability
if (navigator.xr) {{
  navigator.xr.isSessionSupported('immersive-vr').then(ok =>
    document.getElementById('h-vr').textContent = ok ? '✅ Yes' : '⚠ No WebXR');
}} else {{
  document.getElementById('h-vr').textContent = '❌ Not supported';
}}

// Mouse / touch orbit
let isDragging=false, prevMouse={{x:0,y:0}}, camTheta=0, camPhi=0.3, camR=3;
document.addEventListener('mousedown', e=>{{ isDragging=true; prevMouse={{x:e.clientX,y:e.clientY}}; }});
document.addEventListener('mouseup',   ()=>isDragging=false);
document.addEventListener('mousemove', e=>{{
  if (!isDragging) return;
  camTheta -= (e.clientX-prevMouse.x)*0.005;
  camPhi    = Math.max(.1,Math.min(1.4,camPhi-(e.clientY-prevMouse.y)*0.005));
  prevMouse = {{x:e.clientX,y:e.clientY}};
}});
document.addEventListener('wheel', e=>{{ camR=Math.max(1,Math.min(8,camR+e.deltaY*.005)); }});
window.addEventListener('resize', ()=>{{
  camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
}});

// FPS counter
let fps=0,frames=0,lastFPS=performance.now();

// Render loop
let t0=performance.now();
renderer.setAnimationLoop(()=>{{
  const t  = (performance.now()-t0)*0.001;
  frames++;
  if (performance.now()-lastFPS>500) {{
    fps=Math.round(frames*2); frames=0; lastFPS=performance.now();
    document.getElementById('h-fps').textContent=fps;
    const info=renderer.info;
    document.getElementById('h-dc').textContent=info.render.calls;
    document.getElementById('h-tri').textContent=(info.render.triangles/1000).toFixed(1)+'k';
  }}

  // Gentle auto-rotation
  if (animating && meshes[currentIdx]) {{
    meshes[currentIdx].rotation.z = Math.sin(t*0.3)*0.05;
  }}

  // Orbit camera
  camera.position.x = camR*Math.sin(camTheta)*Math.cos(camPhi);
  camera.position.y = 1.6+camR*Math.sin(camPhi);
  camera.position.z = camR*Math.cos(camTheta)*Math.cos(camPhi);
  camera.lookAt(0, 0.3, 0);

  // Depth quality (live variance of visible mesh depths)
  if (ENTRIES[currentIdx]) {{
    const d=ENTRIES[currentIdx].depth64;
    const mean=d.reduce((a,b)=>a+b,0)/d.length;
    const variance=d.reduce((s,v)=>s+(v-mean)**2,0)/d.length;
    document.getElementById('h-dq').textContent=(variance*100).toFixed(1)+'%';
  }}

  renderer.render(scene,camera);
}});

document.getElementById('loader').style.display='none';
showClass(0);
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    _ok(f"VR HTML scene → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VR Metric Computation
# ══════════════════════════════════════════════════════════════════════════════
def compute_vr_metrics(depth_map: np.ndarray,
                       img_np: np.ndarray) -> dict:
    """
    Compute a set of VR-readiness metrics from depth map + colour image.
    All values are normalised or interpretable standalone.
    """
    d = depth_map.astype(np.float32)

    # Depth range utilisation  (0=flat, 1=full range used)
    depth_range       = float(d.max() - d.min())

    # Depth entropy  (Shannon, normalised 0-1)
    hist, _           = np.histogram(d, bins=32, range=(0,1), density=True)
    hist             += 1e-10
    hist             /= hist.sum()
    entropy           = float(-np.sum(hist * np.log2(hist)))
    depth_entropy     = entropy / np.log2(32)

    # Surface complexity  (mean Sobel magnitude of depth)
    sx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    surface_complexity = float(np.mean(np.sqrt(sx**2 + sy**2)))

    # Occlusion estimate  (fraction of pixels with depth > 0.7)
    occlusion_ratio    = float(np.mean(d > 0.7))

    # Parallax potential  (std of depth — higher = more parallax effect in VR)
    parallax_potential = float(np.std(d))

    # Colour richness  (mean saturation in HSV)
    hsv  = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    colour_richness    = float(np.mean(hsv[:,:,1]) / 255.0)

    # Estimated render cost (proxy: mesh triangles / 1000 at MESH_RESOLUTION)
    n_tri              = 2 * (MESH_RESOLUTION - 1) ** 2
    render_cost_ktri   = n_tri / 1000.0

    # Estimated FPS feasibility (very rough heuristic)
    complexity_factor  = 1 + surface_complexity * 5 + occlusion_ratio
    est_fps            = min(TARGET_FPS, TARGET_FPS / complexity_factor * 1.5)

    # VR comfort score (0-100): rewards high depth range, low occlusion, good FPS
    vr_comfort = float(
        30 * depth_range +
        20 * depth_entropy +
        20 * (1 - occlusion_ratio) +
        20 * (min(est_fps, TARGET_FPS) / TARGET_FPS) +
        10 * colour_richness
    ) * 100 / 100  # already 0-1 weighted → ×100 for readability
    vr_comfort = min(100.0, max(0.0, vr_comfort * 100))

    return dict(
        depth_range        = round(depth_range,       4),
        depth_entropy      = round(depth_entropy,     4),
        surface_complexity = round(surface_complexity,4),
        occlusion_ratio    = round(occlusion_ratio,   4),
        parallax_potential = round(parallax_potential,4),
        colour_richness    = round(colour_richness,   4),
        render_cost_ktri   = round(render_cost_ktri,  2),
        est_fps            = round(est_fps,            1),
        vr_comfort_score   = round(vr_comfort,         1),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Plot VR Metrics (9 plots mirroring the GAN metric style)
# ══════════════════════════════════════════════════════════════════════════════
def plot_vr_metrics(all_metrics: dict, save_dir: str):
    """
    all_metrics: {class_name: [metric_dict, …]}
    Saves 6 publication-quality plots to save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor":  "#0d0d1a",
        "axes.facecolor":    "#161628",
        "axes.edgecolor":    "#3a3a5c",
        "axes.labelcolor":   "#c8c8e8",
        "xtick.color":       "#7777aa",
        "ytick.color":       "#7777aa",
        "grid.color":        "#252545",
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "text.color":        "#c8c8e8",
        "font.family":       "monospace",
        "legend.framealpha": 0.15,
        "legend.edgecolor":  "#555577",
    })

    classes   = list(all_metrics.keys())
    n_classes = len(classes)
    palette   = ["#00d4ff","#ff6b6b","#7fff00","#e040fb",
                 "#ffd700","#ff69b4","#40c4ff","#ff8c00"]

    def _avg(cls, key):
        return np.mean([m[key] for m in all_metrics[cls]])

    # ── Plot VR-1: VR Comfort Score per class ────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    scores  = [_avg(c, "vr_comfort_score") for c in classes]
    bars    = ax.bar(classes, scores,
                     color=[palette[i % len(palette)] for i in range(n_classes)],
                     alpha=0.82, width=0.55)
    ax.axhline(70, color="#00ff88", ls="--", lw=1.4, label="Good VR comfort (70)")
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, s+1, f"{s:.1f}",
                ha="center", va="bottom", fontsize=10, color="#e8e8ff")
    ax.set_ylim(0, 105)
    ax.set_title("VR Plot 1 — VR Comfort Score per Craft Class\n"
                 "(higher = smoother, more immersive experience)",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Comfort Score (0–100)")
    ax.legend(); ax.grid(True, axis="y"); plt.tight_layout()
    p = os.path.join(save_dir, "vrplot1_comfort_score.png")
    fig.savefig(p, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig); _ok(f"Saved: {p}")

    # ── Plot VR-2: Depth Range & Entropy ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, label, col in zip(
            axes,
            ["depth_range", "depth_entropy"],
            ["Depth Range (0–1)", "Depth Entropy (0–1)"],
            ["#00d4ff", "#ff6b6b"]):
        vals = [_avg(c, key) for c in classes]
        axes_idx = list(axes).index(ax)
        ax.bar(classes, vals, color=col, alpha=0.78)
        ax.set_title(f"VR Plot 2{'a' if axes_idx==0 else 'b'} — {label}",
                     fontweight="bold")
        ax.set_ylabel(label); ax.grid(True, axis="y")
    plt.suptitle("VR Plot 2 — Depth Quality Indicators", fontweight="bold", y=1.01)
    plt.tight_layout()
    p = os.path.join(save_dir, "vrplot2_depth_quality.png")
    fig.savefig(p, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig); _ok(f"Saved: {p}")

    # ── Plot VR-3: Parallax Potential vs Surface Complexity ───────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cls in enumerate(classes):
        xs = [m["surface_complexity"] for m in all_metrics[cls]]
        ys = [m["parallax_potential"] for m in all_metrics[cls]]
        ax.scatter(xs, ys, label=cls, color=palette[i % len(palette)],
                   s=90, alpha=0.85, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("Surface Complexity"); ax.set_ylabel("Parallax Potential")
    ax.set_title("VR Plot 3 — Parallax Potential vs Surface Complexity\n"
                 "(top-right = best 3D depth illusion in VR)",
                 fontweight="bold", pad=10)
    ax.legend(); ax.grid(True); plt.tight_layout()
    p = os.path.join(save_dir, "vrplot3_parallax_vs_complexity.png")
    fig.savefig(p, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig); _ok(f"Saved: {p}")

    # ── Plot VR-4: Estimated FPS per class ────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cls in enumerate(classes):
        fps_vals = [m["est_fps"] for m in all_metrics[cls]]
        ax.plot(range(1, len(fps_vals)+1), fps_vals,
                marker="o", label=cls, color=palette[i % len(palette)], lw=2)
    ax.axhline(TARGET_FPS, color="#00ff88", ls="--", lw=1.5,
               label=f"Target {TARGET_FPS} FPS")
    ax.axhline(60, color="#ffd700", ls=":", lw=1.2, label="Min comfort 60 FPS")
    ax.set_title("VR Plot 4 — Estimated Render FPS per Image\n"
                 "(≥72 FPS required for comfortable VR)",
                 fontweight="bold", pad=10)
    ax.set_xlabel("Image Index"); ax.set_ylabel("Estimated FPS")
    ax.legend(); ax.grid(True); plt.tight_layout()
    p = os.path.join(save_dir, "vrplot4_estimated_fps.png")
    fig.savefig(p, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig); _ok(f"Saved: {p}")

    # ── Plot VR-5: Occlusion Ratio & Colour Richness (radar-like bar) ─────
    metrics_keys = ["depth_range","depth_entropy","parallax_potential",
                    "colour_richness","occlusion_ratio"]
    metric_labels = ["Depth\nRange","Depth\nEntropy","Parallax\nPotential",
                     "Colour\nRichness","Occlusion\nRatio"]
    x     = np.arange(len(metrics_keys))
    width = 0.8 / max(n_classes, 1)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, cls in enumerate(classes):
        vals = [_avg(cls, k) for k in metrics_keys]
        ax.bar(x + i * width - (n_classes-1)*width/2, vals,
               width=width*0.9, label=cls,
               color=palette[i % len(palette)], alpha=0.82)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_title("VR Plot 5 — VR Readiness Metrics per Class\n"
                 "(higher is better for all except Occlusion Ratio)",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Normalised Value (0–1)")
    ax.legend(); ax.grid(True, axis="y"); plt.tight_layout()
    p = os.path.join(save_dir, "vrplot5_readiness_radar.png")
    fig.savefig(p, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig); _ok(f"Saved: {p}")

    # ── Plot VR-6: Render Cost (k-triangles) vs Comfort Score ────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cls in enumerate(classes):
        xs = [m["render_cost_ktri"] for m in all_metrics[cls]]
        ys = [m["vr_comfort_score"] for m in all_metrics[cls]]
        ax.scatter(xs, ys, label=cls, color=palette[i % len(palette)],
                   s=100, alpha=0.85, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("Render Cost (k-triangles)")
    ax.set_ylabel("VR Comfort Score")
    ax.set_title("VR Plot 6 — Render Cost vs VR Comfort Score\n"
                 "(ideal: bottom-right = low cost, high comfort)",
                 fontweight="bold", pad=10)
    ax.legend(); ax.grid(True); plt.tight_layout()
    p = os.path.join(save_dir, "vrplot6_cost_vs_comfort.png")
    fig.savefig(p, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig); _ok(f"Saved: {p}")

    _ok("All VR metric plots saved.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save depth visualisations
# ══════════════════════════════════════════════════════════════════════════════
def save_depth_visual(img_np, depth, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                             facecolor="#0d0d1a")
    for ax in axes:
        ax.set_facecolor("#0d0d1a"); ax.axis("off")

    axes[0].imshow(img_np)
    axes[0].set_title("Original GAN Image", color="#c8c8e8", pad=6)

    axes[1].imshow(depth, cmap="plasma")
    axes[1].set_title("Depth Map (MiDaS / Fallback)", color="#c8c8e8", pad=6)

    # Pseudo-3D surface
    H, W = depth.shape
    ds   = cv2.resize(depth, (64, 64))
    xx, yy = np.meshgrid(np.linspace(0,1,64), np.linspace(0,1,64))
    ax3d = fig.add_subplot(1, 3, 3, projection="3d")
    ax3d.set_facecolor("#0d0d1a")
    ax3d.plot_surface(xx, yy, ds, cmap="plasma", linewidth=0, antialiased=True)
    ax3d.set_title("3D Surface Preview", color="#c8c8e8", pad=6)
    ax3d.tick_params(colors="#7777aa")

    plt.tight_layout(pad=2)
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_vr_pipeline():
    _hdr("  Craft Workshop VR Integration Pipeline  ")

    for d in [VR_OUTPUT_DIR, MESH_DIR, SCENE_DIR, METRICS_DIR, FINAL_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Step 1: collect images ────────────────────────────────────────────
    catalog = collect_gan_images()

    # ── Steps 2-5: per-image processing ──────────────────────────────────
    _sec("Steps 2–5 — Depth Estimation · Mesh Generation · Metrics")
    all_metrics   = {}
    scene_entries = []

    for cls, img_paths in tqdm(catalog.items(), desc="Classes"):
        cls_mesh_dir  = os.path.join(MESH_DIR, cls)
        cls_depth_dir = os.path.join(VR_OUTPUT_DIR, "depth_visuals", cls)
        os.makedirs(cls_mesh_dir,  exist_ok=True)
        os.makedirs(cls_depth_dir, exist_ok=True)
        all_metrics[cls] = []

        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]

            # Load image
            img_bgr = cv2.imread(ip)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_256 = cv2.resize(img_rgb, (256, 256))

            # Depth estimation
            depth   = estimate_depth(img_256)

            # Depth visual
            vis_path = os.path.join(cls_depth_dir, f"{stem}_depth.png")
            save_depth_visual(img_256, depth, vis_path)

            # OBJ mesh
            obj_str  = depth_to_obj(depth, f"{stem}.png")
            obj_path = os.path.join(cls_mesh_dir, f"{stem}.obj")
            with open(obj_path, "w") as f:
                f.write(obj_str)
            mtl_path = os.path.join(cls_mesh_dir, f"{stem}.mtl")
            shutil.copy(ip, os.path.join(cls_mesh_dir, f"{stem}.png"))
            write_mtl(mtl_path, f"{stem}.png")

            # Metrics
            m = compute_vr_metrics(depth, img_256)
            all_metrics[cls].append(m)

            # Average colour for Three.js tint
            avg_col = img_256.mean(axis=(0,1)).astype(int)
            hex_col = "#{:02x}{:02x}{:02x}".format(*avg_col)

            scene_entries.append(dict(
                class_name = cls,
                depth      = depth,
                avg_color  = hex_col,
                obj_path   = obj_path,
            ))

        _ok(f"{cls}: {len(img_paths)} images processed · "
            f"avg comfort {np.mean([m['vr_comfort_score'] for m in all_metrics[cls]]):.1f}")

    # ── Step 4: VR HTML scene ─────────────────────────────────────────────
    _sec("Step 4 — Building Three.js WebXR VR Scene")
    scene_html = os.path.join(SCENE_DIR, "craft_vr_scene.html")
    build_vr_html(scene_entries, scene_html)

    # ── Step 6: metric plots ──────────────────────────────────────────────
    _sec("Step 6 — Generating VR Metric Plots")
    plot_vr_metrics(all_metrics, METRICS_DIR)

    # ── Step 7: copy deliverables to FINAL ────────────────────────────────
    _sec("Step 7 — Assembling Final Output Folder")
    # VR scene
    shutil.copy(scene_html, os.path.join(FINAL_DIR, "craft_vr_scene.html"))
    # metric plots
    for f in glob.glob(os.path.join(METRICS_DIR, "vrplot*.png")):
        shutil.copy(f, FINAL_DIR)
    # OBJ meshes (zip each class folder)
    for cls in catalog:
        src = os.path.join(MESH_DIR, cls)
        shutil.make_archive(os.path.join(FINAL_DIR, f"meshes_{cls}"),
                            "zip", src)
        _ok(f"meshes_{cls}.zip → final/")

    # Save summary JSON
    summary = {}
    for cls, mlist in all_metrics.items():
        summary[cls] = {
            k: round(float(np.mean([m[k] for m in mlist])), 4)
            for k in mlist[0]
        }
    with open(os.path.join(FINAL_DIR, "vr_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _ok("vr_metrics_summary.json → final/")

    # ── Final summary ─────────────────────────────────────────────────────
    _sec("VR Pipeline Complete — Summary")
    _kv("Total classes processed",  str(len(catalog)),             GREEN)
    _kv("Total images processed",   str(len(scene_entries)),       CYAN)
    _kv("OBJ meshes generated",     str(len(scene_entries)),       WHITE)
    _kv("VR metric plots saved",    "6 plots (vrplot1…6)",         YELLOW)
    _kv("Three.js VR scene",        "vr_output/scenes/craft_vr_scene.html", CYAN)
    _kv("Final deliverables",       "vr_output/final/",            GREEN)
    for cls, mlist in all_metrics.items():
        avg_comfort = np.mean([m["vr_comfort_score"] for m in mlist])
        avg_fps     = np.mean([m["est_fps"]          for m in mlist])
        col = GREEN if avg_comfort > 70 else YELLOW
        _kv(f"  {cls}", f"comfort={avg_comfort:.1f}  est_fps={avg_fps:.1f}", col)
    print(f"\n  {BOLD}{GREEN}VR integration complete → vr_output/final/{RESET}\n")




if __name__ == "__main__":
    run_vr_pipeline()