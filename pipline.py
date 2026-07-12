from google.colab import drive
drive.mount('/content/drive')

!pip install xFormers
!pip install -e /content/drive/MyDrive/data1/NVS_repos/gaussian-splatting/submodules/simple-knn
!pip install colorama einops jaxtyping laspy numpy pillow plyfile scipy wandb pymeshlab meshroom
%cd /content/drive/MyDrive/data1/NVS_repos/gaussian-splatting
!pip install -e submodules/diff-gaussian-rasterization

#!/usr/bin/env python3
"""
================================================================================
 Novel View Synthesis pipeline - BATCH nhieu scene (chay tren Google Colab)
 COLMAP -> Depth Anything V2 -> VANILLA 3D GAUSSIAN SPLATTING (graphdeco-inria)
 -> GAUSSIAN PRUNING -> Exposure Compensation + LPIPS Fine-tuning
 -> QUANTIZATION -> Render test -> (public) Metrics PSNR/SSIM/LPIPS

 ***** BAN NAY DUNG REPO GOC CHUAN: https://github.com/graphdeco-inria/gaussian-splatting *****
 ***** TRAIN CO DINH 1440 ITERATIONS + PIPELINE NEN MODEL: PRUNE -> FINETUNE -> QUANTIZE   *****

 ***** BAN CAP NHAT NAY THEM: *****
   - KIEM TRA DRIVE (output_root) TRUOC KHI XU LY MOI SPLIT: quet tung scene,
     neu da co du finetune + render luu san (folder da train xong) thi BAO CAO
     RO RANG va BO QUA (khong train/finetune/render lai).
   - SUA LOI QUAN TRONG o vong lap main(): truoc day khi bat --transfer_learning,
     trong so GlobalColorMLP KHONG duoc truyen (thread) qua cac scene ke tiep
     trong cung 1 split (process_scene() duoc goi thieu prev_color_mlp_path /
     is_first_in_chain). Ban nay sua lai: scene sau trong CUNG split se TU DONG
     NAP LAI trong so GlobalColorMLP da luu tren Drive tu scene truoc do (hoac
     tu scene gan nhat da xong neu resume giua chung do Colab bi ngat) va fine-
     tune tiep voi it iteration hon (--transfer_finetune_iters), dung nhu mo ta
     trong docstring cua process_scene().

 VI SAO DOI SANG graphdeco-inria/gaussian-splatting (thay vi Scaffold-GS):
   Day la repo THAM CHIEU GOC (official reference implementation) cua paper
   "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al.
   2023). Dung repo nay la "form chuan" ma hau het cac paper/benchmark 3DGS
   dung lam baseline. Cau truc GaussianModel don gian hon Scaffold-GS rat
   nhieu: MOI Gaussian la 1 diem doc lap voi cac thuoc tinh
       _xyz (vi tri), _features_dc + _features_rest (mau, spherical harmonics),
       _opacity (do mo), _scaling, _rotation
   KHONG co anchor/offset/MLP nhu Scaffold-GS. Vi vay:
     - Pruning don gian & chinh xac hon: dung truc tiep gaussians.get_opacity
       (khong can render qua nhieu view de uoc luong nhu Scaffold-GS).
     - KHONG CON "transfer learning trong so mang giua cac scene" nhu ban
       Scaffold-GS truoc, vi vanilla 3DGS khong co module mang dung chung giua
       cac scene (moi scene la 1 tap diem rieng biet, khong co MLP de transfer
       tren chinh cac Gaussian). De van co transfer learning y nghia, pipeline
       dung 1 module nho DOC LAP VOI SO DIEM/SO ANH: GlobalColorMLP (xem BUOC 4).

 CAC BUOC CHINH:
   1) TRAIN CO DINH 1440 ITERATIONS (--iterations, mac dinh 1440) cho MOI scene.
      Cac tham so densification (densify_from_iter, densify_until_iter,
      densification_interval, opacity_reset_interval) duoc scale ti le theo
      so iterations nay (ban goc repo dung cho 30000 iterations).
   2) GAUSSIAN PRUNING: sau train, truoc fine-tune, loai bo cac Gaussian co
      opacity (get_opacity) thap - dong gop rat it vao anh render.
   3) FINE-TUNE: Exposure Compensation (bu sang) + LPIPS perceptual loss,
      toi uu lai _features_dc/_features_rest/_opacity/_scaling sau khi prune.
      Neu bat --transfer_learning: fine-tune dong thoi GlobalColorMLP (nap lai
      trong so tu scene truoc trong cung split neu co).
   4) QUANTIZATION: nen model - FP16 cho hinh hoc (_xyz/_scaling/_rotation),
      INT8 cho mau (_features_dc/_features_rest), luu ban nen that su ra Drive
      + ap dung fake-quantization (ep xuong roi ep nguoc ve fp32) de danh gia
      chat luong render SAU KHI NEN.
   5) RENDER test_poses.csv + (public) tinh PSNR/SSIM/LPIPS.
   6) RESUME (KIEM TRA DRIVE): scene nao da co finetune+render luu san tren
      Drive se duoc bo qua train lai, chi doc lai ket qua (huu ich khi Colab
      bi ngat giua chung). Neu bat --transfer_learning, trong so GlobalColorMLP
      cua scene da xong cung duoc doc lai de TIEP TUC CHUOI TRANSFER cho cac
      scene con lai chua xong, thay vi khoi tao lai tu dau.
================================================================================

DU LIEU KY VONG (tren Google Drive) - CO 2 SPLIT: public va private
----------------------------------------------------------------------
NVS_data/
├── public_set/
│   ├── HCM0001/
│   │   ├── train/{images/, sparse/0/ (hoac sparse/ chua truc tiep *.bin)}
│   │   └── test/
│   │       ├── test_poses.csv
│   │       └── images/            <- ANH GROUND TRUTH (dung de tinh metrics)
│   └── ...
└── private_set1/
    ├── HCM0100/
    │   ├── train/...
    │   └── test/
    │       └── test_poses.csv     <- KHONG co ground truth (dung de nop bai)
    └── ...

CACH DUNG TREN COLAB
---------------------
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/drive/MyDrive/data/run_pipeline_gpu.py \
        --data_root "/content/drive/MyDrive/data/phase1" \
        --output_root "/content/drive/MyDrive/data/NVS_output"

   Chinh so iterations (mac dinh 1440):
    !python run_pipeline_gpu.py --iterations 1440

   Bat transfer learning (GlobalColorMLP dung chung, truyen tiep giua cac scene
   trong cung split, TU DONG RESUME dung tu scene da xong gan nhat tren Drive):
    !python run_pipeline_gpu.py --transfer_learning

   Tat pruning / quantization neu khong muon nen model:
    !python run_pipeline_gpu.py --no_prune --no_quantize

   Tat mixed precision:
    !python run_pipeline_gpu.py --no_amp

   Neu Colab bi ngat/disconnect giua chung, CHI CAN CHAY LAI Y HET LENH CU:
   pipeline se TU DONG QUET DRIVE, bo qua cac scene da train+finetune+render
   xong, va (neu bat --transfer_learning) tu dong nap lai GlobalColorMLP cua
   scene xong gan nhat de tiep tuc chuoi transfer cho cac scene con lai.
================================================================================
"""

import os
import sys
import glob
import math
import json
import random
import struct
import argparse
import subprocess
import collections
from contextlib import nullcontext

import numpy as np


# =============================================================================
# CONFIG - CHINH O DAY (hoac truyen qua --data_root / --output_root)
# =============================================================================

DEFAULT_DRIVE_DATA_ROOT = "/content/drive/MyDrive/data/phase1"       # chua public_set/ va private_set1/
DEFAULT_DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/data1/NVS_output"   # noi luu ket qua

REPO_ROOT = "/content/drive/MyDrive/data1/NVS_repos"
GS_REPO_URL = "https://github.com/graphdeco-inria/gaussian-splatting.git"
GS_REPO_DIR = f"{REPO_ROOT}/gaussian-splatting"

SPLITS = ("public_set", "private_set1")

# Depth Anything V2
DA_ENCODER = "vitl"
DA_CKPT_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
MAX_NEW_POINTS_PER_IMAGE = 500
DENSIFY_OUTLIER_PERCENTILE = 95
DEFAULT_DEPTH_BATCH_SIZE = 16         # so anh xu ly song song moi lan forward Depth Anything V2

# ====== VANILLA 3DGS TRAINING - CO DINH SO ITERATIONS ======
SH_DEGREE = 3                          # bac spherical harmonics mac dinh cua repo goc
FIXED_TRAIN_ITERATIONS = 1440           # <-- TRAIN CO DINH 1440 ITERATIONS CHO MOI SCENE

# Fine-tune (Exposure Compensation + LPIPS) - theo iteration (batch hieu dung)
FINETUNE_ITERS = 1000
LPIPS_WEIGHT_START = 0.02
LPIPS_WEIGHT_END = 0.20
LAMBDA_DSSIM = 0.2
DEFAULT_FINETUNE_BATCH_SIZE = 4        # so camera / buoc toi uu (tich luy gradient)
DEFAULT_METRICS_BATCH_SIZE = 8         # so cap anh moi lan goi LPIPS khi tinh metrics

# ====== GAUSSIAN PRUNING (sau train, truoc fine-tune) ======
DEFAULT_PRUNE_OPACITY_THRESHOLD = 0.01    # Gaussian co opacity (get_opacity, da sigmoid) < nguong nay bi loai
DEFAULT_PRUNE_MIN_KEEP_RATIO = 0.3        # luon giu it nhat 30% so Gaussian (an toan, tranh prune qua tay)

# ====== QUANTIZATION (sau fine-tune, truoc render) ======
QUANT_GEOM_FP16 = True    # ep tensor hinh hoc (_xyz/_scaling/_rotation/_opacity) ve FP16
QUANT_COLOR_INT8 = True   # quant tensor mau (_features_dc/_features_rest, spherical harmonics) ve INT8

# Cac tensor "cot loi" cua GaussianModel goc (graphdeco-inria) - dung cho pruning + quantization
GEOM_TENSOR_NAMES = ["_xyz", "_scaling", "_rotation", "_opacity"]
COLOR_TENSOR_NAMES = ["_features_dc", "_features_rest"]

# ====== TRANSFER LEARNING (module MAU DUNG CHUNG giua cac scene) ======
# Vanilla 3DGS (graphdeco-inria) KHONG co MLP/anchor dung chung giua cac scene
# nhu Scaffold-GS (moi scene la 1 tap diem 3D doc lap: _xyz, _features_dc...).
# De van co "transfer learning" y nghia, pipeline them 1 MODULE NHO, DOC LAP VOI
# SO DIEM/SO ANH cua tung scene: GlobalColorMLP - 1 mang nho hieu chinh mau
# RENDER RA (gia dinh cac scene trong cung dataset chup cung thiet bi/dieu kien
# anh sang nen co "gu mau" tuong tu nhau). Day la module DUY NHAT co the luu va
# nap lai giua cac scene khac nhau.
GLOBAL_COLOR_MLP_HIDDEN = 32                # so unit an cua GlobalColorMLP
TRANSFER_FINETUNE_ITERS = 1200               # so iteration fine-tune cho scene TRANSFER (it hon scene dau)


def run(cmd, check=True):
    """Chay lenh va stream output real-time (khong capture stdout -> tren Colab
    output cua train.py nhu tqdm, log se hien ngay lap tuc)."""
    print(f"\n>>> {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=check)


# =============================================================================
# TOI UU GPU
# =============================================================================

_GPU_CONFIGURED = False


def configure_gpu_for_max_utilization():
    global _GPU_CONFIGURED
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        print("[GPU] Khong tim thay GPU - pipeline se chay tren CPU (rat cham).")
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not _GPU_CONFIGURED:
        props = torch.cuda.get_device_properties(0)
        print(f"[GPU] {torch.cuda.get_device_name(0)} | VRAM tong = {props.total_memory / 1e9:.1f} GB "
              f"| TF32=on | cudnn.benchmark=on")
        _GPU_CONFIGURED = True


def amp_autocast_ctx(enabled):
    """Tra ve context manager autocast(fp16) neu enabled=True, nguoc lai la no-op."""
    import torch
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# =============================================================================
# ITERATION HELPERS - MILESTONE + SCALE DENSIFY PARAMS (theo so iterations co dinh)
# =============================================================================

def count_train_images(local_root):
    train_imgs = sorted(glob.glob(f"{local_root}/train/images/*"))
    n = len(train_imgs)
    if n == 0:
        raise RuntimeError(f"Khong tim thay anh train tai {local_root}/train/images/")
    return n


def calc_iteration_milestones(train_iterations):
    """Tra ve danh sach iteration milestone de test/save (vd: [360, 720, 1440])."""
    milestones = set()
    milestones.add(train_iterations)
    if train_iterations >= 4:
        milestones.add(train_iterations // 2)
    if train_iterations >= 10:
        milestones.add(train_iterations // 4)
    return sorted(m for m in milestones if m > 0)


def calc_densify_params(train_iterations):
    """Scale cac tham so densification cua VANILLA 3DGS ti le voi total iterations.
    Ban goc repo (mac dinh cho 30000 iterations) dung:
        densify_from_iter = 500
        densify_until_iter = 15000   (~50% tong iterations)
        densification_interval = 100
        opacity_reset_interval = 3000  (~10% tong iterations)
    Voi train_iterations nho (vd 1440), scale xuong tuong ung.
    """
    densify_from_iter = min(500, max(20, train_iterations // 10))
    densify_until_iter = max(densify_from_iter + 1, int(train_iterations * 0.5))
    densification_interval = min(100, max(5, train_iterations // 30))
    opacity_reset_interval = min(3000, max(50, train_iterations // 10))
    return densify_from_iter, densify_until_iter, densification_interval, opacity_reset_interval


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", default=DEFAULT_DRIVE_DATA_ROOT,
                   help="Thu muc goc chua 2 split con: public_set/ va private_set1/")
    p.add_argument("--output_root", default=DEFAULT_DRIVE_OUTPUT_ROOT)

    p.add_argument("--splits", default=",".join(SPLITS),
                   help="Danh sach split can xu ly, cach nhau boi dau phay.")

    p.add_argument("--scenes", default=None,
                   help="Danh sach scene cach nhau boi dau phay, vd: HCM0001,HCM0002.")

    p.add_argument("--skip_env_setup", action="store_true",
                   help="Bo qua buoc cai dat/clone repo.")

    p.add_argument("--skip_depth", action="store_true",
                   help="Bo qua Depth Anything V2 + densify.")

    # ===== TRAIN CO DINH SO ITERATIONS =====
    p.add_argument("--iterations", type=int, default=FIXED_TRAIN_ITERATIONS,
                   help=f"So iterations CO DINH de train 3DGS cho MOI scene "
                        f"(mac dinh: {FIXED_TRAIN_ITERATIONS}).")
    p.add_argument("--sh_degree", type=int, default=SH_DEGREE,
                   help=f"Bac spherical harmonics (mac dinh: {SH_DEGREE}, giong repo goc).")

    p.add_argument("--finetune_iters", type=int, default=FINETUNE_ITERS,
                   help="So iteration fine-tune LPIPS (batch hieu dung) cho scene DAU chuoi "
                        "(hoac moi scene neu khong bat --transfer_learning).")

    # ===== TRANSFER LEARNING (qua GlobalColorMLP dung chung) =====
    p.add_argument("--transfer_learning", action="store_true",
                   help="Bat transfer learning: scene dau chuoi train GlobalColorMLP (module mau "
                        "dung chung, DOC LAP voi so diem/so anh) tu dau; cac scene sau trong CUNG "
                        "1 split se NAP LAI trong so MLP nay va fine-tune tiep voi it iteration hon "
                        "(--transfer_finetune_iters). Neu resume tu Drive (mot so scene da xong san), "
                        "pipeline se TU DONG doc lai GlobalColorMLP cua scene xong gan nhat de tiep tuc "
                        "chuoi transfer cho cac scene con lai, khong khoi tao lai tu dau. Day la module "
                        "DUY NHAT co the transfer giua cac scene vi vanilla 3DGS khong co MLP/anchor "
                        "dung chung nhu Scaffold-GS.")
    p.add_argument("--transfer_finetune_iters", type=int, default=TRANSFER_FINETUNE_ITERS,
                   help=f"So iteration fine-tune cho cac scene TRANSFER (sau scene dau chuoi), "
                        f"mac dinh {TRANSFER_FINETUNE_ITERS} (it hon --finetune_iters vi MLP da "
                        f"co san trong so khoi tao tot tu scene truoc).")

    p.add_argument("--continue_on_error", action="store_true", default=True)

    p.add_argument("--force_retrain", action="store_true",
                   help="Bat co nay de BO QUA logic resume: luon train lai tat ca scene (khong quet Drive).")

    # ===== GAUSSIAN PRUNING =====
    p.add_argument("--prune", dest="prune", action="store_true", default=True,
                   help="Bat Gaussian Pruning sau train, truoc fine-tune (mac dinh: BAT).")
    p.add_argument("--no_prune", dest="prune", action="store_false",
                   help="Tat Gaussian Pruning.")
    p.add_argument("--prune_opacity_threshold", type=float, default=DEFAULT_PRUNE_OPACITY_THRESHOLD,
                   help=f"Nguong opacity (get_opacity, 0..1) de loai Gaussian "
                        f"(mac dinh: {DEFAULT_PRUNE_OPACITY_THRESHOLD}).")
    p.add_argument("--prune_min_keep_ratio", type=float, default=DEFAULT_PRUNE_MIN_KEEP_RATIO,
                   help=f"Ti le toi thieu Gaussian phai giu lai (mac dinh: {DEFAULT_PRUNE_MIN_KEEP_RATIO}).")

    # ===== QUANTIZATION =====
    p.add_argument("--quantize", dest="quantize", action="store_true", default=True,
                   help="Bat Quantization sau fine-tune, truoc render (mac dinh: BAT).")
    p.add_argument("--no_quantize", dest="quantize", action="store_false",
                   help="Tat Quantization.")

    # ===== TOI UU GPU =====
    p.add_argument("--amp", dest="amp", action="store_true", default=True,
                   help="Bat mixed precision (fp16 autocast). Mac dinh: BAT.")
    p.add_argument("--no_amp", dest="amp", action="store_false",
                   help="Tat mixed precision, chay fp32 thuan.")
    p.add_argument("--depth_batch_size", type=int, default=DEFAULT_DEPTH_BATCH_SIZE)
    p.add_argument("--finetune_batch_size", type=int, default=DEFAULT_FINETUNE_BATCH_SIZE)
    p.add_argument("--metrics_batch_size", type=int, default=DEFAULT_METRICS_BATCH_SIZE)

    args, _ = p.parse_known_args()
    return args


def discover_scenes(data_root, splits):
    result = {}
    for split in splits:
        split_dir = f"{data_root}/{split}"
        scenes = []
        if not os.path.isdir(split_dir):
            print(f"[Bo qua split] Khong tim thay thu muc: {split_dir}")
            result[split] = scenes
            continue
        for name in sorted(os.listdir(split_dir)):
            scene_dir = f"{split_dir}/{name}"
            if not os.path.isdir(scene_dir):
                continue
            has_train_images = os.path.isdir(f"{scene_dir}/train/images")
            has_test_csv = os.path.isfile(f"{scene_dir}/test/test_poses.csv")
            if has_train_images and has_test_csv:
                scenes.append(name)
            else:
                print(f"[Bo qua] {split}/{name}: thieu train/images hoac test/test_poses.csv")
        result[split] = scenes
    return result


def find_scene_split(data_root, splits, scene_name):
    for split in splits:
        if os.path.isdir(f"{data_root}/{split}/{scene_name}"):
            return split
    return None


# =============================================================================
# BUOC 0: MOI TRUONG - CLONE REPO GOC graphdeco-inria/gaussian-splatting
# =============================================================================

def setup_environment():
    print("\n========== BUOC 0: Cai dat moi truong ==========")
    run("apt-get -qq update")
    run("apt-get -qq install -y colmap ffmpeg > /dev/null")
    run("pip -q install plyfile tqdm opencv-python-headless lpips scikit-image "
        "pandas scipy imageio imageio-ffmpeg einops timm")

    if not os.path.isdir(GS_REPO_DIR):
        run(f"git clone --recursive {GS_REPO_URL} {GS_REPO_DIR}")
    else:
        print("gaussian-splatting (graphdeco-inria) da co, bo qua clone.")

    depth_repo = f"{REPO_ROOT}/Depth-Anything-V2"
    if os.path.exists(depth_repo):
        print("Depth-Anything-V2 da co, bo qua clone.")
    else:
        run(f"git clone https://github.com/DepthAnything/Depth-Anything-V2.git {depth_repo}")

    # Cai dat cac submodule rasterizer + simple-knn CUA REPO GOC (khong swap Mip-Splatting,
    # dung dung "form chuan" cua graphdeco-inria/gaussian-splatting).
    marker = f"{GS_REPO_DIR}/submodules/.installed"
    if not os.path.isfile(marker):
        run(f'pip -q install "{GS_REPO_DIR}/submodules/diff-gaussian-rasterization" --no-build-isolation')
        run(f'pip -q install "{GS_REPO_DIR}/submodules/simple-knn" --no-build-isolation')
        run(f"touch {marker}")
    print("Da san sang: graphdeco-inria/gaussian-splatting (rasterizer + simple-knn) + Depth-Anything-V2.")


# =============================================================================
# BUOC 1: DU LIEU + COLMAP
# =============================================================================

def prepare_data(data_dir, local_root):
    print("\n========== BUOC 1: Chuan bi du lieu + kiem tra COLMAP ==========")
    assert os.path.isdir(f"{data_dir}/train/images"), f"Khong thay {data_dir}/train/images tren Drive"
    assert os.path.isfile(f"{data_dir}/test/test_poses.csv"), f"Khong thay {data_dir}/test/test_poses.csv tren Drive"

    if not os.path.isdir(f"{local_root}/train"):
        run(f"mkdir -p {local_root}")
        run(f'cp -r "{data_dir}/train" "{local_root}/train"')
        run(f'cp -r "{data_dir}/test" "{local_root}/test"')
    else:
        print("Du lieu local da ton tai, bo qua copy.")

    needed = ["cameras.bin", "images.bin", "points3D.bin"]
    sparse_root = f"{local_root}/train/sparse"
    train_sparse = f"{sparse_root}/0"
    if not all(os.path.isfile(f"{train_sparse}/{f}") for f in needed):
        if all(os.path.isfile(f"{sparse_root}/{f}") for f in needed):
            print("Phat hien sparse/*.bin nam truc tiep trong sparse/, dang chuan hoa thanh sparse/0/ ...")
            run(f'mkdir -p "{sparse_root}/0"')
            for f in needed:
                run(f'mv "{sparse_root}/{f}" "{sparse_root}/0/{f}"')

    have_all = all(os.path.isfile(f"{train_sparse}/{f}") for f in needed)
    if have_all:
        print("COLMAP sparse reconstruction da co san, bo qua chay COLMAP.")
    else:
        print("Thieu sparse reconstruction, dang chay COLMAP tu dau...")
        run_colmap(f"{local_root}/train/images", f"{local_root}/train/colmap_ws")
        sparse_out = f"{local_root}/train/colmap_ws/sparse/0"
        run(f"rm -rf {local_root}/train/sparse")
        run(f"mkdir -p {local_root}/train/sparse")
        run(f"cp -r {sparse_out} {local_root}/train/sparse/0")


def run_colmap(images_dir, workspace_dir):
    os.makedirs(workspace_dir, exist_ok=True)
    db_path = f"{workspace_dir}/database.db"
    sparse_dir = f"{workspace_dir}/sparse"
    os.makedirs(sparse_dir, exist_ok=True)
    run(f"colmap feature_extractor --database_path {db_path} --image_path {images_dir} "
        f"--ImageReader.single_camera 1 --SiftExtraction.use_gpu 1 "
        f"--SiftExtraction.num_threads -1")
    run(f"colmap exhaustive_matcher --database_path {db_path} --SiftMatching.use_gpu 1")
    run(f"colmap mapper --database_path {db_path} --image_path {images_dir} --output_path {sparse_dir} "
        f"--Mapper.num_threads -1")
    return sparse_dir


# =============================================================================
# BUOC 2: DEPTH ANYTHING V2 + DENSIFY POINT CLOUD
# =============================================================================

CameraModel = collections.namedtuple("CameraModel", ["id", "model", "width", "height", "params"])
ImageModel = collections.namedtuple("ImageModel", ["id", "qvec", "tvec", "camera_id", "name"])


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def read_cameras_binary(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12, 6: 5, 7: 4, 8: 5, 9: 12, 10: 5}.get(model_id, 4)
            params = struct.unpack("<" + "d" * n_params, f.read(8 * n_params))
            cams[cid] = CameraModel(cid, model_id, w, h, np.array(params))
    return cams


def read_images_binary(path):
    imgs = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n_pts)
            imgs[iid] = ImageModel(iid, np.array(qvec), np.array(tvec), cam_id, name.decode("utf-8"))
    return imgs


def read_points3D_binary(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)
            x, y, z = struct.unpack("<ddd", f.read(24))
            r, g, b = struct.unpack("<BBB", f.read(3))
            f.read(8)
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)
            xyz.append([x, y, z])
            rgb.append([r, g, b])
    return np.array(xyz), np.array(rgb)


def get_intrinsics(cam):
    if cam.model in (1, 4):  # PINHOLE / OPENCV
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
    else:  # SIMPLE_PINHOLE / SIMPLE_RADIAL
        fx = fy = cam.params[0]
        cx, cy = cam.params[1], cam.params[2]
    return fx, fy, cx, cy


def write_ply(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(len(xyz), dtype=dtype)
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['red'], elements['green'], elements['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def _run_depth_batch(da_model, raw_imgs, use_amp):
    import torch
    depths = []
    with torch.no_grad(), amp_autocast_ctx(use_amp):
        for raw_img in raw_imgs:
            depth = da_model.infer_image(raw_img)
            depths.append(depth.astype(np.float32))
    return depths


def run_depth_anything_and_densify(local_root, out_dir, skip_depth=False, depth_batch_size=DEFAULT_DEPTH_BATCH_SIZE,
                                    use_amp=True):
    print("\n========== BUOC 2: Depth Anything V2 + densify point cloud (batch + AMP) ==========")
    configure_gpu_for_max_utilization()
    sparse_dir = f"{local_root}/train/sparse/0"
    densified_ply = f"{sparse_dir}/points3D_densified.ply"

    if skip_depth:
        print("Bo qua Depth Anything V2 theo yeu cau (--skip_depth). Dung point cloud COLMAP goc.")
        return None

    if os.path.isfile(densified_ply):
        print("Point cloud densify da ton tai, bo qua buoc nay.")
        return densified_ply

    import cv2
    import torch
    from tqdm import tqdm

    sys.path.append(f"{REPO_ROOT}/Depth-Anything-V2")
    from depth_anything_v2.dpt import DepthAnythingV2

    ckpt_dir = f"{out_dir}/da_v2_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/depth_anything_v2_{DA_ENCODER}.pth"
    if not os.path.isfile(ckpt_path):
        run(f'wget -q -O "{ckpt_path}" {DA_CKPT_URL}')

    model_cfg = {'encoder': DA_ENCODER, 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    da_model = DepthAnythingV2(**model_cfg)
    da_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    da_model = da_model.to('cuda').eval()

    train_imgs = sorted(glob.glob(f"{local_root}/train/images/*"))
    depth_dir = f"{local_root}/train/depth"
    os.makedirs(depth_dir, exist_ok=True)

    pending_paths, pending_imgs = [], []

    def flush_batch():
        if not pending_paths:
            return
        depths = _run_depth_batch(da_model, pending_imgs, use_amp)
        for path, depth in zip(pending_paths, depths):
            name = os.path.splitext(os.path.basename(path))[0]
            np.save(f"{depth_dir}/{name}.npy", depth)
        pending_paths.clear()
        pending_imgs.clear()

    todo_imgs = [p for p in train_imgs if not os.path.isfile(f"{depth_dir}/{os.path.splitext(os.path.basename(p))[0]}.npy")]
    for img_path in tqdm(todo_imgs, desc=f"Depth Anything V2 inference (batch={depth_batch_size}, amp={use_amp})"):
        raw_img = cv2.imread(img_path)
        pending_paths.append(img_path)
        pending_imgs.append(raw_img)
        if len(pending_paths) >= depth_batch_size:
            flush_batch()
    flush_batch()

    del da_model
    torch.cuda.empty_cache()

    run(f'mkdir -p "{out_dir}/depth" && cp -r "{depth_dir}" "{out_dir}/depth/"')

    cams = read_cameras_binary(f"{sparse_dir}/cameras.bin")
    imgs = read_images_binary(f"{sparse_dir}/images.bin")
    xyz_colmap, rgb_colmap = read_points3D_binary(f"{sparse_dir}/points3D.bin")
    print(f"COLMAP: {len(cams)} cameras, {len(imgs)} images, {len(xyz_colmap)} sparse points")

    name_to_img = {im.name: im for im in imgs.values()}
    extra_xyz, extra_rgb = [], []

    for img_path in train_imgs:
        name = os.path.basename(img_path)
        if name not in name_to_img:
            continue
        im = name_to_img[name]
        cam = cams[im.camera_id]
        fx, fy, cx, cy = get_intrinsics(cam)
        R = qvec2rotmat(im.qvec)
        t = im.tvec.reshape(3, 1)

        cam_pts = (R @ xyz_colmap.T + t).T
        z = cam_pts[:, 2]
        valid = z > 1e-3
        if valid.sum() < 20:
            continue
        u = (fx * cam_pts[valid, 0] / z[valid] + cx).astype(np.int32)
        v = (fy * cam_pts[valid, 1] / z[valid] + cy).astype(np.int32)
        gt_depth = z[valid]

        depth_path = f"{depth_dir}/{os.path.splitext(name)[0]}.npy"
        rel_depth = np.load(depth_path)
        H, W = rel_depth.shape
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if inb.sum() < 20:
            continue
        u, v, gt_depth = u[inb], v[inb], gt_depth[inb]
        pred_rel = rel_depth[v, u]

        eps = 1e-6
        A = np.stack([1.0 / (pred_rel + eps), np.ones_like(pred_rel)], axis=1)
        sol, *_ = np.linalg.lstsq(A, gt_depth, rcond=None)
        a, b = sol
        metric_depth_map = np.clip(a / (rel_depth + eps) + b, 1e-3, None)

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ys = np.random.randint(0, H, MAX_NEW_POINTS_PER_IMAGE)
        xs = np.random.randint(0, W, MAX_NEW_POINTS_PER_IMAGE)
        d = metric_depth_map[ys, xs]
        x_cam = (xs - cx) * d / fx
        y_cam = (ys - cy) * d / fy
        pts_cam = np.stack([x_cam, y_cam, d], axis=1)
        pts_world = (R.T @ (pts_cam.T - t)).T
        colors = img_rgb[ys, xs]

        extra_xyz.append(pts_world)
        extra_rgb.append(colors)

    if len(extra_xyz) > 0:
        extra_xyz = np.concatenate(extra_xyz, axis=0)
        extra_rgb = np.concatenate(extra_rgb, axis=0)
        med = np.median(xyz_colmap, axis=0)
        dist = np.linalg.norm(extra_xyz - med, axis=1)
        thr = np.percentile(dist, DENSIFY_OUTLIER_PERCENTILE)
        keep = dist < thr
        extra_xyz, extra_rgb = extra_xyz[keep], extra_rgb[keep]
    else:
        extra_xyz = np.zeros((0, 3))
        extra_rgb = np.zeros((0, 3))

    merged_xyz = np.concatenate([xyz_colmap, extra_xyz], axis=0)
    merged_rgb = np.concatenate([rgb_colmap, extra_rgb], axis=0).astype(np.uint8)
    print(f"Point cloud sau merge: {len(merged_xyz)} diem (goc {len(xyz_colmap)} + densify {len(extra_xyz)})")

    write_ply(densified_ply, merged_xyz, merged_rgb)
    run(f'cp "{densified_ply}" "{out_dir}/depth/points3D_densified.ply"')
    return densified_ply


# =============================================================================
# BUOC 3: TRAIN VANILLA 3D GAUSSIAN SPLATTING (graphdeco-inria) - CO DINH ITERATIONS
# =============================================================================

def train_gaussian_splatting(local_root, out_dir, densified_ply, train_iterations):
    """Train vanilla 3DGS (repo goc graphdeco-inria/gaussian-splatting) cho 1 scene,
    voi SO ITERATIONS CO DINH. Moi scene train DOC LAP tu dau (repo goc khong co
    module mang dung chung giua cac scene nen KHONG co "transfer learning" o
    muc Gaussian; transfer learning duoc thuc hien rieng qua GlobalColorMLP o BUOC 4).

    Returns:
        (model_path_local, train_iterations)
    """
    print("\n========== BUOC 3: Train 3D Gaussian Splatting (graphdeco-inria, repo goc) ==========")
    model_path_local = f"{local_root}/gs_model"
    sparse0 = f"{local_root}/train/sparse/0"

    if os.path.isdir(model_path_local) and os.path.isfile(f"{model_path_local}/cfg_args"):
        print("Model 3DGS da ton tai, bo qua train (xoa thu muc neu muon train lai).")
        return model_path_local, train_iterations

    if densified_ply is not None:
        backup = f"{sparse0}/points3D_original_backup.bin"
        if os.path.isfile(f"{sparse0}/points3D.bin") and not os.path.isfile(backup):
            run(f'cp "{sparse0}/points3D.bin" "{backup}"')
        run(f'rm -f "{sparse0}/points3D.bin"')
        run(f'cp "{densified_ply}" "{sparse0}/points3D.ply"')

    num_train_images = count_train_images(local_root)
    milestone_iters = calc_iteration_milestones(train_iterations)
    test_iters = milestone_iters
    save_iters = milestone_iters

    densify_from_iter, densify_until_iter, densification_interval, opacity_reset_interval = \
        calc_densify_params(train_iterations)

    print(f"\n{'='*70}")
    print(f"  TRAIN 3D GAUSSIAN SPLATTING (graphdeco-inria) - THONG TIN TRAINING")
    print(f"  So iterations (CO DINH):    {train_iterations}")
    print(f"  So anh train:                {num_train_images}")
    print(f"  Iteration test/save:         {milestone_iters}")
    print(f"  SH degree:                   {SH_DEGREE}")
    print(f"  Densify params (scaled):")
    print(f"    densify_from_iter      = {densify_from_iter}")
    print(f"    densify_until_iter     = {densify_until_iter}")
    print(f"    densification_interval = {densification_interval}")
    print(f"    opacity_reset_interval = {opacity_reset_interval}")
    print(f"{'='*70}\n")

    test_iters_str = " ".join(str(i) for i in test_iters)
    save_iters_str = " ".join(str(i) for i in save_iters)

    cmd = (
        f'cd {GS_REPO_DIR} && python train.py '
        f'-s "{local_root}/train" -m "{model_path_local}" --eval '
        f'--sh_degree {SH_DEGREE} '
        f'--iterations {train_iterations} '
        f'--densify_from_iter {densify_from_iter} '
        f'--densify_until_iter {densify_until_iter} '
        f'--densification_interval {densification_interval} '
        f'--opacity_reset_interval {opacity_reset_interval} '
        f'--test_iterations {test_iters_str} '
        f'--save_iterations {save_iters_str} '
        f'--checkpoint_iterations {train_iterations}'
    )
    run(cmd)

    run(f'mkdir -p "{out_dir}" && cp -r "{model_path_local}" "{out_dir}/gs_model_backup"')
    print(f"\n  >>> Da backup model 3DGS len Drive: {out_dir}/gs_model_backup")
    print(f"  >>> Train xong: {train_iterations} iterations (co dinh)")
    return model_path_local, train_iterations


# =============================================================================
# BUOC 3.5: GAUSSIAN PRUNING (sau train, TRUOC fine-tune)
# =============================================================================

def prune_gaussians_by_opacity(gaussians,
                                opacity_threshold=DEFAULT_PRUNE_OPACITY_THRESHOLD,
                                min_keep_ratio=DEFAULT_PRUNE_MIN_KEEP_RATIO):
    """*** GAUSSIAN PRUNING ***
    Vanilla 3DGS (graphdeco-inria) co opacity la 1 tham so TRUC TIEP tren moi
    Gaussian (khong view-dependent nhu Scaffold-GS), truy cap qua thuoc tinh
    `gaussians.get_opacity` (da qua sigmoid activation, gia tri trong [0, 1]).
    Vi vay pruning o day chinh xac & don gian hon nhieu: khong can render qua
    nhieu camera de uoc luong nhu ban Scaffold-GS truoc.

    Uu tien goi `gaussians.prune_points(mask)` (ham co san trong repo goc, tu
    dong xu ly ca optimizer state); neu khong co thi fallback loc tensor thu
    cong (optimizer se duoc tao lai tu dau o buoc fine-tune ngay sau do).
    """
    import torch

    n_before = gaussians._xyz.shape[0] if hasattr(gaussians, "_xyz") else None
    if n_before is None or n_before == 0:
        print("[Pruning] Khong tim thay tensor '_xyz', bo qua pruning.")
        return 0, 0

    with torch.no_grad():
        avg_opacity = gaussians.get_opacity.detach().float().flatten()

    if avg_opacity.shape[0] != n_before:
        print("[Pruning] Kich thuoc get_opacity khong khop so Gaussian, bo qua pruning de an toan.")
        return n_before, n_before

    prune_mask = avg_opacity < opacity_threshold
    keep_ratio = 1.0 - prune_mask.float().mean().item()

    if keep_ratio < min_keep_ratio:
        k = int(min_keep_ratio * n_before)
        k = max(1, min(k, n_before))
        sorted_vals, _ = torch.sort(avg_opacity, descending=True)
        safe_threshold = sorted_vals[k - 1].item()
        prune_mask = avg_opacity < safe_threshold
        print(f"[Pruning] Nguong {opacity_threshold} qua manh tay (chi giu {keep_ratio*100:.1f}%), "
              f"tu dong dieu chinh de giu toi thieu {min_keep_ratio*100:.0f}% Gaussian.")

    keep_mask = ~prune_mask

    if hasattr(gaussians, "prune_points") and callable(getattr(gaussians, "prune_points")):
        try:
            gaussians.prune_points(prune_mask)
            print("[Pruning] Da goi gaussians.prune_points(mask) (ham chuan cua repo graphdeco-inria).")
        except Exception as e:
            print(f"[Pruning] gaussians.prune_points(mask) loi ({e}), fallback sang loc tensor thu cong.")
            _manual_prune_tensors(gaussians, keep_mask)
    else:
        _manual_prune_tensors(gaussians, keep_mask)

    n_after = gaussians._xyz.shape[0]
    print(f"[Pruning] So Gaussian: {n_before} -> {n_after} "
          f"(giu {n_after/max(1,n_before)*100:.1f}%, nguong opacity={opacity_threshold})")
    return n_before, n_after


def _manual_prune_tensors(gaussians, keep_mask):
    """Fallback: loc thu cong cac tensor cot loi cua GaussianModel theo keep_mask.
    CANH BAO: khong dong bo optimizer state (Adam momentum se lech shape) - chi
    dung khi prune_points() khong co san; optimizer se duoc tao lai tu dau o
    buoc fine-tune ngay sau do nen van an toan cho pipeline nay."""
    import torch
    tensors_to_prune = GEOM_TENSOR_NAMES + COLOR_TENSOR_NAMES
    for name in tensors_to_prune:
        attr = getattr(gaussians, name, None)
        if isinstance(attr, torch.nn.Parameter):
            attr.data = attr.data[keep_mask]
        elif isinstance(attr, torch.Tensor):
            setattr(gaussians, name, attr[keep_mask])
    # Cac buffer phu (khong bat buoc nhung nen dong bo neu ton tai)
    for name in ["max_radii2D", "xyz_gradient_accum", "denom"]:
        attr = getattr(gaussians, name, None)
        if isinstance(attr, torch.Tensor) and attr.shape[0] == keep_mask.shape[0]:
            setattr(gaussians, name, attr[keep_mask])


# =============================================================================
# BUOC 4: EXPOSURE COMPENSATION + LPIPS FINE-TUNING (batch hieu dung + AMP)
# + TRANSFER LEARNING qua GlobalColorMLP (module DUY NHAT dung chung giua cac scene)
# =============================================================================

def _build_global_color_mlp(nn, hidden=GLOBAL_COLOR_MLP_HIDDEN):
    """Tao mot mang nho, DOC LAP VOI SO DIEM/SO ANH cua tung scene: nhan mau
    RGB render ra va hoc cach hieu chinh (residual). Vi khong phu thuoc kich
    thuoc scene, day la module DUY NHAT co the luu/nap lai (transfer) giua cac
    scene khac nhau - thay the vai tro cua cac MLP dung chung trong Scaffold-GS."""
    import torch

    class _GlobalColorMLP(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, 3),
            )
            # Khoi tao lop cuoi ve 0 -> ban dau MLP la identity (residual = 0),
            # tranh pha hong anh render ngay tu dau khi chua fine-tune.
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)

        def forward(self, img):
            # img: (1, 3, H, W) trong [0, 1]
            b, c, h, w = img.shape
            flat = img.permute(0, 2, 3, 1).reshape(-1, 3)
            residual = self.net(flat)
            out = (flat + residual).reshape(b, h, w, c).permute(0, 3, 1, 2)
            return torch.clamp(out, 0.0, 1.0)

    return _GlobalColorMLP(hidden)


def finetune_exposure_and_lpips(local_root, out_dir, model_path_local, finetune_iters, trained_iterations,
                                 finetune_batch_size=DEFAULT_FINETUNE_BATCH_SIZE, use_amp=True,
                                 do_prune=True, prune_opacity_threshold=DEFAULT_PRUNE_OPACITY_THRESHOLD,
                                 prune_min_keep_ratio=DEFAULT_PRUNE_MIN_KEEP_RATIO,
                                 transfer_learning=False, prev_color_mlp_path=None):
    print("\n========== BUOC 4: [Pruning] -> Exposure Compensation + LPIPS Fine-tuning ==========")
    configure_gpu_for_max_utilization()
    finetune_dir_local = f"{local_root}/finetune"
    if os.path.isfile(f"{finetune_dir_local}/point_cloud_finetuned.ply"):
        print("Da fine-tune truoc do, bo qua buoc nay.")
        return _reload_finetuned_state(local_root, model_path_local, trained_iterations, finetune_dir_local)

    sys.path.append(GS_REPO_DIR)
    os.chdir(GS_REPO_DIR)


    sys.path.insert(0, GS_REPO_DIR)
    sys.path.insert(0, os.path.join(GS_REPO_DIR, "submodules"))
    sys.path.insert(0, os.path.join(GS_REPO_DIR, "submodules/simple-knn"))
    sys.path.insert(0, os.path.join(GS_REPO_DIR, "submodules/diff-gaussian-rasterization"))

    import torch
    import torch.nn as nn
    import lpips
    from tqdm import tqdm

    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim
    from arguments import ModelParams, PipelineParams, OptimizationParams

    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    args = parser.parse_args([
        "-s", f"{local_root}/train",
        "-m", model_path_local,
        "--eval",
        "--sh_degree", str(SH_DEGREE),
    ])
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=trained_iterations, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()

    # ===== GAUSSIAN PRUNING (BUOC 3.5) - CHAY NGAY SAU KHI LOAD MODEL DA TRAIN,
    # TRUOC KHI TAO OPTIMIZER FINE-TUNE. THU TU: TRAIN -> PRUNE -> FINE-TUNE =====
    if do_prune:
        prune_gaussians_by_opacity(
            gaussians,
            opacity_threshold=prune_opacity_threshold,
            min_keep_ratio=prune_min_keep_ratio,
        )
    else:
        print("[Pruning] Bo qua theo yeu cau (--no_prune).")

    lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    print(f"Nap lai {len(train_cams)} camera train de fine-tune. "
          f"Load iteration = {trained_iterations}. "
          f"Batch hieu dung/buoc = {finetune_batch_size}, AMP = {use_amp}")

    class ExposureCompensation(nn.Module):
        def __init__(self, num_images):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(num_images, 3, 1, 1))
            self.bias = nn.Parameter(torch.zeros(num_images, 3, 1, 1))

        def forward(self, img, idx):
            return torch.clamp(img * self.scale[idx] + self.bias[idx], 0.0, 1.0)

    exposure_module = ExposureCompensation(len(train_cams)).to('cuda')

    # ===== GLOBAL COLOR MLP (TRANSFER LEARNING) =====
    # Module NHO, DOC LAP voi so diem/so anh -> co the transfer giua cac scene.
    # NAP LAI TU DRIVE (prev_color_mlp_path) neu co - day la buoc "TIEP TUC
    # TRANSFER" tu trong so cua scene truoc (hoac scene xong gan nhat khi resume).
    color_mlp = None
    if transfer_learning:
        color_mlp = _build_global_color_mlp(nn).to('cuda')
        if prev_color_mlp_path is not None and os.path.isfile(prev_color_mlp_path):
            try:
                color_mlp.load_state_dict(torch.load(prev_color_mlp_path, map_location='cuda'))
                print(f"[Transfer learning] Da nap trong so GlobalColorMLP tu scene truoc (Drive): "
                      f"{prev_color_mlp_path}")
            except Exception as e:
                print(f"[Transfer learning] Khong nap duoc GlobalColorMLP tu scene truoc ({e}), "
                      f"khoi tao lai tu dau (identity).")
        else:
            print("[Transfer learning] Khong co GlobalColorMLP cua scene truoc -> "
                  "day la SCENE DAU CHUOI, khoi tao MLP tu dau (identity).")

    # Fine-tune truc tiep cac tensor cot loi cua GaussianModel (khong co MLP nhu
    # Scaffold-GS): mau (_features_dc/_features_rest), do mo (_opacity), ti le (_scaling).
    # Tao optimizer MOI hoan toan (tranh optimizer state cu bi lech shape sau pruning).
    finetune_params = [{'params': exposure_module.parameters(), 'lr': 1e-3, 'name': 'exposure'}]
    if color_mlp is not None:
        finetune_params.append({'params': color_mlp.parameters(), 'lr': 5e-4, 'name': 'global_color_mlp'})
    for name in ["_features_dc", "_features_rest", "_opacity", "_scaling"]:
        attr = getattr(gaussians, name, None)
        if isinstance(attr, torch.Tensor):
            attr.requires_grad_(True)
            finetune_params.append({'params': [attr], 'lr': 1e-4, 'name': name})

    optimizer = torch.optim.Adam(finetune_params, lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Cac nhom tham so fine-tune:", [g['name'] for g in finetune_params])

    n_cams = len(train_cams)
    total_steps = max(1, finetune_iters // finetune_batch_size)
    global_it = 0

    for step in tqdm(range(1, total_steps + 1), desc="LPIPS fine-tuning (buoc toi uu, moi buoc = batch camera)"):
        optimizer.zero_grad(set_to_none=True)
        step_loss_val = 0.0
        step_l1_val = 0.0
        step_dssim_val = 0.0
        step_lpips_val = 0.0

        for _ in range(finetune_batch_size):
            global_it += 1
            cam_idx = random.randint(0, n_cams - 1)
            viewpoint_cam = train_cams[cam_idx]

            with amp_autocast_ctx(use_amp):
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image = render_pkg["render"].unsqueeze(0)
                gt_image = viewpoint_cam.original_image.unsqueeze(0).to('cuda', non_blocking=True)

                image_comp = exposure_module(image, cam_idx)
                if color_mlp is not None:
                    image_comp = color_mlp(image_comp)

                l1 = l1_loss(image_comp, gt_image)
                dssim = 1.0 - ssim(image_comp, gt_image)
                lpips_w = LPIPS_WEIGHT_START + (LPIPS_WEIGHT_END - LPIPS_WEIGHT_START) * (global_it / finetune_iters)
                lp_loss = lpips_fn(image_comp * 2 - 1, gt_image * 2 - 1).mean()

                loss = (1.0 - LAMBDA_DSSIM) * l1 + LAMBDA_DSSIM * dssim + lpips_w * lp_loss
                loss_to_backward = loss / finetune_batch_size

            scaler.scale(loss_to_backward).backward()

            step_loss_val += loss.item()
            step_l1_val += l1.item()
            step_dssim_val += dssim.item()
            step_lpips_val += lp_loss.item()

        scaler.step(optimizer)
        scaler.update()

        if step % max(1, (200 // finetune_batch_size)) == 0 or step == 1:
            b = finetune_batch_size
            tqdm.write(f"[iter~{global_it}/{finetune_iters}] "
                       f"L1={step_l1_val/b:.4f} DSSIM={step_dssim_val/b:.4f} "
                       f"LPIPS={step_lpips_val/b:.4f} total={step_loss_val/b:.4f}")

    os.makedirs(finetune_dir_local, exist_ok=True)
    gaussians.save_ply(f"{finetune_dir_local}/point_cloud_finetuned.ply")
    torch.save(exposure_module.state_dict(), f"{finetune_dir_local}/exposure_module.pt")
    color_mlp_path_local = None
    if color_mlp is not None:
        color_mlp_path_local = f"{finetune_dir_local}/global_color_mlp.pt"
        torch.save(color_mlp.state_dict(), color_mlp_path_local)
        print(f"[Transfer learning] Da luu GlobalColorMLP: {color_mlp_path_local} "
              f"(se dung de transfer sang scene tiep theo).")

    run(f'mkdir -p "{out_dir}" && rm -rf "{out_dir}/finetune" && cp -r "{finetune_dir_local}" "{out_dir}/finetune"')
    print("Da luu model fine-tune len Drive.")

    return {
        "gaussians": gaussians, "scene": scene, "pipe": pipe,
        "background": background, "finetune_dir_local": finetune_dir_local,
        "finetune_dir_drive": f"{out_dir}/finetune",
        "color_mlp": color_mlp,
        "color_mlp_path_drive": (f"{out_dir}/finetune/global_color_mlp.pt" if color_mlp is not None else None),
    }


def _reload_finetuned_state(local_root, model_path_local, trained_iterations, finetune_dir_local):
    """Dung khi buoc fine-tune da chay xong tu truoc va script duoc chay lai:
    load lai truc tiep tu ply da fine-tune (khong can load_iteration cua Scene goc)."""
    configure_gpu_for_max_utilization()
    sys.path.append(GS_REPO_DIR)
    os.chdir(GS_REPO_DIR)
    import torch
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams, OptimizationParams

    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    args = parser.parse_args([
        "-s", f"{local_root}/train",
        "-m", model_path_local,
        "--eval",
        "--sh_degree", str(SH_DEGREE),
    ])
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=trained_iterations, shuffle=False)
    gaussians.load_ply(f"{finetune_dir_local}/point_cloud_finetuned.ply")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    color_mlp = None
    color_mlp_path = f"{finetune_dir_local}/global_color_mlp.pt"
    if os.path.isfile(color_mlp_path):
        import torch.nn as nn
        color_mlp = _build_global_color_mlp(nn).to('cuda')
        color_mlp.load_state_dict(torch.load(color_mlp_path, map_location='cuda'))
        print(f"[Transfer learning] Da nap lai GlobalColorMLP da luu: {color_mlp_path}")

    return {
        "gaussians": gaussians, "scene": scene, "pipe": pipe,
        "background": background, "finetune_dir_local": finetune_dir_local,
        "color_mlp": color_mlp,
        "color_mlp_path_drive": None,
    }


# =============================================================================
# BUOC 4.5: QUANTIZATION (sau fine-tune, TRUOC render)
# =============================================================================

def _tensor_bytes(t):
    return t.numel() * t.element_size()


def quantize_gaussians(gaussians, finetune_dir_local, out_dir,
                        quant_geom_fp16=QUANT_GEOM_FP16, quant_color_int8=QUANT_COLOR_INT8):
    """*** QUANTIZATION ***
    1) Luu ban NEN THAT SU (fp16 hinh hoc + int8 mau/SH) ra Drive de biet dung
       luong tiet kiem duoc bao nhieu.
    2) Ap dung FAKE-QUANTIZATION (ep xuong roi ep nguoc ve fp32) TRUC TIEP len
       `gaussians` dang dung trong bo nho, de buoc render/metrics phia sau
       phan anh dung chat luong SAU KHI NEN.
    """
    import torch

    quant_dir_local = f"{finetune_dir_local}_quantized"
    os.makedirs(quant_dir_local, exist_ok=True)

    size_before = 0
    size_after = 0
    quant_report = {}

    # ---- 1) Hinh hoc (_xyz, _scaling, _rotation, _opacity): FP16 ----
    for name in GEOM_TENSOR_NAMES:
        attr = getattr(gaussians, name, None)
        if attr is None:
            continue
        tensor = attr.data if isinstance(attr, torch.nn.Parameter) else attr
        if not isinstance(tensor, torch.Tensor):
            continue
        size_before += _tensor_bytes(tensor)
        if quant_geom_fp16:
            half_tensor = tensor.detach().half()
            np.save(f"{quant_dir_local}/{name}_fp16.npy", half_tensor.cpu().numpy())
            size_after += _tensor_bytes(half_tensor)
            new_val = half_tensor.float()
            if isinstance(attr, torch.nn.Parameter):
                attr.data.copy_(new_val)
            else:
                setattr(gaussians, name, new_val)
            quant_report[name] = "fp32 -> fp16 (fake-quant, size that su giam 2x)"
        else:
            size_after += _tensor_bytes(tensor)

    # ---- 2) Mau / spherical harmonics (_features_dc, _features_rest): INT8 (per-tensor scale) ----
    for name in COLOR_TENSOR_NAMES:
        attr = getattr(gaussians, name, None)
        if attr is None:
            continue
        tensor = attr.data if isinstance(attr, torch.nn.Parameter) else attr
        if not isinstance(tensor, torch.Tensor):
            continue
        size_before += _tensor_bytes(tensor)
        if quant_color_int8:
            scale = (tensor.detach().abs().max() / 127.0).clamp(min=1e-8)
            q = torch.clamp((tensor.detach() / scale).round(), -127, 127).to(torch.int8)
            np.save(f"{quant_dir_local}/{name}_int8.npy", q.cpu().numpy())
            np.save(f"{quant_dir_local}/{name}_scale.npy", scale.cpu().numpy())
            size_after += _tensor_bytes(q) + 4  # + 4 byte cho scale (float32)
            dequant = q.float() * scale
            if isinstance(attr, torch.nn.Parameter):
                attr.data.copy_(dequant)
            else:
                setattr(gaussians, name, dequant)
            quant_report[name] = f"fp32 -> int8 (per-tensor scale={scale.item():.6f}), size giam ~4x"
        else:
            size_after += _tensor_bytes(tensor)

    mb_before = size_before / (1024 ** 2)
    mb_after = size_after / (1024 ** 2)
    ratio = (size_before / max(1, size_after))
    print(f"\n[Quantization] Dung luong model (uoc luong): {mb_before:.2f} MB -> {mb_after:.2f} MB "
          f"(nen ~{ratio:.2f}x)")
    for k, v in quant_report.items():
        print(f"  - {k}: {v}")

    report = {
        "size_before_mb": mb_before,
        "size_after_mb": mb_after,
        "compression_ratio": ratio,
        "details": quant_report,
    }
    with open(f"{quant_dir_local}/quantization_report.json", "w") as f:
        json.dump(report, f, indent=2)

    run(f'mkdir -p "{out_dir}" && rm -rf "{out_dir}/finetune_quantized" '
        f'&& cp -r "{quant_dir_local}" "{out_dir}/finetune_quantized"')
    print(f"Da luu ban model NEN (fp16/int8) + bao cao dung luong vao: {out_dir}/finetune_quantized")

    return report


# =============================================================================
# BUOC 5: RENDER TEST_POSES.CSV
# =============================================================================

class MiniCam:
    def __init__(self, width, height, fovy, fovx, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = world_view_transform.inverse()
        self.camera_center = view_inv[3][:3]


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def get_projection_matrix(znear, zfar, fovX, fovY):
    import torch
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def build_minicam(qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height, znear=0.01, zfar=100.0):
    import torch
    R = qvec2rotmat([qw, qx, qy, qz])
    t = np.array([tx, ty, tz], dtype=np.float64)
    W2C = np.eye(4)
    W2C[:3, :3] = R
    W2C[:3, 3] = t
    world_view_transform = torch.tensor(W2C, dtype=torch.float32).transpose(0, 1).cuda()

    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    proj = get_projection_matrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

    return MiniCam(width, height, fovy, fovx, world_view_transform, full_proj_transform)


def render_test_poses(local_root, out_dir, finetune_state, use_amp=True):
    print("\n========== BUOC 6: Render test_poses.csv ==========")
    configure_gpu_for_max_utilization()
    import torch
    import torchvision
    import pandas as pd
    from tqdm import tqdm
    from gaussian_renderer import render

    render_dir_local = f"{local_root}/renders_test"
    os.makedirs(render_dir_local, exist_ok=True)

    gaussians = finetune_state["gaussians"]
    pipe = finetune_state["pipe"]
    background = finetune_state["background"]
    color_mlp = finetune_state.get("color_mlp", None)
    if color_mlp is not None:
        color_mlp.eval()
        print("[Transfer learning] Ap dung GlobalColorMLP (da fine-tune/transfer) khi render test.")

    test_csv = pd.read_csv(f"{local_root}/test/test_poses.csv")
    print(test_csv.head())

    with torch.no_grad():
        for _, row in tqdm(test_csv.iterrows(), total=len(test_csv), desc="Rendering test poses"):
            cam = build_minicam(
                row['qw'], row['qx'], row['qy'], row['qz'],
                row['tx'], row['ty'], row['tz'],
                row['fx'], row['fy'], row['cx'], row['cy'],
                int(row['width']), int(row['height']),
            )
            with amp_autocast_ctx(use_amp):
                out = render(cam, gaussians, pipe, background)
                img = torch.clamp(out["render"], 0.0, 1.0)
                if color_mlp is not None:
                    img = color_mlp(img.unsqueeze(0)).squeeze(0)
            save_name = row['image_name']
            if not save_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                save_name += '.png'
            torchvision.utils.save_image(img.float(), f"{render_dir_local}/{save_name}")

    run(f'mkdir -p "{out_dir}" && rm -rf "{out_dir}/renders_test" && cp -r "{render_dir_local}" "{out_dir}/renders_test"')
    print("Da render toan bo anh test va luu vao Drive:", f"{out_dir}/renders_test")
    return render_dir_local


# =============================================================================
# BUOC 7: METRICS (CHI CHO SPLIT PUBLIC - CO GROUND TRUTH), tinh theo BATCH
# =============================================================================

def compute_public_metrics(local_root, out_dir, scene_name, render_dir_local,
                            metrics_batch_size=DEFAULT_METRICS_BATCH_SIZE, use_amp=True):
    print("\n========== BUOC 7: Tinh metrics PSNR/SSIM/LPIPS (public, batch GPU) ==========")
    configure_gpu_for_max_utilization()
    gt_dir = f"{local_root}/test/images"
    if not os.path.isdir(gt_dir):
        print(f"[Canh bao] Split public nhung khong thay ground truth tai {gt_dir}, bo qua metrics.")
        return None

    import cv2
    import torch
    import lpips
    import pandas as pd
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim

    test_csv = pd.read_csv(f"{local_root}/test/test_poses.csv")
    lpips_fn = lpips.LPIPS(net='alex').to('cuda').eval()

    pairs = []
    for _, row in test_csv.iterrows():
        name = str(row['image_name'])
        base_no_ext = os.path.splitext(name)[0]
        render_name = name if name.lower().endswith(('.png', '.jpg', '.jpeg')) else f"{name}.png"
        render_path = f"{render_dir_local}/{render_name}"
        gt_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            candidate = f"{gt_dir}/{base_no_ext}{ext}"
            if os.path.isfile(candidate):
                gt_path = candidate
                break
        if gt_path is None or not os.path.isfile(render_path):
            print(f"  [Bo qua] Thieu anh de so sanh cho '{name}' "
                  f"(render_ton_tai={os.path.isfile(render_path)}, gt_ton_tai={gt_path is not None})")
            continue
        pairs.append((name, render_path, gt_path))

    rows = []
    with torch.no_grad():
        for i in range(0, len(pairs), metrics_batch_size):
            batch = pairs[i:i + metrics_batch_size]
            renders_np, gts_np, names = [], [], []
            for name, render_path, gt_path in batch:
                img_render = cv2.cvtColor(cv2.imread(render_path), cv2.COLOR_BGR2RGB)
                img_gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
                if img_render.shape != img_gt.shape:
                    img_gt = cv2.resize(img_gt, (img_render.shape[1], img_render.shape[0]),
                                         interpolation=cv2.INTER_AREA)
                psnr_val = compute_psnr(img_gt, img_render, data_range=255)
                ssim_val = compute_ssim(img_gt, img_render, channel_axis=2, data_range=255)
                rows.append({'image_name': name, 'psnr': psnr_val, 'ssim': ssim_val, 'lpips': None})
                renders_np.append(img_render)
                gts_np.append(img_gt)
                names.append(name)

            same_shape = len(set(a.shape for a in renders_np)) == 1
            if same_shape and len(renders_np) > 0:
                t_render = torch.from_numpy(np.stack(renders_np) / 255.0).permute(0, 3, 1, 2).float().cuda() * 2 - 1
                t_gt = torch.from_numpy(np.stack(gts_np) / 255.0).permute(0, 3, 1, 2).float().cuda() * 2 - 1
                with amp_autocast_ctx(use_amp):
                    lpips_vals = lpips_fn(t_render, t_gt).squeeze().detach().float().cpu().numpy()
                lpips_vals = np.atleast_1d(lpips_vals)
                for j, name in enumerate(names):
                    for r in rows:
                        if r['image_name'] == name and r['lpips'] is None:
                            r['lpips'] = float(lpips_vals[j])
                            break
            else:
                for render_np, gt_np, name in zip(renders_np, gts_np, names):
                    t_render = torch.from_numpy(render_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
                    t_gt = torch.from_numpy(gt_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
                    with amp_autocast_ctx(use_amp):
                        lp_val = lpips_fn(t_render, t_gt).item()
                    for r in rows:
                        if r['image_name'] == name and r['lpips'] is None:
                            r['lpips'] = lp_val
                            break

    del lpips_fn
    torch.cuda.empty_cache()

    if not rows:
        print("  Khong co cap anh (render, ground truth) nao hop le -> khong the tinh metrics.")
        return None

    df = pd.DataFrame(rows)
    metrics_csv_local = f"{local_root}/metrics.csv"
    df.to_csv(metrics_csv_local, index=False)
    run(f'mkdir -p "{out_dir}" && cp "{metrics_csv_local}" "{out_dir}/metrics.csv"')

    avg_psnr, avg_ssim, avg_lpips = df['psnr'].mean(), df['ssim'].mean(), df['lpips'].mean()
    print(f"[{scene_name}] Metrics trung binh tren {len(df)} anh:")
    print(f"    PSNR  = {avg_psnr:.3f} dB")
    print(f"    SSIM  = {avg_ssim:.4f}")
    print(f"    LPIPS = {avg_lpips:.4f}")
    print(f"  Da luu chi tiet tung anh vao: {out_dir}/metrics.csv")

    return {'scene': scene_name, 'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips, 'n_images': len(df)}


# =============================================================================
# RESUME / SKIP-TRAIN HELPERS (KIEM TRA DRIVE O PHAN OUTPUT)
# =============================================================================

def _finetune_dir_has_required_files(finetune_dir):
    if finetune_dir is None or not os.path.isdir(finetune_dir):
        return False
    return os.path.isfile(f"{finetune_dir}/point_cloud_finetuned.ply")


def _render_dir_has_output(render_dir):
    if render_dir is None or not os.path.isdir(render_dir):
        return False
    return len(glob.glob(f"{render_dir}/*")) > 0


def get_completed_scene_state(out_dir):
    """Kiem tra tren Drive (out_dir = {output_root}/{split}/{scene}) xem scene
    nay DA CO du finetune + render hay chua. Day la diem kiem tra chinh de
    quyet dinh BO QUA hay TIEP TUC train mot scene."""
    finetune_dir_drive = f"{out_dir}/finetune"
    render_dir_drive = f"{out_dir}/renders_test"
    ok = _finetune_dir_has_required_files(finetune_dir_drive) and _render_dir_has_output(render_dir_drive)
    return ok, (finetune_dir_drive if ok else None)


def get_scene_color_mlp_path(out_dir):
    """Neu scene nay (tren Drive, tai out_dir/finetune/) da luu GlobalColorMLP,
    tra ve duong dan; nguoc lai tra ve None. Dung de tiep tuc chuoi transfer
    learning tu trong so cua mot scene da xong."""
    candidate = f"{out_dir}/finetune/global_color_mlp.pt"
    return candidate if os.path.isfile(candidate) else None


def scan_split_resume_status(output_root, split, scene_list):
    """*** KIEM TRA DRIVE O PHAN OUTPUT ***
    Quet TAT CA scene cua 1 split tren Drive (output_root/split/<scene>/...) va
    in ra bao cao ro rang: scene nao DA CO KET QUA (se BO QUA), scene nao CHUA
    (se train/tiep tuc). Tra ve (done_scenes, pending_scenes)."""
    print(f"\n----- [Kiem tra Drive] Trang thai resume cho split '{split}' "
          f"(output_root={output_root}) -----")
    done_scenes, pending_scenes = [], []
    for scene_name in scene_list:
        out_dir = f"{output_root}/{split}/{scene_name}"
        is_done, _ = get_completed_scene_state(out_dir)
        if is_done:
            done_scenes.append(scene_name)
            note = ""
            if get_scene_color_mlp_path(out_dir) is not None:
                note = "  (co san GlobalColorMLP -> dung de transfer tiep)"
            print(f"  [DA XONG]   {scene_name:<20s} -> {out_dir}  => BO QUA train/finetune/render{note}")
        else:
            pending_scenes.append(scene_name)
            print(f"  [CHUA XONG] {scene_name:<20s} -> se train/tiep tuc")
    print(f"  Tong: {len(done_scenes)} da xong / {len(pending_scenes)} con lai / {len(scene_list)} scene.")
    print("-------------------------------------------------------------------\n")
    return done_scenes, pending_scenes


def load_saved_metrics(out_dir, scene_name):
    metrics_csv = f"{out_dir}/metrics.csv"
    if not os.path.isfile(metrics_csv):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        if len(df) == 0:
            return None
        return {
            'scene': scene_name,
            'psnr': float(df['psnr'].mean()),
            'ssim': float(df['ssim'].mean()),
            'lpips': float(df['lpips'].mean()),
            'n_images': int(len(df)),
        }
    except Exception as e:
        print(f"  [Canh bao] Khong doc duoc metrics.csv da luu ({metrics_csv}): {e}")
        return None


# =============================================================================
# XU LY 1 SCENE
# (kiem tra Drive -> train -> prune -> finetune -> quantize -> render test -> [public] metrics)
# =============================================================================

def process_scene(scene_name, split, args, prev_color_mlp_path=None, is_first_in_chain=True):
    """Xu ly 1 scene. LUON kiem tra Drive (out_dir) truoc: neu scene da co du
    finetune + render luu san thi BO QUA train/finetune/render va CHI DOC LAI
    ket qua (+ doc lai duong dan GlobalColorMLP da luu de tra ve cho vong lap
    ben ngoai tiep tuc chuoi transfer).

    Neu args.transfer_learning=True:
        - is_first_in_chain=True (scene dau cua split, hoac chua co scene nao
          xong truoc do de lay trong so): fine-tune GlobalColorMLP tu dau,
          dung args.finetune_iters.
        - is_first_in_chain=False (co scene truoc da xong): NAP LAI
          GlobalColorMLP tu prev_color_mlp_path (trong so tren Drive cua scene
          truoc), fine-tune tiep voi it iteration hon (args.transfer_finetune_iters).

    Returns:
        (metrics_result, color_mlp_path_drive)
        color_mlp_path_drive: duong dan GlobalColorMLP vua luu/doc duoc tren
        Drive (de truyen cho scene tiep theo trong chuoi); None neu khong bat
        --transfer_learning hoac chua co file nao.
    """
    data_dir = f"{args.data_root}/{split}/{scene_name}"
    out_dir = f"{args.output_root}/{split}/{scene_name}"
    local_root = f"/content/work/{split}/{scene_name}"
    os.makedirs(out_dir, exist_ok=True)

    iterations_for_scene = args.iterations
    if args.transfer_learning and not is_first_in_chain:
        finetune_iters_for_scene = args.transfer_finetune_iters
    else:
        finetune_iters_for_scene = args.finetune_iters

    # ===== KIEM TRA DRIVE (OUTPUT) - NEU DA CO FOLDER TRAIN XONG THI BO QUA =====
    if not args.force_retrain:
        is_done, existing_finetune_dir = get_completed_scene_state(out_dir)
        if is_done:
            print(f"\n{'#'*72}")
            print(f"###  SPLIT : {split}")
            print(f"###  SCENE : {scene_name}")
            print(f"###  [Kiem tra Drive] Da tim thay ket qua (finetune + render) tai:")
            print(f"###      {out_dir}")
            print(f"###  ==> BO QUA TRAIN/PRUNE/FINETUNE/QUANTIZE/RENDER, chi doc lai ket qua.")
            print(f"{'#'*72}")

            metrics_result = None
            if split == "public_set":
                metrics_result = load_saved_metrics(out_dir, scene_name)
                if metrics_result is not None:
                    print(f"[{scene_name}] Doc lai metrics da luu: "
                          f"PSNR={metrics_result['psnr']:.3f} dB  "
                          f"SSIM={metrics_result['ssim']:.4f}  "
                          f"LPIPS={metrics_result['lpips']:.4f}")
                else:
                    print(f"[{scene_name}] [Canh bao] La split public nhung khong doc duoc "
                          f"metrics.csv da luu (co the file bi thieu/hong).")

            existing_color_mlp_path = None
            if args.transfer_learning:
                existing_color_mlp_path = get_scene_color_mlp_path(out_dir)
                if existing_color_mlp_path is not None:
                    print(f"[Transfer learning] Scene da xong tu truoc, DUNG TRONG SO "
                          f"GlobalColorMLP da luu tren Drive de tiep tuc transfer cho scene "
                          f"tiep theo: {existing_color_mlp_path}")
                else:
                    print(f"[Transfer learning] Scene da xong nhung KHONG tim thay "
                          f"global_color_mlp.pt tren Drive (co the luc train chua bat "
                          f"--transfer_learning) -> chuoi transfer se bat dau lai tu scene ke tiep.")
            return metrics_result, existing_color_mlp_path

    print(f"\n\n{'#'*72}")
    print(f"###  SPLIT : {split}")
    print(f"###  SCENE : {scene_name}")
    print(f"###  Input : {data_dir}")
    print(f"###  Output: {out_dir}")
    print(f"###  [Kiem tra Drive] Chua co ket qua day du -> TIEN HANH train/tiep tuc.")
    print(f"###  ITERATIONS : {iterations_for_scene} (co dinh, repo goc graphdeco-inria/gaussian-splatting)")
    if args.transfer_learning:
        if is_first_in_chain:
            mode = "SCENE DAU CHUOI (chua co GlobalColorMLP nao tren Drive de dung -> train tu dau)"
        else:
            mode = f"SCENE TRANSFER (nap GlobalColorMLP tu: {prev_color_mlp_path})"
        print(f"###  Transfer learning: BAT - {mode}")
        print(f"###  Finetune iters cho scene nay: {finetune_iters_for_scene}")
    print(f"###  Pruning: {'BAT' if args.prune else 'TAT'} "
          f"(nguong opacity={args.prune_opacity_threshold}, min_keep_ratio={args.prune_min_keep_ratio})")
    print(f"###  Quantization: {'BAT' if args.quantize else 'TAT'}")
    print(f"###  GPU tuning: amp={args.amp}, depth_batch={args.depth_batch_size}, "
          f"finetune_batch={args.finetune_batch_size}, metrics_batch={args.metrics_batch_size}")
    print(f"{'#'*72}")

    prepare_data(data_dir, local_root)

    densified_ply = run_depth_anything_and_densify(
        local_root, out_dir, skip_depth=args.skip_depth,
        depth_batch_size=args.depth_batch_size, use_amp=args.amp,
    )

    # ===== BUOC 3: TRAIN 3DGS (ITERATIONS CO DINH, DOC LAP moi scene) =====
    model_path_local, trained_iterations = train_gaussian_splatting(
        local_root, out_dir, densified_ply, iterations_for_scene,
    )

    # ===== BUOC 3.5 (Pruning, ben trong ham nay) + BUOC 4: Fine-tune
    # (+ transfer GlobalColorMLP neu bat --transfer_learning) =====
    finetune_state = finetune_exposure_and_lpips(
        local_root, out_dir, model_path_local, finetune_iters_for_scene, trained_iterations=trained_iterations,
        finetune_batch_size=args.finetune_batch_size, use_amp=args.amp,
        do_prune=args.prune, prune_opacity_threshold=args.prune_opacity_threshold,
        prune_min_keep_ratio=args.prune_min_keep_ratio,
        transfer_learning=args.transfer_learning, prev_color_mlp_path=prev_color_mlp_path,
    )
    color_mlp_path_drive = finetune_state.get("color_mlp_path_drive", None)

    # ===== BUOC 4.5: QUANTIZATION (sau fine-tune, truoc render) =====
    quant_report = None
    if args.quantize:
        quant_report = quantize_gaussians(
            finetune_state["gaussians"], finetune_state["finetune_dir_local"], out_dir,
        )
    else:
        print("[Quantization] Bo qua theo yeu cau (--no_quantize).")

    render_dir_local = render_test_poses(local_root, out_dir, finetune_state, use_amp=args.amp)

    metrics_result = None
    if split == "public_set":
        metrics_result = compute_public_metrics(
            local_root, out_dir, scene_name, render_dir_local,
            metrics_batch_size=args.metrics_batch_size, use_amp=args.amp,
        )
        if metrics_result is not None and quant_report is not None:
            metrics_result["compression_ratio"] = quant_report["compression_ratio"]
    else:
        print(f"\n[Private] Khong co ground truth -> bo qua tinh metrics. "
              f"Anh render (dung de NOP BAI, tren model DA NEN) da luu tai: {out_dir}/renders_test")

    try:
        import torch
        del finetune_state
        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"\n================= HOAN TAT SCENE [{split}] {scene_name} =================")
    print(f"Ket qua da luu tai: {out_dir}")
    print(f"  depth/                    -> depth maps + point cloud densify")
    print(f"  gs_model_backup/          -> checkpoint 3DGS goc (truoc pruning)")
    print(f"  finetune/                 -> point cloud sau [Pruning ->] LPIPS fine-tune"
          + (" + global_color_mlp.pt (transfer)" if args.transfer_learning else ""))
    if args.quantize:
        print(f"  finetune_quantized/       -> ban NEN (fp16/int8) + quantization_report.json")
    print(f"  renders_test/             -> anh render cho test_poses.csv")
    if split == "public_set":
        print(f"  metrics.csv               -> PSNR/SSIM/LPIPS tung anh so voi ground truth")

    return metrics_result, color_mlp_path_drive


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    configure_gpu_for_max_utilization()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.scenes:
        requested = [s.strip() for s in args.scenes.split(",") if s.strip()]
        scenes_by_split = {sp: [] for sp in splits}
        for name in requested:
            sp = find_scene_split(args.data_root, splits, name)
            if sp is None:
                print(f"[Canh bao] Khong tim thay scene '{name}' trong cac split {splits} cua "
                      f"{args.data_root}, bo qua.")
                continue
            scenes_by_split[sp].append(name)
    else:
        scenes_by_split = discover_scenes(args.data_root, splits)

    total_scenes = sum(len(v) for v in scenes_by_split.values())
    if total_scenes == 0:
        print("Khong tim thay scene nao hop le trong", args.data_root)
        return

    for sp in splits:
        print(f"[{sp}] {len(scenes_by_split.get(sp, []))} scene: {scenes_by_split.get(sp, [])}")
    print(f"[Repo] Dung repo goc graphdeco-inria/gaussian-splatting (vanilla 3DGS).")
    print(f"[Iterations] MOI scene train CO DINH {args.iterations} iterations, DOC LAP tu dau "
          f"(train Gaussian khong co transfer-learning giua cac scene vi repo goc khong co MLP "
          f"dung chung tren chinh cac Gaussian; xem --transfer_learning cho GlobalColorMLP).")
    print(f"[Pruning] {'BAT' if args.prune else 'TAT'} - nguong opacity={args.prune_opacity_threshold}, "
          f"giu toi thieu {args.prune_min_keep_ratio*100:.0f}% Gaussian.")
    print(f"[Quantization] {'BAT' if args.quantize else 'TAT'} - fp16 hinh hoc + int8 mau (fake-quant de render).")
    print(f"[Transfer learning - GlobalColorMLP] {'BAT' if args.transfer_learning else 'TAT'}"
          + (f" - finetune_iters(scene dau)={args.finetune_iters}, "
             f"transfer_finetune_iters(scene sau)={args.transfer_finetune_iters}"
             if args.transfer_learning else ""))
    if not args.force_retrain:
        print(f"[Resume] BAT - se KIEM TRA DRIVE (output_root) truoc moi scene; scene nao da co du "
              f"finetune + render luu san se duoc BO QUA TRAIN va (neu bat transfer_learning) trong so "
              f"GlobalColorMLP cua no se duoc dung de TIEP TUC transfer cho cac scene con lai "
              f"(dung --force_retrain de tat, luon train lai tat ca).")
    else:
        print(f"[Resume] TAT (--force_retrain) - se train lai tat ca scene ke ca da co ket qua tren Drive.")
    print(f"[Toi uu GPU] amp={args.amp}  depth_batch_size={args.depth_batch_size}  "
          f"finetune_batch_size={args.finetune_batch_size}  metrics_batch_size={args.metrics_batch_size}")

    if not args.skip_env_setup:
        setup_environment()

    done, failed = [], []
    public_metrics_summary = []

    for sp in splits:
        scene_list = scenes_by_split.get(sp, [])
        if not scene_list:
            continue

        # ===== KIEM TRA DRIVE O PHAN OUTPUT TRUOC KHI XU LY SPLIT =====
        if not args.force_retrain:
            scan_split_resume_status(args.output_root, sp, scene_list)

        # prev_color_mlp_path duoc "thread" (truyen tiep) qua tung scene trong
        # CUNG 1 split: scene sau se nap trong so GlobalColorMLP cua scene lien
        # truoc do (du la vua train xong trong lan chay nay, hay doc lai tu
        # Drive vi scene do da xong tu truoc). Nho vay chuoi transfer khong bi
        # dut khi Colab bi ngat giua chung va phai chay lai script.
        prev_color_mlp_path = None

        for scene_name in scene_list:
            is_first_in_chain = (prev_color_mlp_path is None)
            try:
                metrics_result, color_mlp_path_drive = process_scene(
                    scene_name, sp, args,
                    prev_color_mlp_path=prev_color_mlp_path,
                    is_first_in_chain=is_first_in_chain,
                )
                done.append(f"{sp}/{scene_name}")
                if metrics_result is not None:
                    public_metrics_summary.append(metrics_result)
                if args.transfer_learning and color_mlp_path_drive is not None:
                    prev_color_mlp_path = color_mlp_path_drive
            except Exception as e:
                print(f"\n!!! LOI khi xu ly scene {sp}/{scene_name}: {e}")
                failed.append((f"{sp}/{scene_name}", str(e)))
                if not args.continue_on_error:
                    raise

    print("\n\n===================== TONG KET TOAN BO =====================")
    print(f"Thanh cong ({len(done)}): {done}")
    if failed:
        print(f"That bai ({len(failed)}):")
        for name, err in failed:
            print(f"  - {name}: {err}")

    if public_metrics_summary:
        import pandas as pd
        df = pd.DataFrame(public_metrics_summary)
        summary_path_local = "/content/summary_metrics_public.csv"
        df.to_csv(summary_path_local, index=False)
        summary_path_drive = f"{args.output_root}/public_set/summary_metrics_public.csv"
        run(f'mkdir -p "{args.output_root}/public_set" && cp "{summary_path_local}" "{summary_path_drive}"')

        print("\n----- TONG HOP METRICS TAP PUBLIC -----")
        print(df.to_string(index=False))
        print(f"\nTrung binh toan bo {len(df)} scene public:")
        print(f"    PSNR  = {df['psnr'].mean():.3f} dB")
        print(f"    SSIM  = {df['ssim'].mean():.4f}")
        print(f"    LPIPS = {df['lpips'].mean():.4f}")
        if "compression_ratio" in df.columns:
            print(f"    Compression ratio trung binh = {df['compression_ratio'].mean():.2f}x")
        print(f"Da luu bang tong hop tai: {summary_path_drive}")

    private_scenes_done = [d for d in done if d.startswith("private_set1/")]
    if private_scenes_done:
        print(f"\nBai nop (private, khong co ground truth) da san sang tai "
              f"{args.output_root}/private_set1/<scene>/renders_test/ cho: {private_scenes_done}")


if __name__ == "__main__":
    main()
