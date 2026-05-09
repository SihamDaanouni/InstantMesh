import os
import sys
import argparse
import numpy as np
import torch
import rembg
import gc
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from contextlib import contextmanager

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video

syncdreamer_root = "/content/SyncDreamer"
sys.path.append(syncdreamer_root)
from ldm.util import prepare_inputs
from generate import load_model
from ldm.models.diffusion.sync_dreamer import SyncDDIMSampler

# ============================================================
#  OPTION A — Zero123++ v1.2 avec vs sans UNet fine-tuné
#  OPTION B — SyncDreamer avec adaptateur 16→6 vues
# ============================================================

@contextmanager
def cd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras

def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    frames = torch.cat(frames, dim=1)[0]
    return frames

# ============================================================
# OPTION B — Adaptateur SyncDreamer
# ============================================================

# Azimuths des 16 vues générées par SyncDreamer (uniformément réparties)
SYNCDREAMER_AZIMUTHS = np.arange(16) * (360.0 / 16)
# [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]

# Azimuths cibles d'InstantMesh (convention Zero123++)
INSTANTMESH_TARGET_AZIMUTHS = np.array([30, 90, 150, 210, 270, 330])

def compute_syncdreamer_selected_indices():
    """
    Calcule une fois les indices des vues SyncDreamer les plus proches
    des azimuths cibles d'InstantMesh.
    Résultat attendu : [1, 4, 7, 9, 12, 15]
      → azimuths : [22.5, 90.0, 157.5, 202.5, 270.0, 337.5]
      → proches de : [30, 90, 150, 210, 270, 330]
    """
    indices = []
    for target in INSTANTMESH_TARGET_AZIMUTHS:
        diffs = np.abs(SYNCDREAMER_AZIMUTHS - target)
        diffs = np.minimum(diffs, 360 - diffs)
        indices.append(int(np.argmin(diffs)))
    return indices

# Calculé une seule fois, réutilisé dans Stage 1 et Stage 2
SYNCDREAMER_SELECTED_INDICES = compute_syncdreamer_selected_indices()


def select_syncdreamer_views(views_array: np.ndarray, rembg_session=None) -> torch.Tensor:
    """
    Sélectionne les 6 vues SyncDreamer les plus proches des azimuths InstantMesh,
    applique optionnellement un détourage fond blanc, redimensionne à 320×320,
    et retourne un tenseur [6, 3, 320, 320] dans [0, 1].

    Args:
        views_array : np.ndarray uint8 de shape [16, H, W, 3]
        rembg_session : session rembg ou None pour désactiver le détourage
    """
    print(f"[SyncDreamer] Indices sélectionnés : {SYNCDREAMER_SELECTED_INDICES}")
    print(f"[SyncDreamer] Azimuths SyncDreamer sélectionnés : {SYNCDREAMER_AZIMUTHS[SYNCDREAMER_SELECTED_INDICES]}")
    print(f"[SyncDreamer] Azimuths cibles InstantMesh       : {INSTANTMESH_TARGET_AZIMUTHS}")

    selected_resized = []
    for idx in SYNCDREAMER_SELECTED_INDICES:
        img = Image.fromarray(views_array[idx])  # uint8 RGB

        # Détourage + fond blanc pour correspondre à la convention Zero123++
        if rembg_session is not None:
            img = img.convert("RGBA")
            img = rembg.remove(img, session=rembg_session)
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background.convert("RGB")

        img = img.resize((320, 320), Image.LANCZOS)
        selected_resized.append(np.asarray(img, dtype=np.float32) / 255.0)

    selected_resized = np.stack(selected_resized, axis=0)           # [6, 320, 320, 3]
    return torch.from_numpy(selected_resized).permute(0, 3, 1, 2).float()  # [6, 3, 320, 320]


def load_syncdreamer():
    """
    Charge le modèle SyncDreamer depuis le checkpoint local.
    Le state_dict est chargé sur CPU pour économiser la VRAM,
    puis le modèle est déplacé sur GPU après instanciation.
    """
    try:
        cfg  = f"{syncdreamer_root}/configs/syncdreamer.yaml"
        ckpt = f"{syncdreamer_root}/ckpt/syncdreamer-pretrain.ckpt"
        with cd(syncdreamer_root):
            config     = OmegaConf.load(cfg)
            state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
            model      = instantiate_from_config(config.model)
            model.load_state_dict(state_dict, strict=True)
            model      = model.cuda().eval()
        return model
    except ImportError as e:
        raise ImportError(
            "SyncDreamer non installé. "
            "Cloner https://github.com/liuyuan-pal/SyncDreamer "
            "et placer le ckpt dans ckpt/syncdreamer-pretrain.ckpt"
        ) from e


###############################################################################
# Arguments
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config',       type=str,            help='Path to config file.')
parser.add_argument('input_path',   type=str,            help='Path to input image or directory.')
parser.add_argument('--output_path',    type=str,   default='outputs/',  help='Output directory.')
parser.add_argument('--diffusion_steps',type=int,   default=75,          help='Denoising Sampling steps.')
parser.add_argument('--seed',           type=int,   default=42,          help='Random seed for sampling.')
parser.add_argument('--scale',          type=float, default=1.0,         help='Scale of generated object.')
parser.add_argument('--distance',       type=float, default=4.5,         help='Render distance.')
parser.add_argument('--view',           type=int,   default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg',       action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap',  action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video',     action='store_true', help='Save a circular-view video.')
parser.add_argument(
    '--diffusion_model',
    type=str,
    default='zero123plus_finetuned',
    choices=['zero123plus_finetuned', 'zero123plus_base', 'syncdreamer'],
    help='Quel modèle de diffusion utiliser'
)
args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0 : Configuration
###############################################################################
config      = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = config_name.startswith('instant-mesh')
device        = torch.device('cuda')

print(f'[Diffusion] Modèle choisi : {args.diffusion_model}')

if args.diffusion_model in ['zero123plus_finetuned', 'zero123plus_base']:
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    if args.diffusion_model == 'zero123plus_finetuned':
        print('[Diffusion] Chargement du UNet fine-tuné InstantMesh (fond blanc)...')
        unet_ckpt_path = infer_config.unet_path if os.path.exists(infer_config.unet_path) else \
            hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        pipeline.unet.load_state_dict(state_dict, strict=True)
        print('[Diffusion] UNet fine-tuné chargé ✓')
    else:
        print('[Diffusion] OPTION A : UNet de base Zero123++ v1.2 (sans fine-tuning)')
    pipeline = pipeline.to(device)
    syncdreamer_model = None

elif args.diffusion_model == 'syncdreamer':
    print('[Diffusion] OPTION B : Chargement de SyncDreamer...')
    pipeline          = None
    syncdreamer_model = load_syncdreamer()
    print('[Diffusion] SyncDreamer chargé ✓')

# Dossiers de sortie
output_subfolder = f"{config_name}_{args.diffusion_model}"
image_path = os.path.join(args.output_path, output_subfolder, 'images')
mesh_path  = os.path.join(args.output_path, output_subfolder, 'meshes')
video_path = os.path.join(args.output_path, output_subfolder, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path,  exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# Liste des fichiers d'entrée
if os.path.isdir(args.input_path):
    input_files = [
        os.path.join(args.input_path, f)
        for f in os.listdir(args.input_path)
        if f.endswith(('.png', '.jpg', '.webp'))
    ]
else:
    input_files = [args.input_path]

print(f'Total input images: {len(input_files)}')

###############################################################################
# Stage 1 : Génération multi-vues
###############################################################################
rembg_session = None if args.no_rembg else rembg.new_session()
outputs = []

for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).split('.')[0]
    print(f'\n[{idx+1}/{len(input_files)}] Imagining {name} ... (mode: {args.diffusion_model})')

    input_image = Image.open(image_file)
    if not args.no_rembg:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    # ------------------------------------------------------------------
    # OPTION A — Zero123++ (fine-tuné ou base)
    # ------------------------------------------------------------------
    if args.diffusion_model in ['zero123plus_finetuned', 'zero123plus_base']:
        output_image = pipeline(
            input_image,
            num_inference_steps=args.diffusion_steps,
        ).images[0]
        output_image.save(os.path.join(image_path, f'{name}.png'))

        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)

    # ------------------------------------------------------------------
    # OPTION B — SyncDreamer
    # ------------------------------------------------------------------
    elif args.diffusion_model == 'syncdreamer':
        with cd(syncdreamer_root), torch.no_grad():
            data = prepare_inputs(image_file, elevation_input=30)
            for k, v in data.items():
                data[k] = v.unsqueeze(0).cuda()
                data[k] = torch.repeat_interleave(data[k], 1, dim=0)

            # 20 steps suffisent pour SyncDreamer (50 = trop lent / OOM sur Colab)
            sampler = SyncDDIMSampler(syncdreamer_model, 20)

            print(f"[SyncDreamer] VRAM avant sample : {torch.cuda.memory_allocated() / 1e9:.2f} Go")
            x_sample = syncdreamer_model.sample(
                sampler,
                data,
                cfg_scale=1.5,   # 2.0 double la charge mémoire inutilement
                batch_view_num=1,
            )
            print(f"[SyncDreamer] Échantillonnage terminé ✓")

            # Dénormalisation [-1,1] → [0,1] → uint8
            x_sample = (torch.clamp(x_sample, -1.0, 1.0) + 1) * 0.5
            x_sample = (x_sample.permute(0, 1, 3, 4, 2).cpu().numpy() * 255).astype(np.uint8)
            # x_sample : [B, 16, H, W, 3]

            # Sauvegarde de la grille 4×4 pour inspection visuelle
            rows = [
                np.concatenate([x_sample[0, r*4 + c] for c in range(4)], axis=1)
                for r in range(4)
            ]
            output_grid = Image.fromarray(np.concatenate(rows, axis=0))
            output_grid.save(os.path.join(image_path, f'{name}_syncdreamer_grid.png'))
            print(f"[SyncDreamer] Grille 4×4 sauvegardée ✓")

            # Sélection des 6 vues + détourage fond blanc (obligatoire pour InstantMesh)
            # rembg_session=None désactivera le détourage si --no_rembg est passé
            images = select_syncdreamer_views(x_sample[0], rembg_session=rembg_session)
            print(f'[SyncDreamer] 6 vues sélectionnées et préparées ✓')

        # Nettoyage VRAM immédiat après le sampling
        try:
            del data, sampler, x_sample, output_grid
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[SyncDreamer] VRAM après nettoyage : {torch.cuda.memory_allocated() / 1e9:.2f} Go")

    outputs.append({'name': name, 'images': images})

# Libération du modèle de diffusion avant la reconstruction
if pipeline is not None:
    del pipeline
if syncdreamer_model is not None:
    del syncdreamer_model
gc.collect()
torch.cuda.empty_cache()

###############################################################################
# Stage 2 : Reconstruction 3D
###############################################################################

# Caméras Zero123++ (6 vues à [30, 90, 150, 210, 270, 330]°)
# Utilisées pour Zero123++ ET SyncDreamer : InstantMesh a été entraîné avec
# ces poses exactes. Les vues SyncDreamer sont sélectionnées aux azimuths
# les plus proches de ces mêmes cibles → cohérence garantie.
input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * args.scale).to(device)
chunk_size    = 20 if IS_FLEXICUBES else 1

print('\nLoading reconstruction model ...')
model = instantiate_from_config(model_config)
model_ckpt_path = infer_config.model_path if os.path.exists(infer_config.model_path) else \
    hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename=f"{config_name.replace('-', '_')}.ckpt",
        repo_type="model"
    )
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)
model = model.to(device)

if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'\n[{idx+1}/{len(outputs)}] Creating mesh for {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    # Sélection des caméras d'entrée selon le mode
    if args.diffusion_model == 'syncdreamer':
        # InstantMesh a été entraîné avec les caméras Zero123++ → on les réutilise
        # directement. Les 6 vues SyncDreamer ont été sélectionnées aux azimuths
        # les plus proches des 6 cibles Zero123++ (30/90/150/210/270/330°),
        # ce qui rend cette association cohérente.
        input_cameras_view = input_cameras
    elif args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras_view = input_cameras[:, indices]
    else:
        input_cameras_view = input_cameras

    with torch.no_grad():
        planes = model.forward_planes(images, input_cameras_view)

        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )

        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)

        print(f"Mesh saved to {mesh_path_idx}")

        if args.save_video:
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.render_resolution

            # Élévation de rendu adaptée au modèle de diffusion :
            # - SyncDreamer génère ses vues à 30° → on rend à 30° pour cohérence visuelle
            # - Zero123++ utilise 20° (convention d'entraînement InstantMesh)
            render_elevation = 30.0 if args.diffusion_model == 'syncdreamer' else 20.0

            render_cameras = get_render_cameras(
                batch_size=1, M=120, radius=args.distance,
                elevation=render_elevation,
                is_flexicubes=IS_FLEXICUBES,
            ).to(device)
            frames = render_frames(
                model, planes,
                render_cameras=render_cameras,
                render_size=render_size,
                chunk_size=chunk_size,
                is_flexicubes=IS_FLEXICUBES,
            )
            save_video(frames, video_path_idx, fps=30)
            print(f"Video saved to {video_path_idx}")

    # Nettoyage VRAM après chaque objet
    try:
        del planes, mesh_out, images
        if args.save_video:
            del frames, render_cameras
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Debug] VRAM après mesh {name} : {torch.cuda.memory_allocated() / 1e9:.2f} Go")
