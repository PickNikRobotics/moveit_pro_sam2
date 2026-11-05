#!/usr/bin/env python3
# Copyright (c) 2026 PickNik Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Export the normal SAM2.1 Hiera-L image/prompt/decoder graphs (opset 17)."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import onnx
import sam2
from sam2.build_sam import build_sam2
import torch

SOURCE_REVISION = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
CHECKPOINT_SHA256 = '2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'
CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
IMAGE_OUTPUTS = ['image_embeddings', 'high_res_feats_0', 'high_res_feats_1']
PROMPT_OUTPUTS = ['sparse_embeddings', 'dense_embeddings', 'image_pe']
FILENAMES = {
    'image_encoder': 'sam2.1_hiera_l_image_encoder.onnx',
    'prompt_encoder': 'sam2.1_prompt_encoder.onnx',
    'decoder': 'sam2.1_decoder.onnx',
}


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_source_revision():
    source = Path(sam2.__file__).parent
    revision = subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f'{source} is not a sam2 checkout at {SOURCE_REVISION}')
    dirty = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(f'{source} has uncommitted tracked changes')


def load_model(checkpoint):
    verify_source_revision()
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint does not match the pinned SAM2.1 Hiera-L weights')
    return build_sam2(CONFIG, str(checkpoint), device='cpu', apply_postprocessing=False)


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_image):
        # forward_image includes the two decoder projections used by the predictor.
        features = self.model.forward_image(input_image)['backbone_fpn']
        embedding = features[-1] + self.model.no_mem_embed.reshape(1, 256, 1, 1)
        return embedding, features[0], features[1]


class PromptEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.sam_prompt_encoder

    def forward(self, point_coords, point_labels, mask_input, has_mask_input):
        sparse = self.encoder._embed_points(point_coords, point_labels, pad=True)
        with_mask = self.encoder._embed_masks(mask_input)
        without_mask = self.encoder.no_mask_embed.weight.reshape(1, 256, 1, 1)
        dense = torch.where(
            has_mask_input.reshape(1, 1, 1, 1) > 0, with_mask, without_mask
        )
        return sparse, dense, self.encoder.get_dense_pe()


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.sam_mask_decoder

    def forward(
        self,
        image_embeddings,
        high_res_feats_0,
        high_res_feats_1,
        sparse_embeddings,
        dense_embeddings,
        image_pe,
    ):
        masks, scores, _, _ = self.decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[high_res_feats_0, high_res_feats_1],
        )
        return masks, scores


def export(checkpoint, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    model = load_model(checkpoint)
    image_encoder = ImageEncoder(model).eval()
    prompt_encoder = PromptEncoder(model).eval()
    decoder = Decoder(model).eval()
    image = torch.zeros(1, 3, 1024, 1024)
    prompt = (
        torch.tensor([[[300.0, 300.0], [500.0, 500.0], [800.0, 800.0]]]),
        torch.tensor([[1.0, 1.0, 0.0]]),
        torch.zeros(1, 1, 256, 256),
        torch.zeros(1),
    )
    with torch.inference_mode():
        image_features = image_encoder(image)
        prompt_features = prompt_encoder(*prompt)
        specs = [
            (
                'image_encoder',
                image_encoder,
                (image,),
                ['input_image'],
                IMAGE_OUTPUTS,
                {},
            ),
            (
                'prompt_encoder',
                prompt_encoder,
                prompt,
                ['point_coords', 'point_labels', 'mask_input', 'has_mask_input'],
                PROMPT_OUTPUTS,
                {
                    'point_coords': {1: 'num_points'},
                    'point_labels': {1: 'num_points'},
                    'sparse_embeddings': {1: 'num_tokens'},
                },
            ),
            (
                'decoder',
                decoder,
                (*image_features, *prompt_features),
                IMAGE_OUTPUTS + PROMPT_OUTPUTS,
                ['masks', 'scores'],
                {'sparse_embeddings': {1: 'num_tokens'}},
            ),
        ]
        for component, module, inputs, input_names, output_names, dynamic_axes in specs:
            path = output_dir / FILENAMES[component]
            print(f'Exporting {component}: {path}', flush=True)
            torch.onnx.export(
                module,
                inputs,
                str(path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                dynamo=False,
            )
            onnx.checker.check_model(str(path), full_check=True)
    artifacts = {
        name: {'path': file, 'sha256': sha256(output_dir / file)}
        for name, file in FILENAMES.items()
    }
    manifest = {
        'schema_version': 1,
        'model_id': 'sam2',
        'model_version': 'sam2.1_hiera_l',
        'graph_abi': 'moveit_pro.sam2.v1',
        'runtimes': {'onnxruntime': artifacts},
    }
    # JSON is valid YAML, but emit YAML for consistency with the bundle consumer.
    import yaml

    (output_dir / 'sam2_bundle.yaml').write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    provenance = {
        'source_repository': 'https://github.com/facebookresearch/sam2',
        'source_revision': SOURCE_REVISION,
        'config': CONFIG,
        'checkpoint_url': CHECKPOINT_URL,
        'checkpoint_sha256': sha256(checkpoint),
        'torch': torch.__version__,
        'onnx': onnx.__version__,
        'python': platform.python_version(),
        'package_versions': {
            name: importlib.metadata.version(name)
            for name in (
                'torch',
                'torchvision',
                'onnx',
                'onnxruntime',
                'numpy',
                'hydra-core',
                'iopath',
                'Pillow',
                'tqdm',
            )
        },
        'opset': 17,
        'precision': 'float32',
        'apply_postprocessing': False,
        'export_script': 'tools/export_sam2.py',
        'export_script_sha256': sha256(Path(__file__)),
        'artifacts': artifacts,
    }
    (output_dir / 'sam2.1_provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('models'))
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir)
