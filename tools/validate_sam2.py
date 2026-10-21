#!/usr/bin/env python3
# Copyright (c) 2026 PickNik Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Compare all three exported graphs against upstream PyTorch SAM2.1."""

import argparse
import hashlib
import json
from pathlib import Path

from export_sam2 import FILENAMES, IMAGE_OUTPUTS, load_model, PROMPT_OUTPUTS, sha256
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch

# Absolute tolerances for CPU FP32. IoU is measured at logit zero; both empty is 1.
LIMITS = {
    'features_max_abs': 0.01,
    'prompts_max_abs': 0.0001,
    'logits_max_abs': 0.01,
    'logits_mean_abs': 0.001,
    'scores_max_abs': 0.0001,
    'mask_iou_min': 0.999,
}


def fixtures():
    for width, height in [(768, 768), (640, 480), (480, 640)]:
        image = Image.new('RGB', (width, height), (42, 62, 87))
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (width // 5, height // 5, width * 3 // 5, height * 4 // 5),
            fill=(220, 88, 44),
            outline=(255, 230, 140),
            width=5,
        )
        draw.rectangle(
            (width * 3 // 5, height // 4, width * 9 // 10, height * 3 // 4),
            fill=(44, 180, 130),
        )
        yield f'synthetic-{width}x{height}', np.asarray(image), None


def max_abs(actual, expected):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    return float(np.max(np.abs(actual - expected)))


def validate(checkpoint, model_dir, images, report):
    torch.set_num_threads(8)
    torch.manual_seed(0)
    model = load_model(checkpoint)
    predictor = SAM2ImagePredictor(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    sessions = {
        name: ort.InferenceSession(
            str(model_dir / file), options, providers=['CPUExecutionProvider']
        )
        for name, file in FILENAMES.items()
    }
    result = {
        'limits': LIMITS,
        'torch': torch.__version__,
        'onnxruntime': ort.__version__,
        'provider': 'CPUExecutionProvider',
        'checkpoint_sha256': sha256(checkpoint),
        'artifact_sha256': {f: sha256(model_dir / f) for f in FILENAMES.values()},
        'fixtures': [],
        'cases': [],
    }
    inputs = list(fixtures())
    for path in images:
        inputs.append(
            (
                f'local-{sha256(path)[:12]}',
                np.asarray(Image.open(path).convert('RGB')),
                sha256(path),
            )
        )
    coords = torch.tensor(
        [
            [
                [384.0, 384.0],
                [480.0, 500.0],
                [100.0, 100.0],
                [400.0, 600.0],
                [350.0, 440.0],
                [50.0, 900.0],
                [750.0, 150.0],
                [440.0, 350.0],
            ]
        ]
    )
    labels = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]])
    with torch.inference_mode():
        for name, image, digest in inputs:
            predictor.set_image(image)
            reference_features = [
                predictor._features['image_embed'],
                *predictor._features['high_res_feats'],
            ]
            encoder_input = predictor._transforms(image).unsqueeze(0).numpy()
            actual_features = sessions['image_encoder'].run(
                None, {'input_image': encoder_input}
            )
            feature_error = max(
                max_abs(a, b.numpy())
                for a, b in zip(actual_features, reference_features)
            )
            assert feature_error <= LIMITS['features_max_abs'], feature_error
            result['fixtures'].append(
                {
                    'name': name,
                    'shape': list(image.shape),
                    'image_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                    'file_sha256': digest,
                    'features_max_abs': feature_error,
                }
            )
            prior = None
            for count in [1, 2, 3, 8]:
                for with_prior in [False, True]:
                    # Refinement uses the upstream singleton logits as the same prior
                    # in both paths.
                    mask = prior if with_prior else None
                    sparse, dense = model.sam_prompt_encoder(
                        points=(coords[:, :count], labels[:, :count]),
                        boxes=None,
                        masks=mask,
                    )
                    expected_prompts = [
                        sparse,
                        dense,
                        model.sam_prompt_encoder.get_dense_pe(),
                    ]
                    # Deliberately nonzero when disabled to verify the gate ignores mask_input.
                    mask_input = (
                        mask if mask is not None else torch.full((1, 1, 256, 256), 7.0)
                    )
                    actual_prompts = sessions['prompt_encoder'].run(
                        None,
                        {
                            'point_coords': coords[:, :count].numpy(),
                            'point_labels': labels[:, :count].numpy(),
                            'mask_input': mask_input.numpy(),
                            'has_mask_input': np.array(
                                [float(with_prior)], dtype=np.float32
                            ),
                        },
                    )
                    prompt_error = max(
                        max_abs(a, b.numpy())
                        for a, b in zip(actual_prompts, expected_prompts)
                    )
                    assert prompt_error <= LIMITS['prompts_max_abs'], prompt_error
                    ref_masks, ref_scores, _, _ = model.sam_mask_decoder(
                        image_embeddings=reference_features[0],
                        image_pe=expected_prompts[2],
                        sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=dense,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=reference_features[1:],
                    )
                    if prior is None:
                        prior = ref_masks.clamp(-32, 32)
                    feed = dict(
                        zip(
                            IMAGE_OUTPUTS + PROMPT_OUTPUTS,
                            actual_features + actual_prompts,
                        )
                    )
                    masks, scores = sessions['decoder'].run(None, feed)
                    assert masks.shape == (1, 1, 256, 256) and scores.shape == (1, 1)
                    expected = ref_masks.numpy()
                    error = max_abs(masks, expected)
                    mean_error = float(np.mean(np.abs(masks - expected)))
                    score_error = max_abs(scores, ref_scores.numpy())
                    union = np.logical_or(masks > 0, expected > 0).sum()
                    iou = (
                        float(np.logical_and(masks > 0, expected > 0).sum() / union)
                        if union
                        else 1.0
                    )
                    case = {
                        'image': name,
                        'points': count,
                        'labels': labels[0, :count].tolist(),
                        'with_prior': with_prior,
                        'prompts_max_abs': prompt_error,
                        'logits_max_abs': error,
                        'logits_mean_abs': mean_error,
                        'scores_max_abs': score_error,
                        'mask_iou': iou,
                    }
                    print(json.dumps(case), flush=True)
                    result['cases'].append(case)
                    assert error <= LIMITS['logits_max_abs'], case
                    assert mean_error <= LIMITS['logits_mean_abs'], case
                    assert score_error <= LIMITS['scores_max_abs'], case
                    assert iou >= LIMITS['mask_iou_min'], case
    result['passed'] = len(result['cases'])
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(
        f"Passed {result['passed']} end-to-end prompt cases; report: {report}",
        flush=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path('models'))
    parser.add_argument('--image', type=Path, action='append', default=[])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    validate(args.checkpoint, args.model_dir, args.image, args.report)
