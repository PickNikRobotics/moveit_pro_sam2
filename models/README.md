# SAM2.1 Hiera-L normal bundle

[`sam2_bundle.yaml`](sam2_bundle.yaml) identifies model `sam2`, version
`sam2.1_hiera_l`, graph ABI `moveit_pro.sam2.v1`. Paths are relative to the
manifest, and each graph has a SHA-256 digest. The three components use ONNX
Runtime and may share sessions between point segmentation and automatic mask
generation. All tensors are float32, with batch size fixed at one.

| Component | Inputs | Outputs |
| --- | --- | --- |
| `image_encoder` | `input_image [1,3,1024,1024]` | `image_embeddings [1,256,64,64]`, `high_res_feats_0 [1,32,256,256]`, `high_res_feats_1 [1,64,128,128]` |
| `prompt_encoder` | `point_coords [1,N,2]`, `point_labels [1,N]`, `mask_input [1,1,256,256]`, `has_mask_input [1]` | `sparse_embeddings [1,N+1,256]`, `dense_embeddings [1,256,64,64]`, `image_pe [1,256,64,64]` |
| `decoder` | All six named outputs of the two encoders | `masks [1,1,256,256]`, `scores [1,1]` |

## Image preprocessing

Convert native RGB bytes to float32 in [0,1], resize directly to 1024 by 1024
(no padding or aspect-ratio preservation), and normalize channels with mean
`[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
Use NCHW layout. The upstream reference uses torchvision bilinear tensor resize,
`align_corners=False`, `antialias=True`. Preprocessing is outside the graph.

The image encoder includes the two high-resolution decoder projections and adds
the learned no-memory embedding to the 64 by 64 image embedding, matching image
prediction in upstream SAM2.1. Encode once per image and reuse all three outputs
for every prompt on that image.

## Prompts and outputs

`N >= 1` labeled points describe one object jointly. Coordinates are `(x,y)` in
1024-space. Scale native coordinates by `1024/width` and `1024/height`; the prompt
encoder adds the upstream half-pixel offset. Labels are `1` for foreground, `0`
for background, and `-1` for padding. The encoder appends one additional
not-a-point token, as upstream does when no box is supplied. Boxes and batched
independent objects are outside this graph contract.

Set `has_mask_input` to `0` without a prior or `1` with a prior. Always supply
`mask_input`; it is ignored when disabled. A prior is low-resolution **logits**,
not a binary mask or probabilities. Upstream iterative refinement clamps previous
low-resolution logits to [-32,32]. Point count remains dynamic in both prompt
encoding and decoder attention, rather than decoding each point independently.

`masks` contains single-mask-token logits; no sigmoid, threshold or native-size
resize is included. `scores` contains that mask's predicted IoU quality score.
A consumer requiring probabilities applies sigmoid; a consumer requiring
native-size logits bilinearly resizes the logits. Neither automatic mask
postprocessing nor stability-based multimask fallback is included.

## Reproduction and qualification

[`sam2.1_provenance.json`](sam2.1_provenance.json) records the source revision,
checkpoint identity, export tools, script digest, artifact digests and
validation-report digest.
[`tools/`](../tools/README.md) contains the reproducible export and numerical
comparison commands. [`sam2.1_validation.json`](sam2.1_validation.json) records
CPU FP32 results and tolerances. All 40 cases passed: three generated images
(square, landscape and portrait) and two local camera images, each tested with
1, 2, 3 and 8 points, with and without a prior. Camera images are identified by
SHA-256 in the report and are not redistributed.

| Metric | Worst observed | Limit |
| --- | --- | --- |
| Image-feature absolute error | 0.003365 | 0.01 |
| Prompt-feature absolute error | 0.000001506 | 0.0001 |
| Mask-logit absolute error | 0.003125 | 0.01 |
| Mask-logit mean absolute error | 0.00003087 | 0.001 |
| Score absolute error | 0.000001073 | 0.0001 |
| Binary mask IoU at logit zero | 0.99991 | >= 0.999 |

These results qualify graph agreement with the
PyTorch reference; application and accelerator qualification are separate.

Qualcomm AI Hub fused graphs have a separate graph ABI and remain unchanged by
this export.
