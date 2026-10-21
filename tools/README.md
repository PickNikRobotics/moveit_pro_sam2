# SAM2.1 export and validation

Use Python 3.12 and the pinned packages in `export-environment.txt`, which
records the complete resolved Linux export environment. Obtain the upstream
source at revision `2b90b9f5ceec907a1c18123530e92e794ad901a4` from
[facebookresearch/sam2](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4)
and install it with `SAM2_BUILD_CUDA=0 pip install --no-deps -e /path/to/sam2`.
The exporter and validator reject an imported `sam2` package whose source
checkout is not at that git revision or has uncommitted tracked changes, and
checkpoints whose SHA-256 differs from the pinned value.
Download `sam2.1_hiera_large.pt` from the checkpoint URL in
[`sam2.1_provenance.json`](../models/sam2.1_provenance.json).

```sh
python -m pip install -r tools/export-environment.txt
CUDA_VISIBLE_DEVICES='' python tools/export_sam2.py --checkpoint /path/to/sam2.1_hiera_large.pt --output-dir models
CUDA_VISIBLE_DEVICES='' python tools/validate_sam2.py --checkpoint /path/to/sam2.1_hiera_large.pt \
  --model-dir models --report /tmp/sam2-validation.json
```

Copy the validation report to `models/sam2.1_validation.json` and record its
SHA-256 under `validation_report` in `models/sam2.1_provenance.json`.

`--image /path/to/image` may be repeated to extend the three deterministic
synthetic fixtures with local RGB images. Neither source checkpoints nor test
images are published by these scripts. CPU execution uses eight threads. Hiding
CUDA devices also prevents upstream positional-encoding initialization from
warming a GPU cache, even though the model is requested on CPU.

The exporter uses the unmodified upstream model with `apply_postprocessing=False`,
FP32, ONNX opset 17 and the legacy PyTorch exporter (`dynamo=False`). Dynamic axes
are specified before tracing the prompt encoder and decoder, so point count
propagates through attention reshapes. The trace uses three points; validation
executes 1, 2, 3 and 8 points. The exported decoder selects the single-mask token;
it does not apply a sigmoid, threshold, output resize, stability-based multimask
fallback, or hole filling. The checkpoint is neither trained nor fine-tuned.

The validator uses upstream `SAM2ImagePredictor.set_image` as the feature and
preprocessing reference, followed by upstream prompt encoding and mask decoding.
Each ONNX decoder consumes ONNX encoder outputs, testing accumulated numerical
error across all three graphs. Refinement uses the same clamped upstream
singleton logits as prior input in both paths. Disabled-mask cases deliberately
supply a nonzero mask. Comparisons cover feature tensors, sparse/dense prompt
embeddings, positional encoding, raw mask logits, quality scores and binary IoU
at logit zero. See `LIMITS` in the validator and the recorded validation report.

This is graph-level numerical qualification. Application-level postprocessing,
auto-mask filtering, cancellation, concurrency and accelerator qualification
require separate integration tests.
