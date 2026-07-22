# moveit_pro_sam2

ROS package containing SAM 2 ONNX models for image segmentation.

## Usage

Submodule this repository into a user workspace to make the models available in the `moveit_pro_sam2` package for use by MoveIt Pro Behaviors.

## Model license

The ONNX model files in [`models/`](models/) are exports of Meta's SAM 2 model and are distributed under the [Apache License 2.0](LICENSE), with upstream attribution in [NOTICE](NOTICE). If you redistribute the models or derivative works, include the license text and the NOTICE file with them.

## Provenance

The files in `models/` were exported by PickNik in December 2024 from Meta's SAM 2 `sam2_hiera_large` checkpoint ([facebookresearch/sam2](https://github.com/facebookresearch/sam2)) using `torch.onnx.export` (PyTorch 2.5.1, per the ONNX producer metadata). No training or fine-tuning was performed. The export scripts and the exact checkpoint revision were not preserved. The same artifacts were first committed to [moveit_pro_example_ws](https://github.com/PickNikRobotics/moveit_pro_example_ws) in December 2024 (PR #23, "Add example SAM 2 behavior and objective") and moved here unchanged.

| File | SHA-256 |
| --- | --- |
| `sam2_hiera_large_encoder.onnx` | `c99ab89a38385753aff7ea9155f0808ad5535bc55ea2a49320254e39e4011630` |
| `sam2_decoder.onnx` | `1f448cdb479e6ec14e61c4756138eb4081ce7f8a11ca43a0a24856d5e8b61b6f` |

## Package licensing

The ONNX model files in `models/`, and derivative works of those models, are governed by the [Apache License 2.0](LICENSE) with the attribution in [NOTICE](NOTICE). All other files in this repository are governed by the [BSD 3-Clause License](LICENSE.build), unless a file states otherwise.
