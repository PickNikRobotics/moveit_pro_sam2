# moveit_pro_sam2

ROS package containing SAM2.1 ONNX models for image segmentation.

## Usage

Submodule this repository into a user workspace to make the models available in the `moveit_pro_sam2` package for use by MoveIt Pro Behaviors.

## SAM2.1 normal bundle

The three-graph SAM2.1 Hiera-L bundle in
[`models/sam2_bundle.yaml`](models/sam2_bundle.yaml) supports joint labeled
points and an optional prior mask, with high-resolution image features and
single-mask logits. See the [tensor contract](models/README.md),
[export instructions](tools/README.md),
[provenance](models/sam2.1_provenance.json) and
[numerical validation](models/sam2.1_validation.json).

Qualcomm AI Hub fused exports use a separate graph contract.

## Model license

The ONNX model files in [`models/`](models/) are exports of Meta's SAM 2 model and are distributed under the [Apache License 2.0](LICENSE), with upstream attribution in [NOTICE](NOTICE). If you redistribute the models or derivative works, include the license text and the NOTICE file with them.

## Package licensing

The ONNX model files in `models/`, and derivative works of those models, are governed by the [Apache License 2.0](LICENSE) with the attribution in [NOTICE](NOTICE). All other files in this repository are governed by the [BSD 3-Clause License](LICENSE.build), unless a file states otherwise.
