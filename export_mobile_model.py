"""Export DocumentRestorerNet checkpoint to ONNX for mobile inference."""

import argparse
import json
from pathlib import Path

import torch

from backend.models.document_restorer import DocumentRestorerNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--opset', type=int, default=18)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint
    model = DocumentRestorerNet(base_channels=args.base_channels).eval()
    model.load_state_dict(state_dict, strict=True)
    sample = torch.zeros(1, 3, 512, 512)
    temporary_output = output_path.with_name(f'.{output_path.name}.export.onnx')
    torch.onnx.export(
        model,
        sample,
        temporary_output,
        input_names=['image'],
        output_names=['restored', 'shadow_mask'],
        dynamic_axes={
            'image': {0: 'batch', 2: 'height', 3: 'width'},
            'restored': {0: 'batch', 2: 'height', 3: 'width'},
            'shadow_mask': {0: 'batch', 2: 'height', 3: 'width'},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    import onnx
    exported = onnx.load(temporary_output, load_external_data=True)
    onnx.save_model(exported, output_path, save_as_external_data=False)
    temporary_output.unlink(missing_ok=True)
    temporary_output.with_suffix(temporary_output.suffix + '.data').unlink(missing_ok=True)
    metadata = {
        'format': 'onnx',
        'checkpoint': str(checkpoint_path),
        'input': {'name': 'image', 'layout': 'NCHW', 'dtype': 'float32', 'range': [0.0, 1.0], 'color': 'RGB'},
        'outputs': [
            {'name': 'restored', 'layout': 'NCHW', 'dtype': 'float32', 'range': [0.0, 1.0]},
            {'name': 'shadow_mask', 'layout': 'NCHW', 'dtype': 'float32', 'range': [0.0, 1.0]},
        ],
        'spatial_requirement': 'height and width may be dynamic; model pads internally to multiples of 4',
        'recommended_mobile_runtime': 'onnxruntime',
        'base_channels': args.base_channels,
    }
    output_path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'model': str(output_path), 'metadata': str(output_path.with_suffix('.json'))}, indent=2))


if __name__ == '__main__':
    main()
