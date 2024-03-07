import argparse
import os
import glob
import shutil
import json
from pathlib import Path
from mlx_lm.tuner.utils import dequantize
from mlx_lm.utils import fetch_from_hub, get_model_path, save_weights
from mlx.utils import tree_flatten

def build_parser():
  parser = argparse.ArgumentParser(description='Dequantize model')
  parser.add_argument('--model', type=str, required=True, help='Path to the model')
  parser.add_argument('--output', type=str, required=True, help='Path to the output model')
  return parser

def main(args):
  model_path = get_model_path(args.model)
  model, config, tokenizer = fetch_from_hub(model_path)

  model.freeze()
  model = dequantize(model)

  weights = dict(tree_flatten(model.parameters()))
  save_path = Path(args.output)
  save_weights(save_path, weights)

  py_files = glob.glob(str(model_path / '*.py'))
  for py_file in py_files:
    shutil.copy(py_file, save_path)

  tokenizer.save_pretrained(save_path)
  config.pop("quantization", None)

  with open(save_path / 'config.json', 'w') as f:
    json.dump(config, f, indent=4)

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
