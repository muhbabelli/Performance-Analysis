import argparse
from segmentation_module import SegmentationModel
from data_module import SemanticSegmentationDataModule
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
import torch


def main(args):
    torch.set_float32_matmul_precision('high')

    model = SegmentationModel.load_from_checkpoint(args.ckpt)
    model.hparams.DATA.DATASET_ROOT = args.cityscapes_root
    datamodule = SemanticSegmentationDataModule(model.hparams)

    devices = -1
    if args.devices is not None:
        devices = [int(d) for d in args.devices.split(",")]
    output = Trainer(accelerator='gpu', precision= 32, devices=devices).test(model, datamodule=datamodule)
    

    return edict(output[0])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cityscapes Metrics")

    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the model checkpoint"
    )

    parser.add_argument(
        "--cityscapes_root",
        type=str,
        help="Path to the cityscapes dataset",
        default="/datasets/cityscapes/"
    )

    parser.add_argument(
        "--devices",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    main(args)