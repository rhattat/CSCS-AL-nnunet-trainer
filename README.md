# CSCS-AL nnU-Net trainer

Custom nnU-Net v2 trainer used for the CSCS-AL experiments: a Ranger22
optimizer/scheduler and a provenance-weighted loss for training on
partially-corrected volumes. See the companion
[CSCS-AL-partial](https://github.com/rhattat/CSCS-AL-partial) repository for
the acquisition and correction logic that produces the provenance weight
maps this trainer consumes.

## What's here

```
nnunetv2/training/
├── optimizer/ranger22_optimizer.py              Ranger22 optimizer
├── lr_scheduler/ranger22_scheduler.py            Ranger22 warmup/warmdown scheduler
├── nnUNetTrainer/variants/optimizer/
│   ├── nnUNetTrainerRanger.py                    plain Ranger22 trainer
│   └── nnUNetTrainerRangerProvenance.py          + provenance-weighted loss
└── loss/provenance_weighted_loss.py              the weighted Dice+CE loss itself
```

- `ranger22_optimizer.py` / `ranger22_scheduler.py` are Ranger21
  (https://github.com/lessw2020/Ranger21, Apache 2.0) copied in largely
  unmodified, so training doesn't depend on the separate `ranger21` package.
- `nnUNetTrainerRanger.py`, `nnUNetTrainerRangerProvenance.py` and
  `provenance_weighted_loss.py` are original nnU-Net trainer/loss
  extensions for this project.

## Install

Requires a working nnU-Net v2 install (`pip install nnunetv2`). Copy the
files under `nnunetv2/` into the matching paths of your own nnU-Net v2
installation (or clone this repo somewhere on your `PYTHONPATH` next to
your `nnunetv2` package). nnU-Net discovers trainers by class name, so
once the files are in place you can pass e.g. `-tr
nnUNetTrainerRangerProvenance_250epochs` to `nnUNetv2_train` like any
built-in trainer.

## Usage

Plain training (E1, full annotation), no provenance weights:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerRanger_noSmooth_250epochs
```

Provenance-weighted training (E2 / E2-OPT): inject a second target channel
holding the per-voxel weight (0-1000, see `provenance_weighted_loss.py`)
before preprocessing, then train with:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerRangerProvenance_250epochs
```

The loss detects a 1- vs 2-channel target on the first training batch and
switches between plain and weighted mode automatically -- you don't need a
different trainer class for cold-start (fully annotated) volumes.

## References

This trainer is a derivative of nnU-Net (Apache 2.0, copyright DKFZ -- see
the license headers in each modified file) and bundles an adapted copy of
Ranger21 (Apache 2.0). If you use this code, please cite both alongside
the CSCS-AL paper:

```bibtex
@article{isensee2021nnunet,
  title   = {nnU-Net: a self-configuring method for deep learning-based
             biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and
             Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021},
}

@article{wright2021ranger21,
  title   = {Ranger21: a synergistic deep learning optimizer},
  author  = {Wright, Less and Demeure, Nestor},
  journal = {arXiv preprint arXiv:2106.13731},
  year    = {2021},
}
```

nnU-Net: https://github.com/MIC-DKFZ/nnUNet · Ranger21: https://github.com/lessw2020/Ranger21

## License

Apache 2.0 -- see `LICENSE`. This mirrors nnU-Net's own license and
Ranger21's license.
