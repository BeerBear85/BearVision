# BearVision contributor routing

Treat `src/bearvision`, `apps`, `config`, `specs` and `tests/remake` as the
active BearVision 3 system. Read `README.md` before changing runtime behaviour
and follow the relevant specification under `specs/`.

Use `pretraining` and `tests/offline_yolo` only for offline dataset annotation
and YOLO fine-tuning. Read `pretraining/README.md` before changing that
workflow. Runtime code must not import it.

Treat `legacy` as unsupported historical reference. Work there only when the
request explicitly concerns legacy behaviour. Active and offline modules must
not import it.
