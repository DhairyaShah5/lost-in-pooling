# Lost in Pooling

Measuring and recovering small-signal evidence discarded by frozen foundation models.

CSCI 566, Deep Learning and its Applications, University of Southern California, Fall 2026.

## Idea

Frozen vision transformers keep evidence about small objects in their patch tokens,
but the usual recipe (take the global CLS/mean embedding, fit a linear probe) collapses
those tokens into one vector and the evidence is lost. We measure detection quality as a
function of relative target size (a "blindness curve"), compare several read-outs over
the *same* cached tokens, and train a small attention head that learns where to look
from image-level labels only.

## Layout

```
src/lip/        library code (data, features, readouts, models, eval)
scripts/        numbered pipeline steps, run in order
results/        committed JSON results and figures
report/         midterm report (LaTeX + PDF)
docs/           plan, data/feature contracts, todo
data/ features/ local only, gitignored
```

## Reproduce

See `docs/CONTRACTS.md` for file formats and `TODO.md` for the step list.
Reproduction commands are filled in as each script lands.
