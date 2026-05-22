# ExtremeLSTMMemo Refactor

This is a modular refactor of the original `ExtremeLSTMMemo` implementation.
The algorithmic behavior and public model interface are kept as close as possible to the original version.

## File structure

```text
extreme_lstm_memo_refactor/
  ExtremeLSTMMemo.py              # backward-compatible wrapper
  extreme_lstm_memo/
    __init__.py
    model.py                      # main ExtremeLSTMMemo model
    prior.py                      # StudentTMixturePrior
    router.py                     # RouterFromEmbeddingPreTrain
    moe.py                        # LSTMExpert and BackboneMoE
    gate.py                       # RetrievalBetaGate
    retrieval.py                  # RetrievalMemory and cosine retrieval
    losses.py                     # auxiliary balance losses
```

## How to use

Recommended import:

```python
from extreme_lstm_memo import ExtremeLSTMMemo
```

If your old code imports the model from a single file, you can place `ExtremeLSTMMemo.py`
and the `extreme_lstm_memo/` directory in the same model folder, then keep:

```python
from ExtremeLSTMMemo import ExtremeLSTMMemo
```

or, if it is inside a `models` package:

```python
from models.ExtremeLSTMMemo import ExtremeLSTMMemo
```

## Notes

- `forward(...)` keeps the original return style: `(out, total_aux_loss)` or `(out, total_aux_loss, aux_dict)`.
- `construct_index`, `add_key_value`, `retrieval`, and `cosine_similarity` are kept on the main model for compatibility.
- Retrieval internals are moved to `RetrievalMemory`.
- State prior, router, MoE backbone, retrieval gate, and losses are separated for readability and paper-code submission.
