"""
Backward-compatible import wrapper.

If your original project imports:
    from models.ExtremeLSTMMemo import ExtremeLSTMMemo

you can keep the file name and replace its content with this wrapper after
copying the `extreme_lstm_memo/` directory into the same package.
"""

from extreme_lstm_memo import ExtremeLSTMMemo

__all__ = ["ExtremeLSTMMemo"]
