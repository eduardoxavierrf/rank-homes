"""Structured recoverable workflow diagnostics."""


class DiagnosticWarning(UserWarning):
    def __init__(self, message: str, category: str):
        super().__init__(message)
        self.category = category
