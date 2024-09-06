"""Custom exceptions for the Spark AI agent."""


class UnsupportedModelTypeError(ValueError):
    """Unsupported model type error."""

    def __init__(self, model_name: str):
        super().__init__(f"Unsupported model type: {model_name}")
