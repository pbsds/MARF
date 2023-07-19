import pytorch_lightning as pl


class DtypeMixin:
    def __init_subclass__(cls):
        assert issubclass(cls, pl.LightningModule), \
            f"{cls.__name__!r} is not a subclass of 'pytorch_lightning.LightningModule'!"

    @property
    def device_and_dtype(self) -> dict:
        """
        Examples:
        ```
        torch.tensor(1337, **self.device_and_dtype)
        some_tensor.to(**self.device_and_dtype)
        ```
        """

        return {
            "dtype": self.dtype,
            "device": self.device,
        }
