from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mangrove_seg.inference import predict_tiled


class FirstChannelModel:
    def __init__(self, mode: str = "array") -> None:
        self.mode = mode

    def predict(self, batch: np.ndarray, *, verbose: int) -> Any:
        del verbose
        prediction = batch[..., :1]
        if self.mode == "dict":
            return {"mask": prediction, "aux_1": prediction}
        if self.mode == "list":
            return [prediction]
        return prediction


@pytest.mark.parametrize("mode", ["array", "dict", "list"])
def test_tiled_prediction_preserves_values_across_output_styles(mode: str) -> None:
    image = np.arange(7 * 9, dtype=np.float32).reshape(7, 9, 1) / 100
    result = predict_tiled(FirstChannelModel(mode), image, tile_size=4, stride=2, batch_size=3)
    assert result.shape == (7, 9, 1)
    assert np.allclose(result, image)


def test_tiled_prediction_pads_small_images() -> None:
    image = np.ones((2, 3, 1), dtype=np.float32)
    result = predict_tiled(FirstChannelModel(), image, tile_size=8)
    assert np.array_equal(result, image)


def test_invalid_inference_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="channels-last"):
        predict_tiled(FirstChannelModel(), np.zeros((4, 4)), tile_size=4)
    with pytest.raises(ValueError, match="Batch size"):
        predict_tiled(FirstChannelModel(), np.zeros((4, 4, 1)), tile_size=4, batch_size=0)


class WrongShapeModel:
    def predict(self, batch: np.ndarray, *, verbose: int) -> np.ndarray:
        del verbose
        return batch[:, :-1, :-1, :1]


def test_wrong_model_output_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        predict_tiled(WrongShapeModel(), np.zeros((4, 4, 1)), tile_size=4)
