"""Public local-extractor configuration failures must precede model loading."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rank_homes.vision import create_extractor


class VisualExtractorTests(unittest.TestCase):
    def test_missing_local_detector_fails_without_acquiring_weights(self):
        with self.assertRaisesRegex(ValueError, "Local detector checkpoint"):
            create_extractor(detector_path=Path("missing-detector.pth"))

    def test_malformed_checkpoint_has_an_actionable_validation_error(self):
        import torch
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.pth"
            torch.save({"args": None, "model_config": None}, path)
            with self.assertRaisesRegex(ValueError, "checkpoint metadata"):
                create_extractor(detector_path=path)

    def test_legacy_architecture_and_noncanonical_class_maps_are_rejected(self):
        import torch
        from rank_homes.vision import CLASSES
        config = {"model_name": "RFDETRSmall", "encoder": "dinov2_windowed_small",
                  "resolution": 512, "num_classes": 23}
        variants = [
            ("RFDETRNano", config, list(CLASSES)),
            ("RFDETRSmall", {**config, "resolution": 384}, list(CLASSES)),
            ("RFDETRSmall", config, list(reversed(CLASSES))),
            ("RFDETRSmall", config, [name.replace("shower door", "shower") for name in CLASSES]),
            ("RFDETRSmall", config, list(CLASSES[:-1])),
        ]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "incompatible.pth"
            for architecture, settings, classes in variants:
                with self.subTest(architecture=architecture, settings=settings, classes=classes):
                    torch.save({"model_name": architecture, "model_config": settings,
                                "args": {"class_names": classes}}, path)
                    with self.assertRaisesRegex(ValueError, "canonical full 23-class"):
                        create_extractor(detector_path=path)


if __name__ == "__main__":
    unittest.main()
