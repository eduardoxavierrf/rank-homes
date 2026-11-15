import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from scripts.prepare_rooms import SOURCE_CLASSES, export_dataset


class ExpandedDatasetExportTests(unittest.TestCase):
    def make_split(self, root, split, *, omit=()):
        folder = root / split / "bathroom"
        folder.mkdir(parents=True)
        Image.new("RGB", (12, 10)).save(folder / "room.jpg")
        mask = Image.new("L", (12, 10))
        mask.paste(255, (2, 3, 8, 9))
        mask.save(folder / "visible.png")
        objects = [
            {"name": name, "parts": {"part_level": 0}, "instance_mask": "visible.png"}
            for name, target in SOURCE_CLASSES.items()
            if target not in omit
        ]
        (folder / "room.json").write_text(
            json.dumps({"annotation": {"filename": "room.jpg", "object": objects}}), encoding="utf-8"
        )

    def test_export_retains_shower_door_without_validation_examples(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.make_split(root, "train")
            self.make_split(root, "validation", omit={"shower door"})
            output = root / "export"
            summary = json.loads(export_dataset(root, output))
            train = json.loads((output / "train/_annotations.coco.json").read_bytes())
            valid = json.loads((output / "valid/_annotations.coco.json").read_bytes())
            self.assertEqual(train["categories"], valid["categories"])
            self.assertEqual(len(train["categories"]), 23)
            names = {item["name"] for item in train["categories"]}
            additions = {"shower stall", "shower door", "floor", "wall", "countertop", "faucet", "window"}
            self.assertTrue(additions | {"shower"} <= names)
            self.assertEqual(summary["train"]["boxes_by_class"]["shower door"], 1)
            self.assertEqual(summary["valid"]["boxes_by_class"]["shower door"], 0)
            self.assertEqual(summary["valid"]["classes_without_boxes"], ["shower door"])
            self.assertTrue(all(item["bbox"] == [2, 3, 6, 6] for item in train["annotations"]))

    def test_missing_training_class_does_not_publish_dataset(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.make_split(root, "train", omit={"shower door"})
            self.make_split(root, "validation")
            output = root / "export"
            with self.assertRaisesRegex(ValueError, "Classes with no boxes.*shower door"):
                export_dataset(root, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
