from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray


class DebugImageExporter:
    def __init__(self, output_dir: Path) -> None:
        if not str(output_dir).strip():
            raise ValueError("output_dir must not be empty")
        self.output_dir = output_dir

    def export(self, images: dict[str, NDArray[np.uint8]]) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid4().hex[:8]
        exported_paths: list[Path] = []
        for name, image in images.items():
            output_path = self.output_dir / f"{run_id}_{name}.png"
            success = cv2.imwrite(str(output_path), image)
            if success:
                exported_paths.append(output_path)
        return exported_paths
