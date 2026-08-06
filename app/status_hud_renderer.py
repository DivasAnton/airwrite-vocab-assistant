from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


class StatusHudRenderer:
    def __init__(
        self,
        *,
        target_font_height_px: int = 13,
        margin_px: int = 10,
        padding_px: int = 10,
        line_gap_px: int = 6,
        panel_color: tuple[int, int, int] = (245, 245, 245),
        text_color: tuple[int, int, int] = (20, 20, 20),
        panel_opacity: float = 0.88,
    ) -> None:
        if target_font_height_px <= 0:
            raise ValueError("target_font_height_px must be positive")
        if margin_px < 0 or padding_px < 0 or line_gap_px < 0:
            raise ValueError("HUD spacing values must be non-negative")
        if not 0.0 <= panel_opacity <= 1.0:
            raise ValueError("panel_opacity must be between 0 and 1")
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.thickness = 1
        self.font_scale = cv2.getFontScaleFromHeight(
            self.font,
            target_font_height_px,
            self.thickness,
        )
        self.target_font_height_px = target_font_height_px
        self.margin_px = margin_px
        self.padding_px = padding_px
        self.line_gap_px = line_gap_px
        self.panel_color = panel_color
        self.text_color = text_color
        self.panel_opacity = panel_opacity

    def render(
        self,
        frame: NDArray[np.uint8],
        lines: Sequence[str],
    ) -> NDArray[np.uint8]:
        output = frame.copy()
        normalized = tuple(line.strip() for line in lines if line.strip())
        if not normalized:
            return output

        frame_height, frame_width = output.shape[:2]
        available_width = max(1, frame_width - 2 * (self.margin_px + self.padding_px))
        wrapped = self._wrap_lines(normalized, available_width)
        line_step = self.target_font_height_px + self.line_gap_px
        max_visible_lines = max(
            1,
            (frame_height - 2 * (self.margin_px + self.padding_px)) // line_step,
        )
        visible_lines = wrapped[:max_visible_lines]
        text_width = max(self._text_size(line)[0] for line in visible_lines)
        panel_width = min(
            frame_width - 2 * self.margin_px,
            text_width + 2 * self.padding_px,
        )
        panel_height = min(
            frame_height - 2 * self.margin_px,
            len(visible_lines) * line_step - self.line_gap_px + 2 * self.padding_px,
        )
        right = frame_width - self.margin_px
        left = max(self.margin_px, right - panel_width)
        top = self.margin_px
        bottom = min(frame_height - self.margin_px, top + panel_height)

        overlay = output.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), self.panel_color, -1)
        output = cast(
            NDArray[np.uint8],
            cv2.addWeighted(
                overlay,
                self.panel_opacity,
                output,
                1.0 - self.panel_opacity,
                0.0,
            ),
        )
        baseline_y = top + self.padding_px + self.target_font_height_px
        for line in visible_lines:
            cv2.putText(
                output,
                line,
                (left + self.padding_px, baseline_y),
                self.font,
                self.font_scale,
                self.text_color,
                self.thickness,
                cv2.LINE_AA,
            )
            baseline_y += line_step
        return output

    def _wrap_lines(
        self,
        lines: tuple[str, ...],
        max_width: int,
    ) -> tuple[str, ...]:
        wrapped: list[str] = []
        for line in lines:
            words = line.split()
            if not words:
                continue
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                if self._text_size(candidate)[0] <= max_width:
                    current = candidate
                else:
                    wrapped.append(current)
                    current = word
            wrapped.append(current)
        return tuple(wrapped)

    def _text_size(self, text: str) -> tuple[int, int]:
        size, _baseline = cv2.getTextSize(
            text,
            self.font,
            self.font_scale,
            self.thickness,
        )
        return size
