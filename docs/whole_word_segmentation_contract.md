# Whole-Word Segmentation Contract

The segmenter crops the configured ROI, thresholds foreground, removes components below the
configured area, groups likely detached dots/crossbars, and finds sufficiently wide zero-column
projection gaps. A gap is suppressed when a detached component group or recorded stroke spans it.

Output segments are immutable grayscale `uint8` crops ordered left to right. Positions are
continuous from zero and each segment contains a global canvas bounding box, source stroke IDs,
and segmentation confidence. The validator reports empty, too few, too many, ambiguous, failed,
or successful segmentation.

Connected components are evidence, not direct character labels. Manual `S` split and `V` merge are
available when the automatic boundary is wrong. Segment confidence is separate from model
confidence.
