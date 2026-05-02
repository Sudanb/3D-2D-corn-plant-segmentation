# Design Decisions — MaizeField3D Instance Segmentation

This document explains the rationale behind key decisions made throughout the project: from data construction through model selection to evaluation design.

---

## 1. Data Source — Terrestrial Laser Scanning (TLS)

**Decision**: Use TLS point clouds as the primary data source rather than real RGB images.

**Rationale**:
- Field-grown maize is densely occluded; RGB cameras cannot capture per-leaf geometry without destructive sampling
- TLS scanners capture full 3D geometry at millimetre resolution, enabling accurate per-leaf instance labelling in 3D space
- Manual annotation of real RGB images for 17-class leaf-level segmentation would be prohibitively expensive and inconsistent
- 3D labelling (assigning a unique colour per leaf segment in the point cloud) is done once per plant and propagates automatically to all 75 projected views — a 75× annotation efficiency gain

**Trade-off**: Synthetic-to-real domain gap. Projected images lack real sensor noise, motion blur, and lighting variation. This is partially mitigated by the contrast-stretched reflectance encoding from the TLS scanner (which differs from photographic RGB) and by YOLO's built-in augmentation (HSV jitter, mosaic).

---

## 2. Projection Strategy — Perspective Projection with Virtual Camera Orbit

**Decision**: Project point clouds to 2D using a simulated Intel RealSense D457 camera orbiting the plant on a hemisphere.

**Rationale**:
- Perspective projection is the most physically accurate 2D rendering of a 3D scene without full mesh reconstruction
- Using real camera intrinsics (D457) makes the synthetic images geometrically consistent with a plausible real deployment scenario
- A 75-view hemisphere orbit (5 azimuths × 5 elevations × 3 distances) provides view diversity within each plant without requiring additional plants to be scanned
- Painter's algorithm (far→near depth sorting) produces visually coherent RGB images without z-buffer artefacts on sparse point clouds

**Trade-off**: Projected images are inherently sparse at large distances. The morphological densification step fills gaps but can blur fine leaf boundaries.

---

## 3. Dataset Split — Plant-Level Not Image-Level

**Decision**: Split 520 plants into train/val/test (364/78/78) at the **plant level**.

**Rationale**:
- All 75 views of a single plant share the same 3D geometry and label set; splitting at the image level would leak correlated views into both train and test
- Plant-level splitting guarantees that no information about a test plant's geometry is seen during training
- A fixed random seed (3) ensures reproducibility

**Trade-off**: Fewer effective test samples (59 plants, 4,425 images) compared to a naive image-level split, but the evaluation is unbiased.

---

## 4. Class Configurations — 2-Class and 17-Class

**Decision**: Evaluate all models under both a 2-class (stem, leaf) and a 17-class (stem, leaf1–leaf16) configuration.

**Rationale**:
- **2-class**: Tests whether models can reliably separate plant organs — the minimum useful segmentation for biomass estimation or counting
- **17-class**: Tests whether models can resolve individual leaf identity by emergence order — required for leaf area index per phytomer, phenological staging, and organ-level growth modelling
- Running both configurations on the same data and models reveals how much performance degrades as the class granularity increases — a direct measure of architectural capacity for fine-grained discrimination

**Trade-off**: The 17-class problem has severe class imbalance (leaf16 appears rarely, leaf1 is often occluded) and is significantly harder. All models are expected to perform worse in this configuration; the comparison is about the *relative* degradation.

---

## 5. Model Selection — Four Architectures

**Decision**: Compare YOLOv8m-seg, YOLO11m-seg (26m), EoMT (DINOv2-S), and Mask R-CNN (ResNet-50).

**Rationale**:

| Model | Why included |
|-------|-------------|
| **YOLOv8m-seg** | Widely used single-stage baseline; fast, COCO-pretrained, well-documented |
| **YOLO11m-seg** | Successor to YOLOv8 with attention-enhanced neck; tests whether intra-family improvements transfer to this domain |
| **EoMT (DINOv2-S)** | State-of-the-art (CVPR 2025) transformer approach; frozen ViT backbone provides strong semantic features; tests query-based mask prediction on agricultural data |
| **Mask R-CNN (ResNet-50)** | Classic two-stage baseline; widely cited; provides the reference point that single-stage and transformer models are compared against |

This selection spans three paradigms (single-stage, two-stage, transformer) and two backbone families (CNN, ViT), enabling architectural conclusions beyond raw benchmark numbers.

**Why not larger models?** All medium-scale variants (~26–44M parameters) were chosen to keep compute comparable. Larger variants would have improved accuracy but confounded the architectural comparison with scale.

---

## 6. Training Budget — 10 Epochs for All Models

**Decision**: Train all models for exactly 10 epochs with batch size 16.

**Rationale**:
- A fixed epoch count is the simplest way to ensure a fair training budget across architectures with different convergence rates
- 10 epochs was chosen as a practical limit given GPU availability and deadline constraints
- All models are pretrained on COCO (or ImageNet-22k for EoMT), so meaningful performance is expected within 10 epochs of fine-tuning
- Early stopping was configured but set to patience=20 (effectively disabled at 10 epochs), ensuring all models train for the full budget

**Trade-off**: Some models (particularly EoMT with its attention mask annealing schedule) may benefit from more epochs. The 10-epoch comparison reflects initial fine-tuning capability, not asymptotic performance.

---

## 7. Evaluation Metric — COCO Mask mAP

**Decision**: Use COCO instance segmentation mAP (mask IoU) as the primary metric across all models.

**Rationale**:
- COCO mAP is the standard benchmark for instance segmentation; it enables comparison with published results
- Mask IoU (not bounding-box IoU) directly measures segmentation quality, which is the task of interest
- mAP50-95 penalises coarse masks more than mAP50, making it sensitive to architectural differences in mask resolution
- Per-class AP exposes which leaf positions (by emergence order) are systematically harder — this is agronomically meaningful

**Why also report Recall?** High precision with low recall indicates a conservative detector; high recall with low precision indicates over-segmentation. Both matter for downstream applications (e.g., counting leaves requires high recall; segmenting for measurement requires high precision).

---

## 8. Inference Speed — FPS and ms/image

**Decision**: Report FPS and per-image latency alongside accuracy metrics.

**Rationale**:
- Accuracy alone is insufficient for deployment decisions; a model that is 2× slower for 1% mAP gain may not be worth the cost in field robotics
- The four models span a wide speed range (YOLO single-pass vs EoMT ViT attention), making speed comparison informative
- Parameters and GFLOPs are included to separate model size from runtime efficiency

**Caveat**: FPS values are not directly comparable across frameworks (YOLO internal timer vs wall-clock for EoMT/Mask R-CNN). They are indicative of relative throughput within the same hardware setup, not absolute deployment benchmarks.

---

## 9. Image Resolution — 640 × 640

**Decision**: Use 640 × 640 px as the input resolution for all models.

**Rationale**:
- 640 × 640 is the native resolution of all projected images (matched to the simulated D457 sensor)
- It is also the default training resolution for Ultralytics YOLO models, avoiding any resolution-related redistortion
- EoMT's ViT-S/14 patch embedding divides 640 px into approximately 46 × 46 tokens — a dense enough feature map for fine leaf boundaries
- Mask R-CNN's FPN handles variable-size inputs; 640 px is within its optimal operating range

---

## 10. Pretrained Weights

**Decision**: Use COCO-pretrained weights for YOLO and Mask R-CNN; ImageNet-22k DINOv2 for EoMT.

**Rationale**:
- Pretraining provides strong low-level feature detectors (edges, textures) that transfer well even to synthetic plant images
- Training from scratch at 10 epochs would severely underfit for all models
- DINOv2 ViT-S is frozen during EoMT training; only the query tokens and heads are fine-tuned (~4M parameters), making it the most parameter-efficient fine-tuning approach in this comparison

**Trade-off**: COCO pretraining biases models toward natural-image statistics. The domain shift to synthetic point-cloud projections is non-trivial, but the 10-epoch fine-tuning is expected to adapt the heads sufficiently.
