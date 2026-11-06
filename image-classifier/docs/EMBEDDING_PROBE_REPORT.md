# Embedding probe: is there learnable signal in the photos?

_Script: `scripts/embedding_probe.py` (local; nothing is sent anywhere). Results: `data/results/embedding_probe/clip_b32_cv.json`._

**Question.** The 16 rule results barely predict the human decision (cross-validated ROC-AUC ~0.5 on 674 unique photos). Do the *pixels* carry signal about the label?
**Method.** Each unique train+validation photo (3,388: 2,651 approved, 737 not approved; the test split is never embedded) is taken from the resolution-normalized copies (long side 1280 px, identical JPEG
encoding), squashed to 224x224 without cropping, and embedded with CLIP ViT-B/32 (`openai/clip-vit-base-patch32`). A plain logistic regression is then evaluated with 5-fold cross-validation in which every group
(same submission, duplicates, near-duplicates) stays in one fold.

| Measure | Result |
|---|---:|
| Cross-validated ROC-AUC (best of C = 0.001 / 0.01 / 0.1) | **0.743** (0.733 / 0.743 / 0.716) |
| Chance level (labels shuffled by group, 5 runs) | 0.510 +/- 0.025 |
| PR-AUC (approved base rate 0.78) | 0.897 |
| Approved photos recovered at 5% false approvals | 18.9% |
| For reference: resolution alone / file size alone | 0.655 / 0.654 |
| ROC-AUC inside resolution bands: < 2 MP / 2-8 MP / >= 8 MP | 0.726 / 0.711 / 0.755 |

**Reading.**
* There is real signal in the images: well above chance, and it holds inside bands of similar original resolution, so it is not just the resolution/file-size shortcut. It is also far above the rule results (~0.5).
* It is modest, not strong. At the false-approval limit the business cares about (5%), only about one in five approved photos would be accepted. A tuned vision model could do better than this simple probe, which is a *lower bound*, but nothing here guarantees it.
* Normalization removes pixel-size, file-size and compression-level cues, but other provenance traces (for example softness from upscaling small images) could remain. The within-band results reduce that worry; they do not rule it out.

**Caveats.** Cross-validated on train+validation only (test untouched). Three values of C were compared, which makes the AUC slightly optimistic. One model (CLIP B/32) and one setting. Heavily duplicated groups were kept together in folds, but the sample (737 not-approved photos) is still small.

**Setup.** Needs `torch` (CPU) and `transformers` (`pip install -e .[probe]`; install torch from `https://download.pytorch.org/whl/cpu`). Embeddings are cached in `data/cache/embeddings_clip_b32.npz` (git-ignored); about 5 photos/s on a laptop CPU.

## Update: stronger image models (same protocol, same 3,388 photos)
| Features | Size | ROC-AUC | PR-AUC | Approved recall at 5% false approvals | Lowest AUC inside a resolution band |
|---|---:|---:|---:|---:|---:|
| CLIP ViT-B/32 (baseline) | 512 | 0.743 | 0.904 | 23.0% | 0.705 |
| DINOv2-base | 1536 | 0.747 | 0.905 | 25.5% | 0.715 |
| CLIP + DINOv2 together | 2048 | **0.766** | 0.914 | 27.3% | 0.728 |

* A much larger/different model alone barely moves the score (0.743 -> 0.747). Two very different models reaching the same level points to a ceiling set by the labels, not by how well the photos are "seen".
* Using both together adds about 0.02 AUC and the gain appears in every resolution band, so it is small but consistent.
* Chance level was 0.49-0.51 in all runs. The choice of C (4 values) makes every figure mildly optimistic. SigLIP and CLIP-L/14 were not run: after two unrelated models plateaued, another hour of CPU time looked unlikely to change the picture.
* Speed on this CPU: DINOv2-base about 1.9 photos/s in the full run (about 30 minutes), CLIP-B/32 about 5 photos/s.
Run it with `python scripts/embedding_probe.py embed --model dinov2_base` then `evaluate --models clip_b32 dinov2_base` and `compare`.
