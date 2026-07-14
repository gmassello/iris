# OCR engine benchmark

Dataset: CORD v2 (test split, 100 receipts). Machine: Apple Silicon.

| Engine | CER | WER | Worst CER | Macro F1 | F1 subtotal | F1 tax | F1 total | F1 items | Median latency | Failures |
|---|---|---|---|---|---|---|---|---|---|---|
| **tesseract** | 0.440 | 0.897 | 16.3 | 0.257 | 0.360 | 0.163 | 0.389 | 0.118 | 284 ms | 0 |
| **doctr** | 0.096 | 0.250 | 1.2 | 0.574 | 0.752 | 0.340 | 0.807 | 0.397 | 697 ms | 0 |
| **mlx-vlm** | 0.202 | 0.732 | 20.6 | 0.411 | 0.518 | 0.245 | 0.496 | 0.385 | 5439 ms | 0 |

CER/WER: raw text quality, lower is better. F1: structured extraction, higher is better.

**CER and WER are medians, not means.** On a handful of receipts with textured backgrounds,
Tesseract reads the noise as if it were text and emits thousands of characters against a ground
truth of two hundred: a CER of 16 on a single receipt moves the mean across the 100 from 0.42 to
0.89. The mean would describe those three disasters, not the engine's behavior. The tail is not
hidden: it goes in the **Worst CER** column, which is exactly where you see that Tesseract without
a clean background has no floor.

**Read `mlx-vlm`'s CER with care.** CER and WER compare sequences, and the VLM transcribes in a
different reading order than the ground truth: it groups by column (descriptions first, then
amounts) while CORD interleaves them. The content is right but the sequence does not line up, and
that penalizes it. For this engine, field-level F1 —which is order-independent— is the
representative measure. It is a limitation of the metric, not of the model, and that is why the
number is published anyway instead of being hidden.

`merchant`, `date` are not scored: **CORD does not annotate them**. Including them would give F1=0 for every engine and sink the macro-F1 over a dataset limitation, not an engine failure. iris does extract them; there is simply nothing here to measure them against.
