# GridWM-Judge: Evaluating VLM Judges in Grid Worlds via World Model Deficits

A benchmark for evaluating Vision-Language Models (VLMs) as judges in embodied AI tasks. We investigate whether VLM "judging instability" stems from "world model deficits" through three diagnostic tasks.

## Overview

| Task | Questions | Description |
|------|-----------|-------------|
| Task A | 200 | Atomic transition prediction |
| Task B | 200 | Structured scene perception |
| Task C | 1200 | Judgment robustness under perturbations |

**Tasks**: DoorKey, Memory, LavaGap, KeyCorridor, MultiRoom, RedBlueDoor  
**Trajectory Variants**: Full (success), NoCue (evidence masked), CF (counterfactual failures)

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Generate datasets
./scripts/generate_all_datasets.sh
python scripts/build_exam.py

# Run evaluation
export SILICONFLOW_API_KEY="your_key"
python scripts/run_inference.py --exam_dir datasets/exams --responses_dir runs/responses \
  --backend openai_compatible --provider siliconflow \
  --model "Pro/Qwen/Qwen2.5-VL-7B-Instruct"

# Score results
python scripts/score_exam.py
```

## Citation

<div align="center">
  <details>
    <summary><b>Click to copy BibTeX</b></summary>
    <br>
    <div style="background-color: #f6f8fa; border-radius: 6px; padding: 16px; text-align: left;">
      <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word;"><code>@inproceedings{zhang2026gridwm,
	title={{GridWM-Judge}: Evaluating Vision-Language Model Judges in Grid Worlds via World Model Deficits},
	author={Zhang, Qinan and Jin, Qihang},
	booktitle={ICLR 2026 Workshop on World Models: Understanding, Modelling and Scaling},
	year={2026}
}</code></pre>
    </div>
  </details>
</div>

## License

MIT OR Apache-2.0
