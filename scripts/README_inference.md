# GridWM-Judge Inference Scripts

This directory contains scripts for running inference on GridWM-Judge exam questions and auditing the results with scientific rigor to prevent capability underestimation.

## Files Overview

- `run_inference.py` - Main inference script with capability-preserving features (v7.6.11)
- `audit_inference_outputs.py` - Post-inference audit script for quality assurance (v2.1.1)
- `README_inference.md` - This documentation

## Key Features (Capability Preservation)

### run_inference.py v7.6.11
- **Task-Aware Token Budgets**: Prevents truncation-caused capability underestimation
  * Task A: 512 tokens (atomic transitions)
  * Task B: 2048 tokens (JSON structures)
  * Task C: 256 tokens (success/failure judgment)
- **Image Fidelity**: PNG-first encoding for storyboard/grid images
- **Experiment Isolation**: Automatic experiment directory management
- **Real-time Progress**: Live updates with ETA and error tracking
- **Zero-Trust Security**: Comprehensive secret masking and API validation

### audit_inference_outputs.py v2.1.1
- **Robust Format Extraction**: Distinguishes format noise from capability limits
- **Auto-Discovery**: Intelligent experiment result location
- **Scientific Auditing**: Prevents engineering artifacts from distorting assessments

## Quick Start

### 0. Generate Requests from Exam Data
```bash
# Generate requests JSONL files from exam data
python scripts/run_inference.py \
  --exam_dir datasets/exams \
  --responses_dir /tmp/requests_output \
  --backend openai_compatible \
  --provider siliconflow \
  --model Pro/Qwen/Qwen2.5-VL-7B-Instruct \
  --api_key test_key \
  --build_exam_requests_only
```

### 1. Run Inference on All Exam Questions

```bash
# Using SiliconFlow Qwen2.5-VL-7B-Instruct (recommended for testing)
export SILICONFLOW_API_KEY="your_key"

# Method 1: Direct exam mode (auto-generates requests)
python scripts/run_inference.py \
  --exam_dir datasets/exams \
  --exam_task all \
  --responses_dir runs/responses/siliconflow \
  --backend openai_compatible \
  --provider siliconflow \
  --model Pro/Qwen/Qwen2.5-VL-7B-Instruct \
  --max_new_tokens 128 \
  --temperature 0.0 \
  --resume

# Method 2: Use pre-generated requests
python scripts/run_inference.py \
  --requests runs/responses/siliconflow/_requests_from_exam/requests_exam_all.jsonl \
  --responses_dir runs/responses/siliconflow \
  --backend openai_compatible \
  --provider siliconflow \
  --model Pro/Qwen/Qwen2.5-VL-7B-Instruct \
  --max_new_tokens 128 \
  --temperature 0.0 \
  --resume

# Generate requests only (no API calls)
python scripts/run_inference.py \
  --exam_dir datasets/exams \
  --exam_task all \
  --responses_dir /tmp/test_requests \
  --backend openai_compatible \
  --provider siliconflow \
  --model Pro/Qwen/Qwen2.5-VL-7B-Instruct \
  --api_key test_key \
  --build_exam_requests_only

# Custom progress reporting interval (default 10 seconds)
python scripts/run_inference.py \
  --requests /path/to/requests.jsonl \
  --responses_dir runs/responses/model \
  --backend openai_compatible \
  --provider siliconflow \
  --model Pro/Qwen/Qwen2.5-VL-7B-Instruct \
  --progress_interval 5.0 \
  [other options...]
```

### 2. Run Inference with Sharding (for parallel processing)

```bash
# Shard 1/4
python scripts/run_inference.py \
  --exam_dir datasets/exams \
  --responses_dir runs/responses/siliconflow \
  --backend openai_compatible \
  --provider siliconflow \
  --model Pro/Qwen/Qwen2.5-VL-7B-Instruct \
  --api_key YOUR_API_KEY \
  --num_shards 4 \
  --shard_id 0

# Shard 2/4
python scripts/run_inference.py \
  --exam_dir datasets/exams \
  --responses_dir runs/responses/siliconflow \
  --backend openai_compatible \
  --provider siliconflow \
  --model Pro/Qwen/Qwen2.5-VL-7B-Instruct \
  --api_key YOUR_API_KEY \
  --num_shards 4 \
  --shard_id 1

# ... repeat for shards 2 and 3
```

### 3. Audit Inference Results

```bash
# Auto-discover and audit latest experiment (recommended)
python scripts/audit_inference_outputs.py \
  --responses_dir runs/responses \
  --exam_task all \
  --out_report audit_report.json

# Audit specific experiment by directory name
python scripts/audit_inference_outputs.py \
  --responses_dir runs/responses \
  --experiment openaicompatible_siliconflow_Qwen2.5-VL-7B-Instruct_native_shard0 \
  --out_report audit_report.json

# Manual specification with explicit files
python scripts/audit_inference_outputs.py \
  --requests runs/responses/_requests_from_exam/requests_exam_all.jsonl \
  --responses runs/responses/openaicompatible_siliconflow_Qwen2.5-VL-7B-Instruct_native_shard0/requests_exam_all.shard0.jsonl \
  --out_report audit_report.json
```

## run_inference.py Usage

### Progress Reporting

The script provides real-time progress updates during inference:

```
🔄 Starting inference on shard 0 (resume: 0 already done)
📊 Shard 0: 50 done, 4.85/sec, elapsed: 10.3s
📊 Shard 0: 100 done, 4.92/sec, elapsed: 20.3s
📊 Shard 0: 150 done, 4.95/sec, elapsed: 30.3s
✅ Shard 0 completed: 200 total, 4.97/sec, total time: 40.2s
```

- **Start message**: Shows shard ID and resume count
- **Progress updates**: Every `--progress_interval` seconds (default 10s)
- **Final summary**: Total count, average rate, and total time
- **Always flushed**: Output appears immediately, no buffering issues

## Command Line Options

#### Required Arguments
- `--responses_dir`: Directory to save inference results

#### Exam Mode (Recommended)
- `--exam_dir`: Directory containing exam JSONL files (default: `datasets/exams`)
- `--exam_task`: Which tasks to run (`all`, `A`, `B`, or `C`; default: `all`)

#### Manual Mode (Advanced)
- `--requests`: Path to custom requests JSONL file
- `--build_exam_requests_only`: Only generate requests files, don't run inference

#### Backend Configuration
- `--backend`: Backend type (`openai_compatible`, `gemini`, `local`; default: `openai_compatible`)
- `--provider`: Provider name (e.g., `siliconflow`, `openai`, `zhizengzeng`)
- `--model`: Model name (provider-specific)
- `--api_key`: API key (can also use environment variables)

#### Generation Parameters
- `--max_new_tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.0)
- `--max_tokens_A`: Task A token budget (default: 512)
- `--max_tokens_B`: Task B token budget (default: 2048)
- `--max_tokens_C`: Task C token budget (default: 256)
- `--api_timeout`: API timeout in seconds (default: 60)
- `--api_image_max_side`: Max image dimension for compression (default: 4096)
- `--api_jpeg_quality`: JPEG compression quality (default: 85)

#### Progress and Logging
- `--progress_every`: Progress update every N items (default: 50)
- `--progress_interval_s`: Heartbeat interval in seconds (default: 10.0)
- `--progress_show_uid`: Show current UID in progress updates
- `--progress_total`: Count total items for ETA calculation

#### Parallel Processing
- `--num_shards`: Number of parallel shards (default: 1)
- `--shard_id`: Shard ID for this process (0 to num_shards-1)
- `--resume`: Resume from existing response files

### Supported Backends

#### OpenAI Compatible (SiliconFlow, etc.)
```bash
python scripts/run_inference.py \
  --backend openai_compatible \
  --provider siliconflow \
  --model Pro/Qwen/Qwen2.5-VL-7B-Instruct \
  --api_key YOUR_API_KEY
```

#### Environment Variables
You can set API keys via environment variables:
- `SILICONFLOW_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- etc.

## audit_inference_outputs.py Usage

### What It Audits

The audit script performs comprehensive quality checks on inference results:

#### P0: Coverage & Integrity
- **Missing answers**: Expected vs answered question counts
- **Duplicates**: Same question answered multiple times
- **Gateway errors**: API failures (`__GW_ERR__` prefix)
- **Metadata continuity**: `env_task`, `group_id`, `variant` fields present

#### P1: Output Format Compliance
- **Task A**: Pure single letters A/B/C/D
- **Task B**: Valid JSON with required schema
- **Task C**: Exact "Success" or "Fail" strings

#### P2: Distribution Sanity
- **Task C success rates** by `env_task` and `variant`
- **Answer distribution bias** detection

### Output Format

The script prints a summary to stdout and optionally writes a detailed JSON report:

```json
{
  "exam_dir": "/path/to/exams",
  "response_files": ["file1.jsonl", "file2.jsonl"],
  "coverage": {
    "expected_total": 1600,
    "answered_total": 1600,
    "missing": 0,
    "extra": 0,
    "duplicates": 0
  },
  "gateway_errors": {
    "n": 0,
    "rate": 0.0
  },
  "by_task": {
    "A": {"ok_strict": 150, "viol:has_letter_but_not_strict": 50},
    "B": {"ok_schema": 180, "viol:no_json_object": 20},
    "C": {"ok_strict": 1000, "viol:case_mismatch": 200}
  }
}
```

## Expected Results for GridWM-Judge

### Coverage (P0)
- **Expected total**: 1600 (200 A + 200 B + 1200 C)
- **Missing**: 0 (all questions answered)
- **Duplicates**: 0 (no duplicate answers)
- **Gateway errors**: <1% (API reliability)

### Format Compliance (P1)
- **Task A recoverable rate**: >95% (A/B/C/D with format noise tolerance)
- **Task B recoverable rate**: >90% (valid JSON with canonicalization)
- **Task C recoverable rate**: >99% (Success/Fail with robust extraction)

### Distribution Sanity (P2)
- **Task C full variant**: Mostly Success
- **Task C cf variant**: Mostly Fail
- **Balanced per env_task**: No extreme biases

## Troubleshooting

### Common Issues

1. **"No experiment results found"**
   - Check `--responses_dir` contains experiment subdirectories
   - Verify experiment ID format: `{backend}_{provider}_{model}_{protocol}_shard{shard_id}`
   - Use `--experiment` to specify exact experiment directory

2. **High gateway error rate**
   - Check API key validity and provider configuration
   - Increase `--api_timeout` (default 60s)
   - Reduce concurrent requests or implement rate limiting
   - Verify model supports required image input formats

3. **Low Task C recoverable rate (<95%)**
   - **CRITICAL**: Indicates token budget too low causing truncation
   - Increase `--max_tokens_C` (current default: 128, may need 256+ for complex reasoning)
   - Check for model-specific token limits

4. **Format compliance failures**
   - Task A: Model may need clearer A/B/C/D instructions
   - Task B: Verify model supports structured JSON output
   - Task C: Check prompt clarity for binary classification
   - Consider temperature adjustments (0.0 recommended for consistency)

5. **Experiment conflicts**
   - Different model configurations automatically create separate experiment directories
   - Use `--resume` to continue interrupted runs
   - Check file permissions in `--responses_dir`

### Performance Tips

- **Parallel processing**: Use `--num_shards` > 1 for large batches
- **Resume capability**: Use `--resume` to continue after interruptions
- **Image compression**: Adjust `--api_image_max_side` to balance quality/speed
- **Rate limiting**: Add delays between requests if hitting API limits

## File Structure

```
runs/responses/
├── _requests_from_exam/              # Generated request files
│   ├── requests_exam_all.jsonl       # All tasks combined
│   ├── requests_A.jsonl              # Task A requests
│   ├── requests_B.jsonl              # Task B requests
│   └── requests_C.jsonl              # Task C requests
├── openaicompatible_siliconflow_Qwen2.5-VL-7B-Instruct_native_shard0/
│   └── requests_exam_all.shard0.jsonl  # Experiment results
├── openaicompatible_gemini_gemini-1.5-pro_native_shard0/
│   └── requests_exam_all.shard0.jsonl  # Another experiment
└── audit_report.json                 # Latest audit results

scripts/
├── run_inference.py                  # Main inference script (v7.6.11)
├── audit_inference_outputs.py        # Audit script (v2.1.1)
└── README_inference.md              # This documentation
```

### Experiment ID Format

Experiments are automatically organized with descriptive IDs:
```
{backend}_{provider}_{model}_{protocol}_shard{shard_id}
```

Examples:
- `openaicompatible_siliconflow_Qwen2.5-VL-7B-Instruct_native_shard0`
- `gemini_gemini_gemini-1.5-pro_native_shard0`
- `qwen2.5vl_custom_Qwen2.5-VL-7B_Instruct_fused_h_shard0`
