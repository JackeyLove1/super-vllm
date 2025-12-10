import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 2048
    max_ouput_len = 512

    path = os.path.expanduser("/root/autodl-tmp/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    
    # 加入prefill token计算系统总吞吐量
    num_input_tokens = sum(len(p) for p in prompt_token_ids)
    num_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    total_tokens = num_input_tokens + num_output_tokens
    
    # total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()

"""
# nano-vllm
(nano-vllm) ➜  nano-vllm-main python bench.py  
`torch_dtype` is deprecated! Use `dtype` instead!
Generating: 100%|███████████████| 1/1 [00:29<00:00, 29.59s/it, Prefill=0tok/s, Decode=299tok/s]
Total: 133966tok, Time: 42.53s, Throughput: 3149.79tok/s

Generating: 100%|███████████████| 1/1 [00:29<00:00, 29.13s/it, Prefill=0tok/s, Decode=300tok/s]
Total: 133966tok, Time: 41.73s, Throughput: 3210.62tok/s



INFO 12-09 10:41:57 [core.py:193] init engine (profile, create kv cache, warmup model) took 64.64 seconds
Adding requests: 100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 456.30it/s]
Processed prompts: 100%|██████████████████████████| 1/1 [00:00<00:00,  9.76it/s, est. speed input: 29.40 toks/s, output: 156.76 toks/s]
Total: 133966tok, Time: 47.50s, Throughput: 2820.08tok/s
"""

"""
`torch_dtype` is deprecated! Use `dtype` instead!
Generating: 100%|█████| 1/1 [00:29<00:00, 29.36s/it, Prefill=0tok/s, Decode=316tok/s]
Total: 276793tok, Time: 40.91s, Throughput: 6765.34tok/s
"""

"""
# FP8
"""