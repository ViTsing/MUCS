from mmengine.config import read_base

# opencompass examples/eval_qwen2_5_7b_tulu.py -p {PARTITION} -l

with read_base():
    # models:
    from opencompass.configs.models.hf_llama.hf_llama3_1_8b_tulu3_0_10k import llama0_10k_models
    from opencompass.configs.models.hf_llama.hf_llama3_1_8b_tulu3_1_10k import llama1_10k_models
    # datasets:
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets


models = llama0_10k_models + llama1_10k_models
datasets = bbh_datasets + mmlu_datasets + ARC_c_datasets + ARC_e_datasets + gsm8k_datasets + humaneval_datasets + ifeval_datasets