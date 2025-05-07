import argparse
from suffix_attack import *
from replacement_attack import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run adversarial SAE attack")
    parser.add_argument("--targeted", action="store_true", help="Enable targeted attack")
    parser.add_argument("--level", choices=["population", "individual"], default="population", help="Level of analysis")
    parser.add_argument("--mode", choices=["suffix", "replacement"], default="suffix", help="Perturbation mode")
    parser.add_argument("--activate", action="store_true", help="Only relevant if level=individual. Whether to activate (True) or deactivate (False) neurons")
    parser.add_argument("--sample_idx", type=int, default=20)
    parser.add_argument("--layer_num", type=int, default=20)
    parser.add_argument("--num_latents", type=int, default=10)
    parser.add_argument("--suffix_len", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--m", type=int, default=300)
    parser.add_argument("--k", type=int, default=192)
    parser.add_argument("--data_file", type=str, default="./two_class_generated.csv")
    parser.add_argument("--base_dir", type=str, default="/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/")
    parser.add_argument("--model_type", type=str, choices=["llama3-8b", "gemma2-9b"], default="llama3-8b", help="Model architecture")
    parser.add_argument("--log", action="store_true", help="Log results")
    # parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    return parser.parse_args()

def launch():
    args = parse_args()

    if args.level == "individual":
        if args.mode == "suffix":
            run_individual_suffix_attack(args)
        else:
            run_individual_replace_attack(args)
        

    else:  # group level
        if args.mode == "suffix":
            if args.targeted:
                run_group_targeted_suffix_attack(args)
            else:
                run_group_untargeted_suffix_attack(args)
        else:
            if args.targeted:
                run_group_targeted_replace_attack(args)
            else:
                run_group_untargeted_replace_attack(args)
        

if __name__ == "__main__":
    launch()
