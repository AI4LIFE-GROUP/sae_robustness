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
    parser.add_argument("--layer_num", type=int, default=20) # 30 for gemma2-9b
    parser.add_argument("--num_latents", type=int, default=10) # only used in individual
    parser.add_argument("--suffix_len", type=int, default=1) # population level = 3, individual level = 1
    parser.add_argument("--batch_size", type=int, default=100) # 100 for individual level or replacement mode; (2/3) * (m * suffix_len) for population level
    parser.add_argument("--num_iters", type=int, default=10) # 50 for targeted population; 20 for untargeted population; 10 for all individual level
    parser.add_argument("--m", type=int, default=300) # 200 for gemma population suffix; 300 otherwise
    parser.add_argument("--k", type=int, default=192) # 192 for llama; 170 for gemma; not important
    parser.add_argument("--data_file", type=str, default="art_science")
    parser.add_argument("--base_dir", type=str, default="/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/")
    parser.add_argument("--model_type", type=str, choices=["llama3-8b", "gemma2-9b"], default="llama3-8b", help="Model architecture")
    parser.add_argument("--log", action="store_true", help="Log results")
    parser.add_argument("--random", action="store_true", help="Switch to random baseline")
    
    return parser.parse_args()

def launch():
    args = parse_args()

    if args.level == "individual":
        if args.mode == "suffix":
            success_rate = run_individual_suffix_attack(args)
        else:
            success_rate = run_individual_replace_attack(args)
        print(f"Individual attack success rate = {success_rate:.4f}")

    else:  
        if args.mode == "suffix":
            d_overlap = run_population_suffix_attack(args)
        else:
            d_overlap = run_population_replace_attack(args)
        print(f"Population attack overlap change = {d_overlap:.4f}")

    if args.log:
        sys.stdout.close()
        
if __name__ == "__main__":
    launch()
