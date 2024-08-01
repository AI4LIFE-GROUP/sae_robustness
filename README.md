## Introduction
This project repo is based on the SAE library https://github.com/EleutherAI/sae, with some customized local package changes. The trained SAEs are loaded from https://huggingface.co/EleutherAI/sae-llama-3-8b-32x.
## Loading pretrained SAEs

To load a pretrained SAE from the HuggingFace Hub, you can use the `Sae.load_from_hub` method as follows:

```python
from sae import Sae

sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10")
```

This will load the SAE for residual stream layer 10 of Llama 3 8B, which was trained with an expansion factor of 32. You can also load the SAEs for all layers at once using `Sae.load_many`:

```python
saes = Sae.load_many("EleutherAI/sae-llama-3-8b-32x")
saes["layers.10"]
```

The dictionary returned by `load_many` is guaranteed to be [naturally sorted](https://en.wikipedia.org/wiki/Natural_sort_order) by the name of the hook point. For the common case where the hook points are named `embed_tokens`, `layers.0`, ..., `layers.n`, this means that the SAEs will be sorted by layer number. We can then gather the SAE activations for a model forward pass as follows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

with torch.inference_mode():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    outputs = model(**inputs, output_hidden_states=True)

    latent_acts = []
    for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
        latent_acts.append(sae.encode(hidden_state))

# Do stuff with the latent activations
```

## Training SAEs

To train SAEs from the command line, you can use the following command:

```bash
python -m sae EleutherAI/pythia-160m togethercomputer/RedPajama-Data-1T-Sample
```

The CLI supports all of the config options provided by the `TrainConfig` class. You can see them by running `python -m sae --help`.

Programmatic usage is simple. Here is an example:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize

MODEL = "EleutherAI/pythia-160m"
dataset = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer)


gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
)

cfg = TrainConfig(
    SaeConfig(gpt.config.hidden_size), batch_size=16
)
trainer = SaeTrainer(cfg, tokenized, gpt)

trainer.fit()
```

## Custom hookpoints

By default, the SAEs are trained on the residual stream activations of the model. However, you can also train SAEs on the activations of any other submodule(s) by specifying custom hookpoint patterns. These patterns are like standard PyTorch module names (e.g. `h.0.ln_1`) but also allow [Unix pattern matching syntax](https://docs.python.org/3/library/fnmatch.html), including wildcards and character sets. For example, to train SAEs on the output of every attention module and the inner activations of every MLP in GPT-2, you can use the following code:

```bash
python -m sae gpt2 togethercomputer/RedPajama-Data-1T-Sample --hookpoints "h.*.attn" "h.*.mlp.act"
```

To restrict to the first three layers:

```bash
python -m sae gpt2 togethercomputer/RedPajama-Data-1T-Sample --hookpoints "h.[012].attn" "h.[012].mlp.act"
```

We currently don't support fine-grained manual control over the learning rate, number of latents, or other hyperparameters on a hookpoint-by-hookpoint basis. By default, the `expansion_ratio` option is used to select the appropriate number of latents for each hookpoint based on the width of that hookpoint's output. The default learning rate for each hookpoint is then set using an inverse square root scaling law based on the number of latents. If you manually set the number of latents or the learning rate, it will be applied to all hookpoints.

## Distributed training

We support distributed training via PyTorch's `torchrun` command. By default we use the Distributed Data Parallel method, which means that the weights of each SAE are replicated on every GPU.

```bash
torchrun --nproc_per_node gpu -m sae meta-llama/Meta-Llama-3-8B --batch_size 1 --layers 16 24 --k 192 --grad_acc_steps 8 --ctx_len 2048
```

This is simple, but very memory inefficient. If you want to train SAEs for many layers of a model, we recommend using the `--distribute_modules` flag, which allocates the SAEs for different layers to different GPUs. Currently, we require that the number of GPUs evenly divides the number of layers you're training SAEs for.

```bash
torchrun --nproc_per_node gpu -m sae meta-llama/Meta-Llama-3-8B --distribute_modules --batch_size 1 --layer_stride 2 --grad_acc_steps 8 --ctx_len 2048 --k 192 --load_in_8bit --micro_acc_steps 2
```

The above command trains an SAE for every _even_ layer of Llama 3 8B, using all available GPUs. It accumulates gradients over 8 minibatches, and splits each minibatch into 2 microbatches before feeding them into the SAE encoder, thus saving a lot of memory. It also loads the model in 8-bit precision using `bitsandbytes`. This command requires no more than 48GB of memory per GPU on an 8 GPU node.
