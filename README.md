# Parler-TTS

```
git clone https://github.com/afrizalhasbi/parler-tts
cd parler-tts
pip install -e .[train]
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation
```

### Train
```
accelerate launch run.py config.json
```

### Compile
```
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

torch_device = "cuda:0"
torch_dtype = torch.bfloat16
model_name = "parler-tts/v007"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation="eager"
).to(torch_device, dtype=torch_dtype)

# compile the forward pass
compile_mode = "default" # chose "reduce-overhead" for 3 to 4x speed-up
model.generation_config.cache_implementation = "static"
model.forward = torch.compile(model.forward, mode=compile_mode)

# warmup
inputs = tokenizer("This is for compilation", return_tensors="pt").to(torch_device)
model_kwargs = {**inputs, "prompt_input_ids": inputs.input_ids, "prompt_attention_mask": inputs.attention_mask, }

n_steps = 1 if compile_mode == "default" else 2
for _ in range(n_steps):
    _ = model.generate(**model_kwargs)


# now you can benefit from compilation speed-ups
```
...


### 🎲 Random voice


**Parler-TTS** has been trained to generate speech with features that can be controlled with a simple text prompt, for example:

```py
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

prompt = "Hey, how are you doing today?"
description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
```

### 🎯 Using a specific speaker

To ensure speaker consistency across generations, this checkpoint was also trained on 34 speakers, characterized by name. The full list of available speakers includes:
Laura, Gary, Jon, Lea, Karen, Rick, Brenda, David, Eileen, Jordan, Mike, Yann, Joy, James, Eric, Lauren, Rose, Will, Jason, Aaron, Naomie, Alisa, Patrick, Jerry, Tina, Jenna, Bill, Tom, Carol, Barbara, Rebecca, Anna, Bruce, Emily.

To take advantage of this, simply adapt your text description to specify which speaker to use: `Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.`

You can replace "Jon" with any of the names from the list above to utilize different speaker characteristics. Each speaker has unique vocal qualities that can be leveraged to suit your specific needs. For more detailed information on speaker performance with voice consistency, please refer [inference guide](INFERENCE.md#speaker-consistency).

```py
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

prompt = "Hey, how are you doing today?"
description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
```

**Tips**:
* Include the term "very clear audio" to generate the highest quality audio, and "very noisy audio" for high levels of background noise
* Punctuation can be used to control the prosody of the generations, e.g. use commas to add small breaks in speech
* The remaining speech features (gender, speaking rate, pitch and reverberation) can be controlled directly through the prompt

### ✨ Optimizing Inference Speed

We've set up an [inference guide](INFERENCE.md) to make generation faster. Think SDPA, torch.compile and streaming!


https://github.com/huggingface/parler-tts/assets/52246514/251e2488-fe6e-42c1-81cd-814c5b7795b0

## Training

<a target="_blank" href="https://github.com/ylacombe/scripts_and_notebooks/blob/main/Finetuning_Parler_TTS_v1_on_a_single_speaker_dataset.ipynb"> 
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> 
</a>

The [training folder](/training/) contains all the information to train or fine-tune your own Parler-TTS model. It consists of:
- [1. An introduction to the Parler-TTS architecture](/training/README.md#1-architecture)
- [2. The first steps to get started](/training/README.md#2-getting-started)
- [3. A training guide](/training/README.md#3-training)

> [!IMPORTANT]
> **TL;DR:** After having followed the [installation steps](/training/README.md#requirements), you can reproduce the Parler-TTS Mini v1 training recipe with the following command line:

```sh
accelerate launch ./training/run_parler_tts_training.py ./helpers/training_configs/starting_point_v1.json
```

> [!IMPORTANT]
> You can also follow [this fine-tuning guide](https://github.com/ylacombe/scripts_and_notebooks/blob/main/Finetuning_Parler_TTS_v1_on_a_single_speaker_dataset.ipynb) on a mono-speaker dataset example.

## Acknowledgements

This library builds on top of a number of open-source giants, to whom we'd like to extend our warmest thanks for providing these tools!

Special thanks to:
- Dan Lyth and Simon King, from Stability AI and Edinburgh University respectively, for publishing such a promising and clear research paper: [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912).
- the many libraries used, namely [🤗 datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index), [🤗 accelerate](https://huggingface.co/docs/accelerate/en/index), [jiwer](https://github.com/jitsi/jiwer), [wandb](https://wandb.ai/), and [🤗 transformers](https://huggingface.co/docs/transformers/index).
- Descript for the [DAC codec model](https://github.com/descriptinc/descript-audio-codec)
- Hugging Face 🤗 for providing compute resources and time to explore!


## Citation

If you found this repository useful, please consider citing this work and also the original Stability AI paper:

```
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```

```
@misc{lyth2024natural,
      title={Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
      author={Dan Lyth and Simon King},
      year={2024},
      eprint={2402.01912},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## Contribution

Contributions are welcome, as the project offers many possibilities for improvement and exploration.

Namely, we're looking at ways to improve both quality and speed:
- Datasets:
    - Train on more data
    - Add more features such as accents
- Training:
    - Add PEFT compatibility to do Lora fine-tuning.
    - Add possibility to train without description column.
    - Add notebook training.
    - Explore multilingual training.
    - Explore mono-speaker finetuning.
    - Explore more architectures.
- Optimization:
    - Compilation and static cache
    - Support to FA2 and SDPA
- Evaluation:
    - Add more evaluation metrics

