# Using the Unsloth Library on Google Colab to FineTune a 1B Model for Ollama Using a Arizona Activities Dataset

This example is similar to that in the last chapter but here we use fine tuning data that I created for Arizona activities, specifically for fun things to do in Arizona:

- National Parks
- State Parks
- Rivers
- Lakes
- Parks in Phoenix
- Parks in Flagstaff

We will be using three Colab notebooks. The [Colab notebook](https://colab.research.google.com/drive/1rCeF7UVZpAkXg1PuGRH6-o_FE6pzXn19#scrollTo=c0HzYFUopDdH) for this chapter is a modified copy of a [Unsloth demo notebook](https://colab.research.google.com/drive/1cTcNv6rD9UZB0bymb2wyAJdTXL15Y6m8)



## Details of Notebook

We start by installing the unsloth library and all dependencies, then unintstalling just the sloth library and reinstalling the latest from source code on GitHub:

```
pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

Now create a model and tokenizer:

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection.
load_in_4bit = True # Use 4bit quantization to reduce memory usage.

# More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

Now add LoRA adapters:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```

The original Sloth example notebook used Maxime Labonne's FineTome-100k dataset for fine tuning data. Since I wanted to fine tune with my own test data I printed out some of Maxime Labonne's data after being loaded into a **Dataset** object. Here are a few snippets to show you, dear reader, the format of the data that I will reproduce:

```json
{'conversations': [{'content': 'Give three tips for staying healthy.', 'role': 'user'}, {'content': '1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats...', 'role': 'assistant'},
 ...
 ]}
{'conversations': [{ ... etc.
{'conversations': [{ ... etc.
```

I used a small Python script on my laptop to get the format correct for my test data:

```python
from datasets import Dataset

json_data = [
 {"conversations": [
    {"content": "What are the two partitioned colors?", "role": "user"},
    {"content": "The two partitioned colors are brown, and grey.",
     "role": "assistant"},
    {"content": "What are the two partitioned colors?", "role": "user"},
    {"content": "The two partitioned colors are brown, and grey.",
     "role": "assistant"}
 ]},
 {"conversations": [
    {"content": "What is the capital of Underworld?", "role": "user"},
    {"content": "The capital of Underworld is Sharkville.",
     "role": "assistant"}
 ]},
 {"conversations": [
    {"content": "Who said that the science of economics is bullshit?",
     "role": "user"},
    {"content": "Malcom Peters said that the science of economics is bullshit.",
     "role": "assistant"}
 ]}
]

# Convert JSON data to Dataset
dataset = Dataset.from_list(json_data)

# Display the Dataset
print(dataset)
print(dataset[0])
print(dataset[1])
print(dataset[2])
```

Output is:

```
Dataset({
    features: ['conversations'],
    num_rows: 3
})
{'conversations': [{'content': 'What are the two partitioned colors?', 'role': 'user'}, {'content': 'The two partitioned colors are brown, and grey.', 'role': 'assistant'}, {'content': 'What are the two partitioned colors?', 'role': 'user'}, {'content': 'The two partitioned colors are brown, and grey.', 'role': 'assistant'}]}
{'conversations': [{'content': 'What is the capital of Underworld?', 'role': 'user'}, {'content': 'The capital of Underworld is Sharkville.', 'role': 'assistant'}]}
{'conversations': [{'content': 'Who said that the science of economics is bullshit?', 'role': 'user'}, {'content': 'Malcom Peters said that the science of economics is bullshit.', 'role': 'assistant'}]}
```

If you look at the [notebook for this chapter on Colab](https://colab.research.google.com/drive/1rCeF7UVZpAkXg1PuGRH6-o_FE6pzXn19#scrollTo=8hyZKqDhhtpZ) you will see that I copied the last Python script as-is to the notebook, replaces code in the orgiginal Unsloth demo notebook.

The following code (copied from the Unsloth demo notebook) slightly reformats the prompts and then trains using the modified dataset:

```python
chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

from unsloth import apply_chat_template
dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # for short segments
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# Now run the trained model on Google Colab with question
# from fine tuning data:

FastLanguageModel.for_inference(model)
messages = [ 
    {"role": "user",
     "content": "What are the two partitioned colors?"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)
```

The output is (edited for brevity and to remove a token warning):

```
The two partitioned colors are brown, and grey.
```

The notebook has a few more tests:

```python
messages = [                    # Change below!
    {"role": "user", "content": "What is the capital of Underworld?"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids,
                   streamer = text_streamer,
                   max_new_tokens = 128,
                   pad_token_id = tokenizer.eos_token_id)
```

The output is:

```
The capital of Underworld is Sharkville.<|eot_id|>
```

## Warning on Limitations of This Example

We used very little training data and in the call to **SFTTrainer** we didn't even use parameters to train one epoch:

```
    max_steps = 60, # a very short training run for this demo
    # num_train_epochs = 1, # For longer training runs!
```

This allows us to fine tune a previously trained model very quickly for this short demo.

We will use much more training data in the next chapter to finetune a model to be an expert in recreational locations in the state of Arizona.

## Save trained model and tokenizer to a GGUF File on the Colab Notebook's File System

Toe experiment in the Colab Notebook Linux environment we can save the data locally:

```
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
```

In order to create a GGUF file to allow us to run this fine tuned model on our laptop we create a local GGUF file that can be downloaded to your laptop:

```python
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
```

In the demo notebook, you can see where the GGUF file was written:

```
!ls -lh /content/model/unsloth.Q4_K_M.gguf
771M 771M Dec  5 15:51 /content/model/unsloth.Q4_K_M.gguf
```



## Copying the GGUF File to Your Laptop and Creating a Ollama Modelfile

Depending on how fast your Internet speed is, it might take five or ten minutes to download the GGUF file since it is &&1M in size:

```python
from google.colab import files
files.download("/content/model/unsloth.Q4_K_M.gguf")
```

We also will need to copy the generated Ollama model file (that the Unsloth library created for us):

```
!cat model/Modelfile
```

The contents of the file is:

```
FROM /content/model/unsloth.F16.gguf

TEMPLATE """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.{{ if .Prompt }}

### Instruction:
{{ .Prompt }}{{ end }}

### Response:
{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|finetune_right_pad_id|>"
PARAMETER stop "<|python_tag|>"
PARAMETER stop "<|eom_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token_"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
```

After downloading the GGUF file to my laptop I made a slight edit to the generated Modelfile got the path to the GGUF file on line 1:

```
FROM ./unsloth.Q4_K_M.gguf

TEMPLATE """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.{{ if .Prompt }}

### Instruction:
{{ .Prompt }}{{ end }}

### Response:
{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|finetune_right_pad_id|>"
PARAMETER stop "<|python_tag|>"
PARAMETER stop "<|eom_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token_"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
```

Create a local Ollama model to use:

```
$ ls -lh
-rw-r--r--  1 markw  staff   580B Dec  5 09:26 Modelfile
-rw-r--r--@ 1 markw  staff   770M Dec  5 09:19 unsloth.Q4_K_M.
$ ollama create unsloth -f Modelfile
```

I can now use the model **unsloth** that was just created on my laptop:

```
$ ollama run unsloth                
>>> what is 2 + 5?
two plus five equals eight.

>>> What are the two partitioned colors?
The two partitioned colors are brown, and grey.

>>> Who said that the science of economics is bullshit?
Malcom Peters said that the science of economics is bullshit.

>>> write a Python program to sum and print a list of numbers
```python
# list of numbers
numbers = [1, 2, 3, 4, 5]

# use the built-in function sum()
sum_of_numbers = sum(numbers)

# print the sum
print(sum_of_numbers)
```

>>> /bye
```

Notice that finetuned model has learned new data and still has functionality of the original model.

## Finetuning Example Wrap Up

This was a short example that can be run on a free Google Colab notebook. In the next chapter we look at a more real world fine tuning example that I created using the 