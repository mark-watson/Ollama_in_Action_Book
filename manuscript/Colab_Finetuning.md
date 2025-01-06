# Using the Unsloth Library on Google Colab to FineTune Models for Ollama

This is a book about running local LLMs using Ollama. That said, I use a Mac M2 Pro with 32G of memory and while my computer could be used for fine tuning models, I prefer using cloud assets. I frequently use Google’s Colab for running deep learning and other experiments.

We will be using three Colab notebooks in this chapter:

- Colab notebook 1: [Colab URI](https://colab.research.google.com/drive/1rCeF7UVZpAkXg1PuGRH6-o_FE6pzXn19#scrollTo=c0HzYFUopDdH) for this chapter is a modified copy of a [Unsloth demo notebook](https://colab.research.google.com/drive/1cTcNv6rD9UZB0bymb2wyAJdTXL15Y6m8). Here we create simple training data to quickly verify the process of fine tuning on Collab using Unsloth and exporting to a local Ollama model on a laptop. We fine tune the 1B model **unsloth/Llama-3.2-1B-Instruct**.
- Colab notebook 2: [Colab URI](https://colab.research.google.com/drive/1uJQx7bx3eQYqyBIM0HdRiE5QtgwK3KVH?usp=sharing) uses my dataset on fun things to do in Arizona. We fine tune the model **unsloth/Llama-3.2-1B-Instruct**.
- Colab notebook 3: [Colab URI](https://colab.research.google.com/drive/1uNlW2S4_3LxxpBZ4jFcvKyc-djHzqpe5?usp=sharing) This is identical to the example in Colab notebook 2 except that we fine tune the larger 3B model **unsloth/Llama-3.2-3B-Instruct**.

The Unsloth fine-tuning library is a Python-based toolkit designed to simplify and accelerate the process of fine-tuning large language models (LLMs). It offers a streamlined interface for applying popular techniques like LoRA (Low-Rank Adaptation), prefix-tuning, and full-model fine-tuning, catering to both novice and advanced users. The library integrates seamlessly with Hugging Face Transformers and other prominent model hubs, providing out-of-the-box support for many state-of-the-art pre-trained models. By focusing on ease of use, Unsloth reduces the boilerplate code needed for training workflows, allowing developers to focus on task-specific adaptation rather than low-level implementation details.

One of Unsloth’s standout features is its efficient resource utilization, enabling fine-tuning even on limited hardware such as single-GPU setups. It achieves this through parameter-efficient fine-tuning techniques and gradient check pointing, which minimize memory overhead. Additionally, the library supports mixed-precision training, significantly reducing computational costs without compromising model performance. With robust logging and built-in tools for hyper parameter optimization, Unsloth empowers developers to achieve high-quality results with minimal experimentation. It is particularly well-suited for applications like text summarization, chatbots, and domain-specific language understanding tasks.

## Colab Notebook 1: A Quick Test of Fine Tuning and Deployment to Ollama on a Laptop

We start by installing the Unsloth library and all dependencies, then uninstalling just the sloth library and reinstalling the latest from source code on GitHub:

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

### Warning on Limitations of this Example

We used very little training data and in the call to **SFTTrainer** we didn't even train one epoch:

```
    max_steps = 60, # a very short training run for this demo
    # num_train_epochs = 1, # For longer training runs!
```

This allows us to fine tune a previously trained model very quickly for this short demo.

We will use much more training data in the next chapter to finetune a model to be an expert in recreational locations in the state of Arizona.

### Save trained model and tokenizer to a GGUF File on the Colab Notebook's File System

To experiment in the Colab Notebook Linux environment we can save the data locally:

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

### Copying the GGUF File to Your Laptop and Creating a Ollama Modelfile

Depending on how fast your Internet speed is, it might take five or ten minutes to download the GGUF file since it is about 1G in size:

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

After downloading the GGUF file to my laptop I made a slight edit to the generated **Modelfile** got the path to the GGUF file on line 1:

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

Once the model is downloaded to your laptop, create a local Ollama model to use:

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
>>> /bye
```

Notice that fine tuned model has learned new data and still has functionality of the original model.

### Fine Tuning Test Wrap Up

This was a short example that can be run on a free Google Colab notebook. Now we will use a larger fine tuning training data set.

## Fine Tuning Using a Fun Things To Do in Arizona Data Set

Here we use the slightly modified [Colab notebook 2](https://colab.research.google.com/drive/1uJQx7bx3eQYqyBIM0HdRiE5QtgwK3KVH?usp=sharing).

I created a GitHub repository for the [Arizona fine tuning data set](https://github.com/mark-watson/llm_datasets) that contains small individual JSON files and a larger file **ArizonaFun.json** that is a concatenation of the smaller files. Let's look at **az_flagstaff_parks.json** (edited to remove some text for brevity):

```json
[
 {
    "conversations": [
        {
            "content": "Can you list some city parks in Flagstaff, Arizona?",
            "role": "user"
        },
        {
            "content": "Here are some city parks in Flagstaff, Arizona:\n\n- **Bushmaster Park**: Located at 3150 N Alta Vista Dr, this 20-acre park features playgrounds, lighted tennis and basketball courts, a skate park, roller skating arena, dog park, picnic ramadas, and permanent restrooms.\n\n- **Thorpe Park**: Situated at 191 N Thorpe Rd, Thorpe Park offers playgrounds, lighted tennis courts, sand volleyball court, basketball court, soccer fields, youth baseball fields, adult softball fields, a disc golf course, dog park, picnic ramada, and both portable and permanent restrooms.\n\n- **Foxglenn Park**: Found at 4200 E Butler Ave, this park includes a playground, basketball courts, soccer fields, a youth softball field, skate track, picnic ramadas, and permanent restrooms.\n\n- **Buffalo Park**: Located at 2400 N Gemini Rd, Buffalo Park is a 215-acre open space with a 2-mile loop trail, ...",
                "role": "assistant"
            }
        ]
    }, ...
```
There are a total of 40 fine tuning examples in the file **ArizonaFun.json**. You can see in the second and third Colab notebooks for this chapter I just pasted the JSON data from the file **ArizonaFun.json** into a cell:

```python
from datasets import Dataset

json_data = [
    {
        "conversations": [
            {
                "content": "Can you list some city parks in Flagstaff, Arizona?",
                "role": "user"
            },
        ... } ]
```

Unfortunately, the fine tooled model often performs well, but also hallucinates. Here is an example of using the fine tuned model in the Colab notebook:

```python
messages = [                    # Change below!
    {"role": "user", "content": "Where is Petrified Forest National Park located?"},
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

The output is:

```
Petrified Forest National Park is situated in northeastern Arizona, near the town of Holbrook. [oai_citation_attribution:2‡National Park Service](https://www.nps.gov/state/az/index.htm)<|eot_id|>
```

This answer is correct.

The second Colab notebook also contains code cells for downloading the fine tuned model and the directions for importing the model into Ollama that we saw earlier also apply here.

## Third Colab Notebook That Fine Tunes a Larger Model

There are only two changes made to the second notebook:

- We now fine tune a 3B model **unsloth/Llama-3.2-3B-Instruct**.
- Because the fine tuned model is large, I added code to store the model in Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.move("/content/model/unsloth.Q4_K_M.gguf", '/content/drive/My Drive/LLM/')
```

I created an empty folder **LLM** on my Google Drive before running this code.

## Fine Tuning Wrap Up

I don't usually fine tune models. I usually use larger prompt contexts and include one shot or two shot examples. That said there are good use cases for fine tuning small models with your data and I hope the simple examples in this chapter will save you time if you have an application requiring fine tuning.


