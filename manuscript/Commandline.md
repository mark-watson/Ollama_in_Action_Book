# Using Ollama From the Command Line

Working with Ollama from the command line provides a straightforward and efficient way to interact with large language models locally. The basic command structure starts with **ollama run modelname**, where **modelname** could be models like 'llama3’, 'mistral', or 'codellama'. You can enhance your prompts using the -f flag for system prompts or context files, and the --verbose flag to see token usage and generation metrics. For example, ollama run llama2 -f system_prompt.txt "Your question here" lets you provide consistent context across multiple interactions.

One powerful technique is using Ollama's model tags to maintain different versions or configurations of the same base model. For any model on the Ollama web site, you can view all available model tags, for example: [https://ollama.com/library/llama2/tags](https://ollama.com/library/llama2/tags).

The **ollama list** command helps you track installed models, and **ollama rm modelname** keeps your system clean. For development work, the **--format json** flag outputs responses in JSON format, making it easier to parse in scripts or applications; for example:

```bash
$ ollama run qwq:latest --format json
>>> What are the capitals of Germany and France?
{ 
  "Germany": {
    "Capital": "Berlin",
    "Population": "83.2 million",
    "Area": "137,847 square miles"
  },
  "France": {
    "Capital": "Paris",
    "Population": "67.4 million",
    "Area": "248,573 square miles"
  }
}

>>> /bye
```



Advanced users can leverage Ollama's multimodal capabilities and streaming options. For models like llava, you can pipe in image files using standard input or file paths. For example:

```bash
$ ollama run llava:7b "Describe this image" markcarol.jpg
 The image is a photograph featuring a person with short hair who appears to be smiling slightly and looking 
off-camera, towards the left side of the frame. The individual is wearing a light-colored top with some darker 
pattern or design. In the background, there are indistinct objects that give the impression of an outdoor setting, 
possibly on a patio or deck.

The image has a watermark in the lower right corner that reads "markcarol," which suggests that this photograph 
may be by a photographer with that name or a related pseudonym. The style of the image is candid and seems to 
capture a relaxed moment rather than a posed or formal portrait. There are no visible texts other than the 
watermark, and the focus and composition suggest that the photo was taken during the day in natural light. 
```