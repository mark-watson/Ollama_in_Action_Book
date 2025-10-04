# Using Ollama Cloud Services

TBD: why use cloud when Ollama was local only?  TBD


## Ollama Cloud Services: Power and Knowledge on Demand

The Ollama cloud platform is built for developers seeking to run large language models on Ollama's GPU hardware in the cloud, still providing privacy, control, and a streamlined user experience. However, the relentless pace of AI research presents a fundamental challenge: the exponential growth in model size and capability is rapidly outstripping the computational power of consumer-grade hardware. State-of-the-art models, often with hundreds of billions of parameters, are simply too large to fit on widely available GPUs or run at a practical speed. This creates a critical bottleneck for developers and researchers aiming to leverage the latest advancements.   

In response to this challenge, Ollama has introduced Ollama Cloud Services, a strategic and seamless extension of the core ecosystem. This is not a separate product with a new learning curve, but rather a natural evolution designed to preserve the beloved Ollama workflow while unlocking access to state-of-the-art performance. The core purpose of the cloud offering is to allow users to "run larger models faster using datacenter-grade hardware". It provides a direct solution to the hardware gap, ensuring that the Ollama community can continue to innovate at the cutting edge.   

This evolution represents a significant shift from a "local-first" to a "hybrid-first" paradigm. By providing a cloud-based execution environment that is perfectly integrated with the existing Ollama App, CLI, and API, the platform creates a frictionless "escalation path". A developer can prototype an application on their laptop using a smaller, local model and then, with a minimal change—often just a single command-line flag or a line of code—deploy the same application using a massive, high-performance model in the cloud. This approach prevents user attrition to purely cloud-based competitors and transforms Ollama from a local utility into a comprehensive platform that spans the entire development lifecycle, from edge devices to the cloud.   

This chapter will explore the two fundamental pillars of Ollama Cloud Services. The first is High-Performance Model Inference, which provides the raw computational Power to run the largest and most capable models available. The second is the Web Search API, which endows these models with up-to-date Knowledge, allowing them to access real-time information from the internet. Together, these services represent a unified strategy to fundamentally expand what is possible within the Ollama ecosystem.

High-Performance Model Inference in the Cloud

The primary service offered by Ollama Cloud is a high-performance inference engine for running large-scale language models. This service directly addresses the physical limitations of local hardware, providing users with on-demand access to datacenter-grade infrastructure.

The Rationale: Why Move Inference to the Cloud?

The value proposition of moving model inference to the cloud extends beyond simply accessing more powerful hardware. It offers a collection of tangible benefits that enhance productivity, user experience, and application capability.

Speed and Scale: The most immediate advantage is the ability to run inference on models that are otherwise inaccessible. The service leverages newer, more powerful hardware to dramatically speed up response times and makes it possible to run exceptionally large models. As of this writing, available cloud-exclusive models include cutting-edge architectures like (with more expected to be added over time):  

- deepseek-v3.1:671b-cloud
- gpt-oss:120b-cloud
- qwen3-coder:480b-cloud   

Advantages of Ollama Cloud:

- Resource Offloading: Running large models locally is computationally expensive, consuming significant CPU, GPU, and RAM resources, which can slow down a user's entire system. By offloading this workload to the cloud, developers can free up their local machine's resources for other tasks, such as coding, compiling, and running other applications. For mobile users, this also translates directly into significant battery life savings.   
- Privacy Commitment: A primary reason developers choose Ollama is its local-first, privacy-centric design. Moving to a cloud service often raises valid concerns about data privacy and security. Ollama Cloud addresses this head-on with a "Privacy first" policy. The platform explicitly states that it does not log or retain any user queries or data, ensuring that the core principle of user privacy is maintained even when leveraging cloud infrastructure. This commitment is a strategic pillar designed to build and maintain trust with its user base, making the transition from local to cloud a more palatable choice.   
- The service is currently available as a preview for a monthly subscription of $20. To manage capacity, hourly and daily usage limits are in place. All hardware powering the service is located in the United States. Looking forward, Ollama plans to introduce usage-based pricing, signaling an ambition to support more granular, metered consumption for enterprise and high-volume commercial applications.   

Access and Interaction: A Practical Guide

A key design principle of Ollama Cloud is its seamless integration with existing workflows. Whether interacting via the command line or a programmatic API, the process is designed to be intuitive and require minimal changes.

Seamless Command-Line Integration
For users of the Ollama CLI, accessing cloud models is remarkably simple. The process involves two steps:

Authentication: First, you must link your local CLI to your ollama.com account. This is a one-time setup accomplished by running the following command in your terminal:

```bash
ollama signin
```

This will open a browser window to authenticate your account.   

Execution: Once signed in, running a cloud model is nearly identical to running a local one. The only difference is the addition of a -cloud suffix to the model name. For example, to run the 120-billion parameter gpt-oss model, you would execute:

```bash
ollama run gpt-oss:120b-cloud
```

This command instructs Ollama to automatically offload the request to the cloud service. All other command-line flags and interactions behave exactly as they would with a local model, demonstrating the elegance and simplicity of the integration.   

Programmatic API Access: The Remote Host Pattern
For developers building applications, Ollama Cloud can be accessed programmatically through a powerful architectural abstraction known as the "Remote Host Pattern." In this model, the cloud endpoint at https://ollama.com functions as a remote instance of the Ollama server, accepting the same API calls as a local installation.   

To use this pattern, you must first generate an API key from your ollama.com account settings. This key is then used for authentication in your API requests.   

The following Python example demonstrates how to interact with a cloud model. Note how closely it mirrors the code for interacting with a local model:

TBD

The critical change is in the Client constructor, where the host is set to https://ollama.com and the headers dictionary includes the Authorization token. This design makes developer code location-agnostic. The decision of whether to run a model locally or in the cloud can be externalized to a configuration file or an environment variable, allowing for the implementation of sophisticated hybrid execution logic—such as trying a local model first and falling back to the cloud if it's unavailable—without altering the core application code.   

## Augmenting Models with the Web Search API

While the cloud inference service provides computational Power, the Web Search API delivers up-to-date Knowledge. This service is designed to address one of the most significant limitations of large language models: their static knowledge base.

The Rationale: Combating Staleness and Hallucination
LLMs are trained on vast but finite datasets, resulting in a "knowledge cutoff" date. They have no awareness of events, discoveries, or information that emerged after their training was completed. This "staleness" can lead them to provide outdated or incorrect answers. Furthermore, when faced with questions about topics they have limited information on, models can "hallucinate," generating plausible but factually incorrect information.

The Ollama Web Search API is a direct solution to this problem. Its purpose is to "augment models with the latest information to reduce hallucinations and improve accuracy". By providing a simple mechanism to fetch real-time information from the web, developers can ground their models in reality, ensuring their applications provide relevant, timely, and factual responses.   

The Web Search API: A Technical Deep Dive
The Web Search API is a straightforward REST endpoint designed for ease of use. Interaction with the API requires an API key generated from a free Ollama account, which must be passed as a bearer token in the authorization header.   

The endpoint's contract is summarized below.

| Property          | Description |
|-------------------|-------------|
| Endpoint URL      | https://ollama.com/api/web_search |
| HTTP Method       | POST |
| Authentication    | Authorization: Bearer <OLLAMA_API_KEY> header |
| Request Body      | JSON object with a required **query** (string) and an optional *max_results* (integer, default: 5, max: 10) |
| Success Response  | JSON object containing an array of results, each with `title`, `url`, and `content` fields. |
| Failure Response  | JSON object with an `error` field describing the issue (e.g., invalid API key, malformed request, or exceeding max_results). |


## Advanced Application: The Search-Augmented Generation (RAG) Pattern

The true power of the Web Search API is realized when it is used not as a standalone tool, but as the first step in a Retrieval-Augmented Generation (RAG) pipeline. This pattern uses the search results to provide context to a language model, enabling it to generate an informed and grounded response.

This reveals a powerful symbiotic relationship between Ollama's two cloud services. The Web Search API can return thousands of tokens of text content from multiple sources. To make effective use of this rich context, a model with a very large context window is required. Stuffing this much information into a local model with a limited context window (e.g., 4,000 or 8,000 tokens) is often inefficient or impossible.   

This is precisely where the High-Performance Inference service becomes essential. The cloud models are explicitly designed for this type of workload, as they "run at the full context length," which can be orders of magnitude larger than what is feasible on local hardware. For example, a model like    

kimi-k2:1t-cloud is capable of handling enormous contexts, making it an ideal candidate for RAG applications. The Web Search API, therefore, creates a compelling technical reason to use the Cloud Inference service, demonstrating how the two services are designed to function as a single, powerful system.   

A conceptual blueprint for a simple RAG agent using Ollama Cloud would be:

Receive User Query: Start with the user's original question (e.g., "What were the key findings from the latest UN climate report?").

Perform Web Search: Call ollama.web_search() with the user's query to retrieve a list of relevant articles and snippets.

Construct Context: Format the search results into a single string of context. This could be a simple concatenation of the content snippets from the API response.

Augment Prompt: Create a new, detailed prompt for a language model. This prompt should include the formatted context and instruct the model to answer the original user query based solely on the provided information.

Generate Response: Send this augmented prompt to a large-context cloud model (e.g., gpt-oss:120b-cloud) using the client.chat() method.

Return Grounded Answer: The model's response will be grounded in the real-time information retrieved from the web, resulting in an accurate, up-to-date, and non-hallucinated answer.

TBD code.. TDB

## Wrap-Up: A Unified Local and Cloud Strategy

Ollama Cloud Services represent a pivotal expansion of the platform's capabilities, thoughtfully designed to address the growing gap between the demands of modern AI and the limitations of local hardware. This is not merely about offloading computation; it is a unified strategy to fundamentally enhance what developers can build within the Ollama ecosystem.

The key takeaways from this chapter are:

A Seamless Escalation Path: Ollama Cloud provides a frictionless transition from local development to cloud-scale deployment. By maintaining consistency with the existing CLI and API, it preserves the developer experience that has been central to Ollama's success.

Power and Knowledge Combined: The two core services—High-Performance Inference and the Web Search API—are symbiotically linked. The inference service provides the computational Power to run massive models, while the search API provides the real-time Knowledge to make them more accurate and relevant. Together, they enable the creation of sophisticated, knowledgeable, and state-of-the-art AI applications.

A Foundation of Trust: Even while embracing the cloud, Ollama has maintained its foundational commitment to user privacy. The explicit "no log, no retain" policy for cloud queries is a critical feature that builds trust and makes the service a viable option for its privacy-conscious user base.   

Looking ahead, the planned introduction of usage-based pricing signals Ollama's ambition to support a broader range of use cases, from individual developers to large-scale commercial applications. By providing a comprehensive, hybrid platform that spans from the local machine to the cloud, Ollama is solidifying its position as an indispensable tool for the entire lifecycle of AI development.
