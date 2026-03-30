# Using AI Coding Agents with Local Ollama Models and Ollama Cloud

You don’t need a high end home computer to run AI coding agents like Claude Code, OpenCode, or pi with local models. Before we get into the technical details and examples, you must always increase the context size when running Ollama. There are a few ways to do this but the easiest is to set an environment variable in the context of the process running Ollama. For example:

```
$ OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

If you fail to increase the default context size the AI coding agents will lose the context information of open files, which files it is editing on your behalf, and your instructions.

## Ideas for Setting Up Your Distributed Environment (Optional Material)

If you work on a single laptop or desktop computer, then please skip this section.

I run a 32G Mac Mini “headless” as a server and usually use a 16G MacBook Air for coding and writing. I use the Tailscale service to access my 32G Mac Mini server.

About once a week I attach my 32G Mac Mini to a keyboard/trackpad/monitor to run software updates and to run `ollama serve` in a terminal (the following is all on one line):

```
OLLAMA_HOST=0.0.0.0 OLLAMA_CONTEXT_LENGTH=32768 ollama serve
```

Setting a larger than default context length is crucial. For example running Claude Code using a Ollama hosted model fails with a smaller context.

It is important to override the default Ollama host IP address but this does open up your computer to incoming connections if you don’t keep the default Apple firewall settings enabled. Please read the firewall and security documentation for your computer (macOS, Windows, and Linux). On macOS check `System Settings > Network > Firewall`.

I like the free [tailscale service](https://tailscale.com) that allows me to access my 32G Mac Mini via **SSH** from any location from my iPad or my MacBook Air.

I can **SSH** or run `ollama` or other command line programs from my laptop by setting the OLLAMA_HOST address assigned by Tailscale for my 32G Mac Mini (set the IP address Tailscale gives you, not the ‘dummy’ one I show here, and replace my account name `mark` with a valid account name on your remote server):

```
$ OLLAMA_HOST=100.122.241.16 ollama list
$ ssh mark@100.122.241.16
```

I wrote a short article with more information about my distributed setup that you can read online: [https://substack.com/home/post/p-185649162](https://substack.com/home/post/p-185649162).

If I want to use a larger model from my laptop then I still run from my laptop but override where the `ollama` command line tool looks for the Ollama service (this is all on one line):

```
OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_HOST=100.122.241.16 ollama launch pi --model qwen3.5:27b
```

Now I am using the Ollama service on my Mac Mini to run a 27B model (that won’t run on my laptop.) 

Compare this to running a small 9B model on my laptop:

```
OLLAMA_CONTEXT_LENGTH=16384 ollama launch pi --model qwen3.5:9b
```

In the rest of this chapter I will usually be running the examples on my laptop.

## Security and Privacy Considerations

While I personally feel comfortable with security and privacy issues for running commercial AI coding agents like Google Antigravity and gemini-cli, Claude Code, and OpenAI codex - I am personally more “nervous” and careful when running open source AI coding agents.

You can make running AI agents more secure by:

- Not giving them access to any authentication tokens, passwords, and not giving them access to services like Google Workplace or Office 365.
- On macOS, Linux, or Windows consider using AI coding agents with a different user account that only has access to the source code of your own projects you want to work on. Using Docker is another alternative.
- Be careful of copying other people’s SKILLS.md and other customizations for agents unless you inspect these files yourself.

### Still “Nervous” About Security and Privacy Considerations? Use the Commercially Maintained Claude Code

For using Claude Code with Ollama models (not using Anthropic’s models) set these environment variables and an alias:

```
export ANTHROPIC_AUTH_TOKEN=ollama

export ANTHROPIC_BASE_URL=http://127.0.0.1:11434

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

alias CLAUDE=’~/.local/bin/claude --model glm-4.7-flash’
```

The model [glm-4.7-flash](https://ollama.com/library/glm-4.7-flash) is one of the strongest 32B class models.

Here is an example (edited for brevity):

```
$ CLAUDE

 Accessing workspace:

 /Users/markwatson

 Quick safety check: Is this a project you created or one you trust? (Like your own code, a
 well-known open source project, or work from your team). If not, take a moment to review
 what's in this folder first.

 Claude Code'll be able to read, edit, and execute files here.

 Security guide

 ❯ 1. Yes, I trust this folder
   2. No, exit

 Enter to confirm · Esc to cancel
Marks-MacBook-Air:~ $ g
Marks-MacBook-Air:GITHUB $ cd kgn 
Marks-MacBook-Air:kgn $ CLAUDE

╭─── Claude Code v2.1.62                                                     │ Tips for getting started     
│                 Welcome back Mark!
    Run /init to create a CLAUDE.md file …
│ Recent activity             
No recent activity         
│         glm-4.7:cloud · Claude API ·                                         │
│         markwatson.com's Organization                                         │
│                    ~/GITHUB/kgn                                        
❯ 
```

You can also launch Claude Code directly from the `ollama` command line tool:

```
ollama launch claude --model glm-4.7-flash
```

You can also directly run OpenAI’s codex and the Open Source OpenCode project:

```
ollama launch codex --model glm-4.7-flash
ollama launch opencode --model glm-4.7-flash
```

Note: This assumes I am running on a computer with 32G or memory. If I was running this on my remote 32B Mac Mini I would use (this is all on one line):

```
OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_HOST=100.122.241.16 ollama launch claude --model glm-4.7-flash
``` 

## Using Larger Models On Ollama Cloud

Dear reader, you learned how to run models on the Ollama Cloud services in the last chapter.

Here we just list a few examples for starting various AI coding agents using the Ollama Cloud services.

I will vary the examples between launching Claude Code (claude), OpenAI Codex (codex), or OpenCode (open code), but any of these coding agents work with the following model recommendations.

### minimax-m2.7:cloud

MiniMax's M2-series model are excellent for coding and agentic workflows. [Read the model documentation](https://ollama.com/library/minimax-m2.7).

Try:

```
ollama launch claude --model minimax-m2.7:cloud
```


### qwen3.5:397b-cloud

Alibaba’s Qwen 3.5 is a family of open-source multimodal models that are very good for coding and also general use. [Read the model documentation](https://ollama.com/library/qwen3.5).

```
ollama launch codex --model qwen3.5:397b-cloud
```

### nemotron-3-super:cloud

NVIDIA’s Nemotron 3 Super is a 120B open MoE model activating just 12B parameters to deliver maximum compute efficiency. This model was designed to be both cost effective and powerful. [Read the model documentation](https://ollama.com/library/nemotron-3-super).

```
ollama launch opencode --model nemotron-3-super:cloud
```


