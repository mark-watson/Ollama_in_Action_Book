# Reasoning with Large Language Models

The Chinese tech conglomerate Alibaba's MarcoPolo Team released the advanced **Qwen3.5** models in 2026.

#### Note: As I update this chapter in March 2026, there are many more small reasoning models that you can also use with Ollama; for example: **qwen3.5:2b**, **qwen3.5:4b**, **qwen3.5:9b**, **nemotron-3-nano:4b**, etc. 

These reasoning models are designed to excel in open-ended problem-solving and complex reasoning tasks, tool use, going beyond traditional AI models that focus on structured tasks like coding or math.
## A Simple Example

I very mush enjoy experimenting with NVIDIA’s **nemotron-3-nano:4b** model in the Ollama REPL. Let’s start with a very simple prompt that most models can solve. Here, we want to see the structure of for **Marco-o1**’s CoT (chain of thought) process:

```text
 $ ollama run nemotron-3-nano:4b
>>> Roger has 5 tennis balls. He finds 2 more tennis balls. How many tennis balls does Roger have n
... ow?
Thinking...
We need to answer the simple addition: 5 + 2 = 7. Provide answer.
...done thinking.

Roger now has **7 tennis balls**.

>>> Roger has 5 tennis balls. He finds 2 more tennis balls. How many tennis balls does Roger have n
... ow? Show your thinking process.
Thinking...
We need to answer with thinking process. So we can explain: start with 5, find 2 more => add 
=> 5+2=7. Provide reasoning.
...done thinking.

**Thinking process**

1. **Identify the initial amount** – Roger starts with 5 tennis balls.  
2. **Determine what “finds 2 more” means** – Adding to the existing number.  
3. **Perform the addition** – \(5 + 2 = 7\).  
4. **Interpret the result** – After adding the 2 balls, Roger has 7 tennis balls in total.

**Answer:** Roger now has **7 tennis balls**.
```


## Key Features of Reasoning Models

Here are some key characteristics of reasoning models:

* **Advanced Reasoning Techniques:** It utilizes Chain-of-Thought (CoT) fine-tuning and Monte Carlo Tree Search (MCTS) to enhance its reasoning capabilities. CoT allows the model to trace its thought patterns, making the problem-solving process more transparent. MCTS enables exploration of multiple reasoning paths by assigning confidence scores to different tokens. Reference: [https://arxiv.org/html/2411.14405](https://arxiv.org/html/2411.14405)
* **Self-Reflection:**  A unique feature is its ability to self-reflect, evaluating its reasoning, identifying inaccuracies, and iterating on its outputs for improved results. This leads to higher accuracy and adaptability.
* **Multilingual Mastery:**  Marco-o1 excels in translation, handling cultural nuances, idiomatic expressions, and colloquialisms effectively. This makes it a powerful tool for global communication.
* **Focus on Open-Ended Problems:** Unlike models focused on structured tasks with definitive answers, Marco-o1 tackles open-ended problems where clear evaluation metrics might be absent[1].
* **Strong Performance:** It has shown significant improvements in reasoning and translation benchmarks, including increased accuracy on the MGSM dataset (both English and Chinese) and strong performance in machine translation tasks[1].
* **Open Source Datasets and Implementation:** Alibaba has released Marco-o1's datasets and implementation guides on GitHub, encouraging collaboration and further advancements in AI research.


## A More Complex Example: City Traffic Planning

Here we use a tiny (only 4B parameters model) with no web search to ground advice in real data from the web. If you have a Ollama Cloud account you could try this same example with web search for context grounding using:

```
$ ollama run minimax-m2.5:cloud
```

Let’s end this chapter with a more complex example:

```text
$ ollama run nemotron-3-nano:4b
>>> The city of BisbeeBob is facing a severe traffic congestion problem. The mayor wants to impleme
... nt a new traffic management system to alleviate the issue. However, there are several factors t
... o consider, such as: **Budget constraints:** The city has a limited budget for the project. **P
... ublic opinion:** Citizens have different preferences and concerns regarding traffic management 
... solutions. **Environmental impact:** The new system should minimize its environmental footprint
... . **Technological feasibility:** The chosen technology should be reliable and scalable. **Ethic
... al considerations:** The system should not unfairly disadvantage any particular group of citize
... ns.
Thinking...
We need to produce answer: probably a comprehensive suggestion addressing all factors for 
trauma? The user didn't ask for anything specific? The prompt is just a description, likely 
they expect a plan or framework. So we need to propose a traffic management system meeting 
constraints. Provide structured approach: maybe multi-modal strategy: dedicated bus lanes, 
real-time traffic data, predictive routing, EV incentives, smart signals, budgeting, public 
outreach, environmental, tech feasibility, ethics.

Let's produce a detailed plan covering all points, with steps, implementation timeline, 
budgeting, stakeholder engagement, tech (e.g., BRT, IoT sensors, AI). Also include mitigation 
of ethics (fair access). Provide cost estimate. Provide public opinion survey results? We'll 
assume we can design. Let's answer concisely but thorough.

We'll also mention possible alternative tech: smart traffic signals, BRT lanes, congestion 
pricing, but ensure fairness.

Let's produce final answer.
...done thinking.

Below is a **complete, balanced “traffic‑management” portfolio** that satisfies every 
constraint the mayor listed.  It is organized as a short‑term pilot (12‑18 months) followed 
by a 3‑year scaling plan, so the city can stay within budget, keep public support high, 
protect the environment, operate with reliable technology, and be ethically sound.

---

## 1. Vision & Guiding Principles  

| Goal | How it is achieved |
|------|--------------------|
| **Alleviate congestion** | 2 × BRT lanes + high‑frequency bus service + dynamic routing. |
| **Stay within budget** | Total cost ≈ $6 M (pilot) → $45 M (3 yr). Funding mix: 60 % 
federal/state grants, 30 % city capital budget, 10 % private‑public co‑investment. |
| **Preserve public trust** | Transparent survey, free‑riding eligibility, opt‑out opt‑in 
model for congestion charging. |
| **Minimize environmental footprint** | Low‑emission bus fleet, electric vehicle (EV) 
incentives, carbon‑offset for construction. |
| **Ensure tech reliability & scalability** | OTA‑enabled IoT sensors, AI‑driven signal 
controllers, modular hardware (scalable to 30 % city increase). |
| **No unfair discrimination** | Congestion‑charging only during peak hours, refundable for 
low‑income households; universal bus fares. |

---

## 2. System Components (Pilot Phase)

| Component | Description | Key Benefits | Why it meets all constraints |
|-----------|-------------|--------------|------------------------------|
| **1. Dedicated Bus‑Rapid‑Transit (BRT) Lanes** | Two 12‑ft protected, traffic‑calmed bus 
lanes on the main east‑west artery (e.g., Main St ↔ Central Ave). | Directly removes cars, 
cuts travel time by ~20 %. | Low‑cost (only lane‑marking + signage) → fits budget; reduces 
emissions → environment. |
| **2. Real‑time Traffic & Bus Information System** | Edge‑computing gateway with LoRaWAN 
sensors at intersections → AI platform (traffic volume + bus ETAs). | Citizens see live 
updates (mobile/web), signal operators adjust green time. | OTA‑upgradable → scalable; low 
maintenance cost → budget‑friendly. |
| **3. Adaptive Traffic Signals** | Pre‑programmed “priority” phases for BRT + dynamic 
merging for non‑peak flow. | Faster green for buses/compressed queues; smoother citywide 
flow. | Technically proven (e.g., Siemens City) → reliable. |
| **4. Bicycle & Pedestrian Infrastructure** | Protected bike‑lane, curb‑side sensors for 
bike volume → integrated into AI flow. | Encourages multimodal shift. | Zero‑cost (just 
construction), fully environmentally friendly. |
| **5. Congestion‑Management Feasibility (Fee‑payer option – 3 yr pilot)** | Dynamic pricing 
(toll‑pays, mobile app) for non‑BRT vehicles during 7 am‑7 pm, with revenue split 70 % to 
city, 30 % to fund public‑transit improvements. | Off‑peak users pay a small fee → reduces 
overall traffic volume. | Feasibility: pilot limited to 2 km corridor, revenue modest 
(≈ $300 k/yr) – does not strain budget. <br> Ethics: automatic fee‑waiver for income below 
$30 k/yr via city‑issued QR code; transparent. |
| **6. EV & Fleet Incentive Program** | Subsidies for bus fleet electrification, and $2 k 
rebates for citizen EVs. | Improves environment, reduces noise. | Uses a portion of 
congestion‑fee revenue; city‑matched grants → no net cost increase. |

---

## 3. Budget & Timeline (Pilot – 18 months)

| Phase | Cost (USD) | Funding Source | Key Milestones |
|-------|------------|----------------|----------------|
| **A. Design & Permits** | $0.5 M | City capital (2024) | Finalise BRT corridor, signage, 
signal plans (Month 0‑2). |
| **B. Construction – BRT Lanes & Infrastructure** | $2.0 M | Federal grant (Transit‑Equity) 
| Pavement resurfacing, lane marking, signage (Month 3‑9). |
| **C. Adaptive Signals** | $0.8 M | State grant (Smart Cities) | Sensor install, 
edge‑compute gateway, software (Month 4‑12). |
| **D. BRT Bus Fleet (2 × modern BRT‑type buses)** | $0.4 M | Federal + city match | 
Procurement, training (Month 10‑12). |
| **E. Congestion‑Fee System (pilot)** | $0.2 M | Private‑public partnership | Payment app 
development, QR‑code distribution (Month 11‑12). |
| **F. Bicycle & Pedestrian lanes** | $0.25 M | City budget | Bike racks, surface resurfacing 
(Month 9‑11). |
| **G. Environmental & Equity Offset** | $0.05 M | Carbon‑offset fund | Permaculture plant 
for construction waste, low‑income fee waiver database. |
| **Total Pilot** | **≈ $6.35 M** | — | Launch public launch event (Month 18). |

*Scaled 3‑year plan adds $45 M (additional lanes, full bus fleet, city‑wide congestion 
pricing, 30 % route expansion).*

---

## 4. Public‑Engagement & Ethical Safeguards

| Action | Mechanism | Guarantee of Fairness |
|--------|-----------|-----------------------|
| **Citizen Survey & Advisory Board** | Online questionnaire + 3‑month town‑hall series. | 
Input shapes lane locations, fee levels. |
| **Equitable Fee Waiver** | Income‑verification portal → auto‑waive fees < $30 k/yr. | No 
low‑income rider penalised. |
| **Universal Bus Fares** | $5/day cap (subsidised by congestion‑fee revenue). | Everyone can 
use public transit without extra cost. |
| **Real‑time Dashboard** | Public website showing exact arrival times, congestion levels. | 
Transparency reduces suspicion. |
| **Annual Independent Audit** | Non‑profit environmental/ethics auditor. | Verifies 
financial & impact metrics. |

---

## 5. Technology Feasibility & Maintenance Plan

* **Hardware:** LoRaWAN‑compatible, IP67 waterproof sensors placed at 8 major junctions. 
Powered by solar‑micro‑inverters (off‑grid capability).  
* **Software:** Azure IoT Edge → AI model (Python) runs on a city‑owned edge server; OTA 
updates ≤ 10 % downtime risk.  
* **Redundancy:** Dual redundant power & network links; fail‑over in <5 min.  
* **Scalability:** Adding a new lane only requires sensor + signal upgrade (≈ $0.4 M per 
lane).  

---

## 6. Environmental Impact Summary

| Metric | Projected Reduction |
|--------|----------------------|
| **CO₂ (g/million‑km⁻¹)** | –18 % vs. baseline car‑only flow. |
| **Noise (decibels)** | –4 dB on BRT corridor (quiet streets for pedestrians). |
| **Storm‑water** | 30 % lower runoff due to permeable pavement in lane base. |
| **Biodiversity** | No tree‑line felling; lane protects existing native hedgerows. |

---

## 7. Implementation Checklist (What the mayor needs to sign off)

1. **Funding approval** – $6.35 M pilot budget.
2. **Legal approval** – Right‑of‑way clearance (City Engineer).
3. **Public‑Engagement completion** – Survey ≥ 200 responses, Advisory Board charter signed.
4. **Consultation with utility** – LoRaWAN node integration (Utility consent).
5. **Project‑Management Office (PMO)** – Hire PM with 6‑month critical path for milestone 
compliance.

---

### Bottom Line
The **pilot** invests a modest **$6 M** to deliver **instant, visible congestion relief** 
(BRT lanes + real‑time data), while **maintaining budget flexibility** (grant‑driven), 
**protecting the environment**, **leveraging proven scalable tech**, and **guaranteeing 
ethical, non‑discriminatory operation**.  The longer‑term 3‑year rollout will progressively 
expand the system, keep costs bounded, and let the city continuously adapt based on 
real‑world data and citizen feedback.  

If you need a deeper dive—e.g., a detailed financial model, stakeholder‑map, or a slide‑deck 
for the city council—just let me know!

>>>
```

I often use the state of the art commercial LLM APIs for models like Claude Sonnet 4.6, GPT-5, Gemini 3, etc. to brainstorm ideas and help me think and plan out new projects. I find it exciting to be able to run close to state of the art reasoning LLM on my personal com computer using Ollama!
