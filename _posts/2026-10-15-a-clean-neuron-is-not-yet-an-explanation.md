---
layout: post
title: "A Clean Neuron is Not Yet an Explanation"
date: 2026-10-15
description: "Why progress in reading individual neurons leaves the central questions of mechanistic interpretability open."
categories: [ai]
tags: [language-models, interpretability, neuroscience]
featured: false
---

## The question

Nick Cammarata's [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) makes a claim I would like to be true. Perhaps we have made neural networks look more tangled than they are. A neuron thought to respond separately to cats and cars may actually detect a shape they share. The dog neurons supposedly hiding information about cars may instead be part of a circuit for pose. Better tools and a closer look can correct these mistakes. That is a substantial finding. It challenges the assumption that every confusing neuron must be decomposed by a sparse autoencoder (SAE).

There is independent evidence for taking neurons seriously. [Arora et al. (2026)](https://proceedings.mlr.press/v306/arora26c.html) use MLP neurons to trace causal circuits for subject–verb agreement and a multi-hop factual task. If the model's own neurons can answer a question, we should study them.

But a stronger conclusion is spreading alongside this result. It says that superposition was largely a mistake, that neuron and circuit maps are close to explaining modern models, and that the rest is a matter of scaling up investigation. I think this changes the question halfway through. We may learn to describe most neurons coherently without knowing which variables and operations explain a model's behavior.

That distinction matters for research direction. I hope more people look at neurons. The concern is that their increasing readability will make other levels of explanation seem optional.

## Mechanism and evidence

### 1. Reinterpreting a famous neuron does not adjudicate superposition in general

The cat-and-car neuron appeared to mix unrelated categories. If it detects a pointed shape instead, that interpretation fails. The proposed car-into-dog circuit may likewise have been distorted by misleading visualizations. These corrections matter.

Whether they refute *superposition* depends on what we mean by the term. Cammarata focuses on unrelated functions sharing a **neuron** because they rarely interfere. The influential [toy-model paper](https://transformer-circuits.pub/2022/toy_model/index.html) studied a different possibility, with sparse features occupying overlapping directions in a space with fewer dimensions than features. Its experiments show that this can happen in a toy setting. They do not tell us how common it is in frontier models. Likewise, coherent responses from many individual neurons do not rule out distributed variables in the residual stream.

Anthropic's [preliminary work on attention](https://www.transformer-circuits.pub/2025/attention-update/index.html) makes this distinction concrete. A decomposition that let features use multiple heads was more efficient at comparable reconstruction error than one that forced each feature into a single head. The authors consider this evidence for attention superposition, though they stop short of claiming a definitive example in a real model. Cammarata also says attention is less understood than neurons. Findings about **MLP neuron poles** cannot settle how attention or the residual stream is organized.

Perhaps these forms of overlap are harmless if the computation remains readable. We need to test that. When a task variable appears distributed, can a neuron-level account predict how the model uses it and what happens when we change it? If so, native neurons have taken us a long way. If not, clean individual neurons have left an important question open.

### 2. The relevant unit depends on the question being explained

Neuroscience offers a useful parallel. In prefrontal cortex, [Rigotti et al. (2013)](https://www.nature.com/articles/nature12160) found neurons responding to combinations of task conditions. Yet the population still carried information about each condition after the classically selective cells were excluded. In a different decision task, [Mante et al. (2013)](https://www.nature.com/articles/nature12742) found that population dynamics made sense of signals mixed within individual neurons. Single neurons mattered in both studies. Their tuning alone did not explain the computation.

Language models let us test the distinction more directly. [Distributed alignment search](https://proceedings.mlr.press/v236/geiger24a.html) can find causal variables in directions that do not coincide with individual neurons. In the [MIB benchmark](https://proceedings.mlr.press/v267/mueller25a.html), SAE features did not outperform native neurons on the tested causal-variable tasks. The supervised distributed alignment method did best. The best unit of explanation depends on the causal question.

Imagine a neuron that responds to a social situation and changes the model's output when steered. It participates in the computation. But is the relevant variable carried by that neuron alone, by a population, or by a state that changes with context? All three accounts might fit the neuron's strongest activations and a simple steering result. Changing the context or selectively removing information could separate them.

### 3. A causal effect does not yet specify a causal role

Cammarata examines connections, intervenes on neurons, and follows attribution graphs. His evidence is much stronger than a list of activating inputs. Still, an intervention shows that a component matters **under a particular manipulation**. It does not yet tell us what operation that component performs.

Take his example in which removing four neurons changes a model's medical follow-up question. The neurons clearly matter. But are they tracking a diagnosis, combining symptoms, or controlling when a medical rule is used? We can vary the symptoms and alternative diagnoses, then check which account predicts both the model's answer and its intermediate states. A readable graph for one prompt helps us form these hypotheses. It cannot choose among them on its own.

The same standard applies to SAE-based work. Anthropic's [circuit-tracing methods paper](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html) distinguishes a graph's readability from its *mechanistic faithfulness*. It reports that agreement with the original model under perturbation worsens across layers. A graph made from native neurons avoids the reconstruction error of replacing MLPs. It still needs tests of the causal roles assigned to its nodes and paths.

### 4. The optimistic extrapolation outruns the current evidence

The essay reports that about 90% of neuron poles in a preliminary Qwen3-32B audit have one main job, and another 7% have two related jobs. A detailed technical paper is forthcoming. What counts as one job involves judgment, and attention has received less study. The examples from larger models are intriguing. They are still some distance from a comprehensive account of how those models work.

Even if those numbers hold up, the denominator matters. The fraction of neurons we can describe is not the fraction of behavior we can explain. A consequential computation could depend on a small remainder or on interactions among well-described neurons. We should therefore sample behaviors and interventions, not just units. Can an account predict what the model will do on new inputs and under changes it has not seen?

The essay compares a growing map to a crossword that gets easier as more squares are filled. That may be true for *discovery*. Several coherent maps could still fit the same local observations. We learn which map is better by testing where their predictions differ. The crucial progress measure is how well an account survives questions it was not built to answer.

## Implications

The essay makes a strong case for studying neurons directly. It does not show that a census of neurons and local circuits will answer every mechanistic question. Native components, learned features, population geometry, and causal abstractions can each reveal something about the same computation. We should judge them by what they explain and predict.

In our recent [position paper](https://openreview.net/forum?id=8Duo4yczA6), we argue that the choice of explanatory unit is itself a hypothesis. We then ask whether the information it carries is used, what causal role it plays, how that role depends on other components, and which parts of the account remain stable across settings. The measurements behind each claim matter. This way of thinking helps us catch the step where an observation is asked to support more than it can.

The most useful experiment prompted by *Already Clean* would compare these levels directly. Choose a behavior whose context and relevant variables can be controlled. Build the strongest account using native neurons and their connections. Then test whether a population-level variable improves its predictions under new interventions. See what transfers across prompts and models, and where each account fails. [Work on circuit consistency across training and scale](https://proceedings.neurips.cc/paper_files/paper/2024/hash/47c7edadfee365b394b2a3bd416048da-Abstract-Conference.html) shows why this is a practical question.

The important question goes beyond whether superposition “exists.” Which parts of a computation are local? Which are distributed? How is information transformed between them? Cleaner neurons would give us a better microscope for answering these questions. We would still need to work out which theory of the system fits what we see.
