---
layout: post
title: "Using Mechanistic Interpretability for Cognitive Science"
date: 2026-10-13
description: "What can looking inside artificial neural networks teach us about computation, intelligence, and human cognition?"
categories: [cognition, ai]
tags: [mechanistic-interpretability, cognitive-science, neural-networks, neuroai]
featured: false
---

*Based on a guest lecture for CS 7651: Human and Machine Learning at Georgia Tech, October 12, 2026.*

Neural networks can recognize objects, use language, play games, and sometimes solve problems that seem to require planning or reasoning. We have become remarkably good at measuring **what** they can do. We are much less certain about **how** they do it.

That difference is the starting point for mechanistic interpretability (MI). It is also the source of a question I have been thinking about for some time: **Can studying the mechanisms of artificial neural networks teach us anything fundamental about cognition?**

This post follows three questions from my lecture:

1. Why study the internal mechanisms of neural networks rather than their behavior alone?
2. What can we learn from neurons, features, circuits, representational geometry, and dynamics?
3. How can an observation about a network become a mechanistic explanation—and, potentially, a scientific insight about cognition?

I will use a few examples, from a network that learns to play Othello to a language model that plans the ending of a poem. Toward the end, I will return to the harder question: what, if anything, do discoveries about *artificial* computation imply about *human* cognition?

## 1. Why look inside a neural network?

Imagine two people answering the same arithmetic problem:

**36 + 59 = 95.**

One computes the units and tens separately, carrying the extra ten. Another notices that 59 is almost 60, computes 36 + 60, and subtracts 1. Both give the same answer. If we observed only their inputs and outputs, we could not tell which procedure they used.

The same ambiguity appears in neural networks. Good performance on a task establishes that a system *can* perform the task, but not which representation or algorithm it uses. Two models might achieve similar accuracy by relying on very different information and computations. Conversely, similar-looking internal components might play different roles in different systems.

This does not make behavioral testing unimportant. Behavior defines the phenomenon we want to explain, and carefully designed behavioral experiments can distinguish many hypotheses. But **input–output agreement rarely identifies a unique internal mechanism**. If we want to understand how a learned system computes, internal evidence gives us another set of constraints.

One motivation for studying mechanisms is practical: perhaps understanding a model will help us diagnose failures, predict generalization, or intervene more reliably. A second motivation is scientific: learned networks may provide clues about which computations can support memory, abstraction, or reasoning.

I also think there is a simpler motivation. Artificial neural networks are fascinating computational systems. We have constructed the learning procedures that produce them, but we often do not understand the resulting organization. It is reasonable to want to study them **simply because they are there**.

### A broad view of mechanistic interpretability

When people hear *mechanistic interpretability*, they often think of attention-head circuits, sparse autoencoders, or the LLM-focused research tradition around reverse-engineering Transformers. Those are important examples, but I prefer a broader definition:

> **Mechanistic interpretability studies how neural networks implement their behavior through internal representations, transformations, and interactions.**

Under this view, neuron analysis, causal interventions, matrix factorization, population geometry, and dynamical-systems analysis can all contribute to mechanistic understanding. There is no reason for MI to be restricted to language models, or to a particular class of interpretability tools.

However, there is an equally important boundary: **not every interpretable observation is already a mechanism**. Finding a feature that correlates with a concept, or a component whose removal hurts performance, is evidence. What that evidence establishes requires a separate argument.

This distinction motivates much of the rest of the post.

## 2. Five ways to look inside

An artificial neural network has many possible explanatory levels. Choosing one is not merely a matter of convenience: it is already a hypothesis about where the relevant computation is organized.

### Neurons: What does a unit respond to?

A natural starting point is the individual neuron. In computer vision, [Network Dissection](https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html) (Bau et al., 2017) quantified whether individual convolutional units responded to recognizable visual concepts such as objects, parts, materials, or textures.

The intuition is simple. For a given image, a convolutional unit produces a spatial activation map. We threshold that map to identify where the unit responds strongly, compare it with a human-labeled segmentation mask, and quantify their overlap—for example, using intersection over union (IoU). If a unit repeatedly activates over windows, trees, or other meaningful regions, we can describe its selectivity.

<figure style="max-width:890px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/network-dissection.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/network-dissection.png' | relative_url }}" alt="Network Dissection illustrates how units can be matched to labeled visual concepts. Example images are drawn from the lecture slide." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">Network Dissection illustrates how units can be matched to labeled visual concepts. Example images are drawn from the lecture slide. <a href="https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html" target="_blank" rel="noopener noreferrer">Source: Bau et al. (2017), Network Dissection</a>. Click to enlarge.</figcaption>
</figure>


A complementary technique is **activation maximization**. Instead of presenting the network with an ordinary image, we optimize an input image to make a selected neuron respond strongly. The interactive [*Feature Visualization* article in Distill](https://distill.pub/2017/feature-visualization/) (Olah, Mordvintsev, & Schubert, 2017) contains some of the clearest demonstrations: moving through a visual network, one encounters patterns resembling edges, textures, parts, and increasingly complex objects.

<figure style="max-width:920px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/feature-visualization.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/feature-visualization.png' | relative_url }}" alt="Optimized images reveal patterns that strongly activate selected units, from textures to recognizable object-like features." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">Optimized images reveal patterns that strongly activate selected units, from textures to recognizable object-like features. <a href="https://distill.pub/2017/feature-visualization/" target="_blank" rel="noopener noreferrer">Source: Olah, Mordvintsev &amp; Schubert (2017), Feature Visualization</a>. Click to enlarge.</figcaption>
</figure>


These are useful observations. Yet a neuron that responds to trees need not be *the mechanism* for recognizing trees. Selectivity alone does not establish what downstream computations do with its activity.


### Features: Information may not live in one neuron

Neural representations can be distributed. A concept may correspond to a direction, subspace, or pattern of activation across many neurons, rather than to one dedicated unit.

This motivates **dictionary learning** and **sparse autoencoders (SAEs)**. Roughly, an SAE tries to express a high-dimensional activation vector as a sparse combination of learned feature directions:

$$
\mathbf{h}\approx D\mathbf{z},\qquad \mathbf{z}\text{ sparse}.
$$

The hope is that the learned coordinates are more interpretable than the original neuron axes. One striking example in Anthropic's [*Mapping the Mind of a Large Language Model*](https://www.anthropic.com/research/mapping-mind-language-model) (2024) is a feature associated with the Golden Gate Bridge, activated by references to the bridge across languages and modalities.

<figure style="max-width:610px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/sae-schematic.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/sae-schematic.png' | relative_url }}" alt="A schematic sparse-autoencoder decomposition: activations are encoded in a larger, sparse feature space and then reconstructed." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">A schematic sparse-autoencoder decomposition: activations are encoded in a larger, sparse feature space and then reconstructed. <a href="https://www.anthropic.com/research/mapping-mind-language-model" target="_blank" rel="noopener noreferrer">Source: SAE schematic from the lecture slide</a>. Click to enlarge.</figcaption>
</figure>


<figure style="max-width:940px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/golden-gate-feature.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/golden-gate-feature.png' | relative_url }}" alt="A feature responding to references to the Golden Gate Bridge across linguistic contexts. Source: Anthropic (2024)." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">A feature responding to references to the Golden Gate Bridge across linguistic contexts. Source: Anthropic (2024). <a href="https://www.anthropic.com/research/mapping-mind-language-model" target="_blank" rel="noopener noreferrer">Source: Anthropic (2024), Mapping the Mind of a Large Language Model</a>. Click to enlarge.</figcaption>
</figure>


This suggests that a meaningful concept need not coincide with an individual neuron. But it raises another question: **Is the recovered feature basis uniquely privileged, or simply a useful decomposition produced by our measurement method?** A readable dictionary can help us investigate a network without automatically becoming the network's true set of atomic computational units.

### Circuits: How do components work together?

Even if we know what individual components represent, we may not know how their interactions produce an answer.

A classic example is [*Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small*](https://arxiv.org/abs/2211.00593) (Wang et al., 2023). Consider a prompt such as:

> When Mary and John went to the store, John gave a drink to ...

The expected next name is **Mary**, not John. A high-level description might say that the model tracks the names, recognizes that John has already appeared in a relevant role, and promotes Mary as the answer. But which internal components implement these operations?

The researchers used interventions, including **activation patching** and **path patching**, to investigate contributions from particular attention heads and the interactions among them. In activation patching, we replace an internal activation from one run with the corresponding activation from another run and observe how the output changes. By targeting paths rather than just components, we can ask whether information travels through a proposed connection.

The resulting account involved multiple classes of attention heads, rather than a single “Mary neuron.” It is also an instructive reminder of the difficulty of circuit analysis: even a relatively localized behavior can involve a surprisingly elaborate internal organization.

<figure style="max-width:930px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/ioi-circuit.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/ioi-circuit.png' | relative_url }}" alt="An IOI circuit hypothesis connecting several attention-head roles; these components are part of a causal account of a specific next-name prediction task." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">An IOI circuit hypothesis connecting several attention-head roles; these components are part of a causal account of a specific next-name prediction task. <a href="https://arxiv.org/abs/2211.00593" target="_blank" rel="noopener noreferrer">Source: Wang et al. (2023), Interpretability in the Wild</a>. Click to enlarge.</figcaption>
</figure>


The goal is not simply to draw a more detailed wiring diagram. It is to reconstruct **a computation performed by interacting parts**, and to determine how much of the model's behavior the proposed circuit really explains.


### Population geometry: How is knowledge organized?

Another approach is to treat neural activity as a population-level representation. Each input corresponds to a point in a high-dimensional activation space. Instead of asking which individual neuron detects a concept, we ask how relations among many inputs are organized geometrically.

This is especially relevant for concepts. A concept must be sufficiently stable to remain recognizable, yet flexible enough to change with context. The word *apple* might evoke a fruit, a company, or something very different depending on the sentence in which it appears.

In our work, [*Language Models Represent and Transform Concepts with Shared Geometry*](https://arxiv.org/abs/2607.04525) (Hu, Niu, & Varma, 2026), we studied contextual changes in conceptual representations as a **field of displacements** over a vocabulary. We found that contexts do not simply shift every concept by one identical vector: the directions and magnitudes of change differ across concepts, with semantically structured variation. Some of this organization is also reproducible across model families.

<figure style="max-width:940px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/concept-population-geometry.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/concept-population-geometry.png' | relative_url }}" alt="Population-level concept geometry from the lecture. Points are concept representations; context changes their spatial relationships." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">Population-level concept geometry from the lecture. Points are concept representations; context changes their spatial relationships. <a href="https://arxiv.org/abs/2607.04525" target="_blank" rel="noopener noreferrer">Source: Hu, Niu &amp; Varma (2026), Language Models Represent and Transform Concepts with Shared Geometry</a>. Click to enlarge.</figcaption>
</figure>


This is a different explanatory lens from individual-neuron analysis. It allows us to ask whether flexible conceptual behavior is related to systematic transformations of an underlying representational structure.

But geometry creates a familiar inferential problem. **A geometric regularity is not, by itself, evidence that the network uses that geometry to produce its behavior.** To establish a functional mechanism, we need to connect the structure to downstream computation, generalization, or intervention.

### Dynamics: Computation can be a trajectory

Finally, some computations are best described not by a static representation but by how activity **evolves over time**.

In [*Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks*](https://direct.mit.edu/neco/article/25/3/626/7854/Opening-the-Black-Box-Low-Dimensional-Dynamics-in) (Sussillo & Barak, 2013), the authors studied recurrent networks trained on tasks including a three-bit flip-flop memory problem.

A three-bit memory has eight possible states. After training, the recurrent network's high-dimensional dynamics could be interpreted using a lower-dimensional structure with stable memory states, or **attractors**, and unstable structures associated with transitions between them. The authors examined fixed points and the local Jacobian of the recurrent dynamics to understand where trajectories remain stable and where small perturbations can grow.

<figure style="max-width:960px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/rnn-dynamics.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/rnn-dynamics.png' | relative_url }}" alt="State-space analysis of an RNN trained on a three-bit memory task, with stable states and transition-related dynamics." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">State-space analysis of an RNN trained on a three-bit memory task, with stable states and transition-related dynamics. <a href="https://direct.mit.edu/neco/article/25/3/626/7854/Opening-the-Black-Box-Low-Dimensional-Dynamics-in" target="_blank" rel="noopener noreferrer">Source: Sussillo &amp; Barak (2013), Opening the Black Box</a>. Click to enlarge.</figcaption>
</figure>


This provides a mechanistic account of memory that does not depend on finding one neuron for each bit. Memory corresponds to a **stable state of a dynamical system**, and changing memory corresponds to moving between those states.

The lesson from these five approaches is not that one is better than the others. They illuminate different aspects of a learned computation. The first scientific decision is to ask **what exactly needs to be explained, and which level of analysis is appropriate**.

## 3. From seeing structure to explaining a mechanism

We have methods for detecting selective units, extracting features, reconstructing circuits, measuring geometry, and analyzing dynamics. But a microscope does not supply the logic of scientific inference.

A finding inside a model might be interesting without settling the mechanism behind a behavior. To see the difference, let us follow three discoveries that ask increasingly demanding questions.

### Case study I: Does a language model construct a world model?

An elegant starting point is [*Emergent World Representations*](https://arxiv.org/abs/2210.13382) (Li et al., ICLR 2023). The authors trained a GPT-like sequence model to predict moves in the game Othello. The model receives a sequence of moves; it is not directly given an image or symbolic encoding of the current board state.

What might the network learn? One possibility is a collection of statistical regularities about legal move sequences. Another is an **internal representation of the evolving board**.

<figure style="max-width:920px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/othello-board-sequence.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/othello-board-sequence.png' | relative_url }}" alt="A sequence of Othello moves changes the board, even though the model receives the moves rather than board images." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">A sequence of Othello moves changes the board, even though the model receives the moves rather than board images. <a href="https://arxiv.org/abs/2210.13382" target="_blank" rel="noopener noreferrer">Source: Li et al. (2023), Emergent World Representations</a>. Click to enlarge.</figcaption>
</figure>


The first test was to probe the hidden activations. A probe takes the model's internal state and tries to reconstruct the state of each square on the board. Successful decoding reveals that board information is accessible in the hidden representation.

But consider the inferential gap. Imagine a computer program that stores a detailed log of its input history but never uses that log when choosing an output. We might be able to reconstruct the current board from the log, yet that would not show that the program uses a board representation in its decision-making.

**Decodable does not necessarily mean used.**

The next step, therefore, was an intervention. The researchers changed internal activations in a way intended to alter the represented status of a board square, while leaving the preceding move sequence unchanged. The model's subsequent move predictions shifted in ways consistent with the modified board state.

<figure style="max-width:750px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/othello-intervention.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/othello-intervention.png' | relative_url }}" alt="Othello intervention: changing the representation of a board square changes subsequent move predictions." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">Othello intervention: changing the representation of a board square changes subsequent move predictions. <a href="https://arxiv.org/abs/2210.13382" target="_blank" rel="noopener noreferrer">Source: Li et al. (2023), Emergent World Representations</a>. Click to enlarge.</figcaption>
</figure>


That is a much stronger result than successful probing alone. It supports the claim that board-state information is **functionally implicated** in the model's computation. It does not, however, provide a complete account of how every move is generated.

This is the central distinction between *finding a representation* and *understanding its computational use*.

For an accessible walkthrough, see Kenneth Li's [Othello-GPT explanation in *The Gradient*](https://thegradient.pub/othello/). The [official Othello World repository](https://github.com/likenneth/othello_world) includes training, probing, and intervention notebooks.


### Case study II: Can a next-token predictor plan ahead?

Autoregressive language models generate text one token at a time. It is tempting to interpret that process as purely reactive: produce one word, inspect the partial sentence, and then produce the next.

But one-token-at-a-time *output* does not imply one-token-at-a-time *planning*.

Consider a model asked to write a rhyming poem. Two explanations are possible:

- **Improvisation:** The model generates most of the line and only then finds a suitable rhyme.
- **Planning:** The model selects a likely ending in advance, then constructs earlier text to reach that ending.

Both strategies might produce equally fluent poetry. How could we tell them apart?

<figure style="max-width:930px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/poetry-planning.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/poetry-planning.png' | relative_url }}" alt="Attribution-graph evidence for candidate rhyme words being represented before they are emitted." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">Attribution-graph evidence for candidate rhyme words being represented before they are emitted. <a href="https://www.transformer-circuits.pub/2025/attribution-graphs/biology.html" target="_blank" rel="noopener noreferrer">Source: Lindsey et al. (2025), On the Biology of a Large Language Model</a>. Click to enlarge.</figcaption>
</figure>


In [*On the Biology of a Large Language Model*](https://www.transformer-circuits.pub/2025/attribution-graphs/biology.html) (Lindsey et al., 2025), researchers used attribution-graph methods to investigate Claude 3.5 Haiku. In poetry examples, they identified internal features associated with possible future rhyming words *before* the model generated the corresponding line.

An early activation is suggestive, but not decisive. The stronger evidence came from interventions: manipulating candidate rhyme-related features at particular points could shift the later ending and influence the intervening words. This is what a planning hypothesis predicts. A feature that contributes to a future target should affect not just the final word but the construction of the line leading up to it.

<figure style="max-width:1000px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/poetry-intervention.jpg' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/poetry-intervention.jpg' | relative_url }}" alt="Suppressing or injecting a rhyme-related feature at a planning location changes the ensuing completion." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">Suppressing or injecting a rhyme-related feature at a planning location changes the ensuing completion. <a href="https://www.transformer-circuits.pub/2025/attribution-graphs/biology.html" target="_blank" rel="noopener noreferrer">Source: Lindsey et al. (2025), On the Biology of a Large Language Model</a>. Click to enlarge.</figcaption>
</figure>


Notice the difference between two experimental questions:

- *Does removing this feature change the answer?*
- *If this feature specifies a planned ending, does changing it at the planning stage reorganize the earlier sequence toward a different ending?*

The second question tests a **computational role**, rather than merely causal relevance.

The evidence does not establish that all language generation works this way. It supports an account of particular poetry-generation behavior in a particular studied model. The researchers also emphasize that their attribution graphs are incomplete reconstructions, not literal and exhaustive renderings of the underlying computation.

That caution matters: an intervention effect is not equivalent to a full algorithmic explanation. The value of an intervention depends on the **specific prediction** it distinguishes.


### Case study III: How does a language model keep track of beliefs?

Now consider a simple story:

> Bob pours beer into a bottle. Carla pours coffee into a cup. What is in Carla's cup?

A model can answer “coffee.” But to do so reliably across longer stories, changing objects, and multiple characters, it must preserve relationships between **who**, **what object**, and **what state**. When characters see different events, the problem becomes harder: the model must also distinguish reality from what each character believes.

The key question is not merely whether a model can answer these questions. It is **how it organizes and retrieves the relevant information**.

[Prakash et al., *Language Models Use Lookbacks to Track Beliefs*](https://belief.baulab.info/) (ICLR 2026) studied this using controlled stories, causal analysis, and abstraction-based interventions. They identified a recurring **lookback** mechanism: information linking characters, objects, and states is encoded through reference-like signals, and subsequent operations retrieve the relevant state and answer when queried. Stories involving who could observe whom required additional processing related to visibility.

<figure style="max-width:950px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/belief-tracking.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/belief-tracking.png' | relative_url }}" alt="Counterfactual interchange intervention on a simplified belief-state retrieval computation." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">Counterfactual interchange intervention on a simplified belief-state retrieval computation. <a href="https://belief.baulab.info/" target="_blank" rel="noopener noreferrer">Source: Prakash et al. (2026), Language Models Use Lookbacks to Track Beliefs</a>. Click to enlarge.</figcaption>
</figure>


The researchers' account matters because the explanation is **relational**. It is not enough to find a neuron associated with “Carla,” another with “cup,” and another with “coffee.” We need to know how the model binds the right pieces together and how one operation supplies information to the next.

To illustrate the intuition, imagine a deliberately simplified algorithm:

```python
def answer(story, question):
    state_pointer = find_relevant_state(story, question)
    return retrieve_value(story, state_pointer)
```

The example is conceptual pseudocode, not a literal implementation recovered in the paper. Its purpose is to separate two computational roles: **selecting the right state** and **retrieving the value stored there**. Interventions that selectively disrupt these roles can test whether a proposed organization explains the model's outputs.

This moves us beyond a collection of causally important components toward a model of **how those components work together**. It also illustrates a critical boundary: identifying a belief-tracking algorithm inside an LLM is not the same as demonstrating that the system possesses human-like Theory of Mind, or that humans use the same algorithm.

The [project website](https://belief.baulab.info/) provides an accessible overview, and the [research code](https://github.com/Nix07/belief_tracking) contains the authors' experiments.

## 4. A logic of mechanistic discovery

What connects the examples above? They use different models and methods, but they all face the same problem: **How do we turn evidence about internal activity into a justified claim about computation?**

This question motivated a perspective I developed with collaborators, *Mechanistic Interpretability Needs a Logic of Inference* (Hu, Jiang, Xie, Ratan Murty, & Varma, 2026). We proposed five recurring principles, underpinned by a general principle about measurement.

| Principle | The question to ask | What the next experiment should accomplish |
|---|---|---|
| **P1 — Explanatory object** | *What should we explain?* | Test whether neurons, features, circuits, geometry, or dynamics capture the relevant phenomenon. |
| **P2 — Represented vs. used** | *Does the model use the information we found?* | Link upstream information to downstream computation or behavior. |
| **P3 — Causal role** | *What operation does this component perform?* | Distinguish competing computational explanations through targeted interventions. |
| **P4 — Mechanistic organization** | *How do the operations depend on one another?* | Reconstruct relationships among components and pathways. |
| **P5 — Invariance and recursion** | *Which part of the explanation survives changing conditions?* | Test across inputs, tasks, training stages, or models—and revise the explanatory level if needed. |

**P0 — Measurement:** At every step, ask how the result depends on the probe, metric, intervention, corruption distribution, or other operational choices. Measurement is part of the evidence, not a transparent window onto a preexisting answer.

These are not five boxes to tick in order. They are a **recursive research logic**. A circuit analysis may reveal that the natural explanatory object is actually a lower-dimensional algorithm. A geometric result may suggest a representation, but intervention may reveal that the relevant information is not used. A mechanism that appears stable on one class of inputs may fail on another.

The point is not to demand a complete reverse-engineering of every neural network. It is to make claims proportionate to evidence—and to design the *next experiment* around what remains uncertain.

## 5. What do ANN mechanisms tell us about cognition?

So far, most examples establish facts about **artificial systems**. These can be fascinating and scientifically valuable in their own right. But the title of this lecture asks a more demanding question: *How can interpreting ANNs contribute to cognitive science?*

I find it useful to distinguish three contributions.

### 5.1 Computational existence proofs

A neural network can demonstrate that a cognitive ability is computationally possible **without a mechanism we had assumed was necessary**.

For example, a network that learns to maintain an internal game state from sequences alone challenges the intuition that a world representation must be explicitly supplied in advance. A next-token generator that sometimes plans future wording suggests that sequential output need not correspond to purely myopic internal computation.

These results enlarge the set of plausible computational explanations. They do **not** establish which explanation humans use. An existence proof shows that a mechanism *can* support a capability, not that it *must* or *does* support it in the brain.

### 5.2 Constraints on cognitive theories

A stronger contribution may come from understanding **which representations and algorithms naturally emerge under particular learning conditions**.

Suppose a cognitive theory proposes that flexible conceptual behavior requires an explicit controller that selects a new representation for every context. But suppose networks repeatedly acquire structured, context-dependent transformations over a shared conceptual space, and we can demonstrate that this organization is sufficient for flexible generalization. That finding would not prove that human concepts work the same way. It would, however, challenge a claim about the *computational requirements* of conceptual flexibility.

To make this argument convincing, we need more than a resemblance between representational plots. We need to identify the conditions that generate the organization, show how it supports a function, and test when the proposed explanation ceases to work.

In this sense, the most interesting question may not be simply *what mechanism does this model use?*, but rather: **Why do learning systems discover this sort of mechanism, and what capabilities does it make possible?**

### 5.3 Brain–model alignment is not mechanistic understanding

There is also a more direct route through **NeuroAI**. We can compare neural-network representations with brain activity, use models to predict neural responses, or train models under constraints derived from neural data. Such work can provide useful *in silico* models of aspects of brain computation.

Yet good brain–model alignment does not settle the computational mechanism. Two models might predict similar neural measurements while implementing different internal algorithms. A representation that aligns well with a brain region may tell us *what information is shared* without explaining *how that information is transformed or used*.

Mechanistic analysis can help formulate more precise hypotheses about these predictive models. But explaining an aligned ANN is still not, by itself, proof that the same mechanism is implemented in the brain. That stronger inference requires evidence that discriminates between possible biological mechanisms.

This leaves an important complementary pair of questions:

- **Correspondence:** Does the artificial system capture relevant properties of neural or behavioral data?
- **Mechanism:** What computation explains those properties, and does that account transfer across systems?

Neither question replaces the other.

## 6. A particularly instructive example: The mathematics of semantic development

One of my favorite papers that connects neural-network analysis with cognitive theory is [*A Mathematical Theory of Semantic Development in Deep Neural Networks*](https://web.stanford.edu/~jlmcc/papers/SaxeMcCGanguli19MathTheorySemCog.pdf) (Saxe, McClelland, & Ganguli, PNAS 2019).

The authors begin with familiar phenomena of semantic development: broad conceptual distinctions often emerge before fine ones; children can temporarily make systematic semantic errors; category typicality affects judgments; and inductive generalization tends to become more specific with development.

Could a common computational principle account for several of these findings?

### From experience to statistical structure

Imagine a learning environment containing objects and their properties. A canary can fly; a salmon can swim; an oak has bark. A network receives an item as input and learns to predict its properties.

<figure style="max-width:890px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/saxe-learning-setup.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/saxe-learning-setup.png' | relative_url }}" alt="Learning item–property associations: individual experiences are used to train a network to predict properties." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">Learning item–property associations: individual experiences are used to train a network to predict properties. <a href="https://doi.org/10.1073/pnas.1820226116" target="_blank" rel="noopener noreferrer">Source: Saxe, McClelland &amp; Ganguli (2019), A Mathematical Theory of Semantic Development</a>. Click to enlarge.</figcaption>
</figure>


The structure of experience can be summarized by an input–output cross-correlation matrix, which the authors decompose using singular value decomposition (SVD):

$$
\Sigma_{yx}=USV^\top.
$$

Different singular modes correspond to statistical distinctions within the domain. In a hierarchically structured environment, a strong mode might distinguish animals from plants, while weaker modes distinguish birds from fish or trees from flowers. The singular vectors characterize **which items and properties participate in each distinction**; the singular values characterize their statistical strengths.

<figure style="max-width:1000px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/saxe-svd-modes.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/saxe-svd-modes.png' | relative_url }}" alt="Input–output correlations are decomposed into item modes, feature modes, and time-dependent learned strengths." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">Input–output correlations are decomposed into item modes, feature modes, and time-dependent learned strengths. <a href="https://doi.org/10.1073/pnas.1820226116" target="_blank" rel="noopener noreferrer">Source: Saxe, McClelland &amp; Ganguli (2019), A Mathematical Theory of Semantic Development</a>. Click to enlarge.</figcaption>
</figure>


At this point, we have a description of the **data**. We have not yet explained learning or cognition.

### Why learning dynamics matter

The crucial step is to analyze a *deep linear network* trained on those associations. A deep linear network has no more input–output expressive power than a single linear layer. Yet its parameterized learning dynamics are different.

Under the assumptions analyzed in the paper, the learned map can be written as

$$
W_2(t)W_1(t)=UA(t)V^\top,
$$

where the diagonal entries of $A(t)$ describe how much each statistical mode has been learned at time $t$. In the relevant regime, stronger modes are acquired earlier and their learning trajectories can show relatively sharp, stage-like transitions.

The distinction is important: **SVD alone reveals the available structure; the learning dynamics predicts when that structure becomes part of the model's computation.** Under idealized conditions, the authors can derive these trajectories analytically rather than relying only on numerical training.

<figure style="max-width:720px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/saxe-semantic-differentiation.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/saxe-semantic-differentiation.png' | relative_url }}" alt="Model representations progressively differentiate semantic categories over learning time." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">Model representations progressively differentiate semantic categories over learning time. <a href="https://doi.org/10.1073/pnas.1820226116" target="_blank" rel="noopener noreferrer">Source: Saxe, McClelland &amp; Ganguli (2019), A Mathematical Theory of Semantic Development</a>. Click to enlarge.</figcaption>
</figure>


That connection between data statistics and learning dynamics generates several cognitive phenomena:

- **Hierarchical differentiation.** Broad distinctions emerge before finer ones because the corresponding modes are learned at different rates.
- **Semantic illusions.** An early-learned mode may temporarily support a mistaken property inference before later modes correct it. The network can make a systematic error even though the training environment does not contain that false fact.
- **Typicality and category coherence.** The geometry of learned semantic modes provides mathematical characterizations of representative category members and of which category structures are easiest to learn.
- **Changing inductive generalization.** Early representations are less differentiated, so a newly learned property can generalize broadly. Later representations distinguish items more finely, leading to narrower generalization.

<figure style="max-width:900px;margin:1.8rem auto 2.1rem;">
  <a href="{{ '/assets/img/posts/mechanistic-interpretability/saxe-typicality.png' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size figure">
    <img src="{{ '/assets/img/posts/mechanistic-interpretability/saxe-typicality.png' | relative_url }}" alt="A semantic axis positions items according to their typicality within a category distinction." loading="lazy" decoding="async" style="display:block;width:100%;height:auto;border:1px solid #e6e8eb;border-radius:5px;" />
  </a>
  <figcaption style="font-size:.88em;line-height:1.5;color:#6b7280;margin-top:.55rem;">A semantic axis positions items according to their typicality within a category distinction. <a href="https://doi.org/10.1073/pnas.1820226116" target="_blank" rel="noopener noreferrer">Source: Saxe, McClelland &amp; Ganguli (2019), A Mathematical Theory of Semantic Development</a>. Click to enlarge.</figcaption>
</figure>


This is what makes the paper instructive. It does not simply report that a network contains a semantic hierarchy. It shows how **environmental statistics, a particular learning mechanism, the geometry of internal representations, and observable behavior** can be connected within one theoretical account.


### A question worth asking: Why train a network if SVD explains so much?

There is an apparent objection. If many semantic distinctions and measures of typicality already exist in the SVD of the item–property matrix, why bother with the network?

For static *structure*, this objection has force: much of what the network ultimately represents is inherited from the training data's statistics.

The additional contribution of the learning analysis is **time and mechanism**. The static decomposition does not, by itself, predict a sequence of developmental transitions or the temporary rise and correction of erroneous beliefs. The learning dynamics explains how modes are acquired at different rates and how their changing contributions alter behavior.

This is also why the theory has informative limits. It relies on simplified networks and specific assumptions about learning and data. It offers an elegant *sufficient computational explanation* of several phenomena, not a demonstration that human brains literally perform this SVD or implement the same learning equations.

Moreover, this is a mathematical cognitive-modeling study, not a case in which researchers first reverse-engineered an unrelated pretrained AI system and then discovered a new fact about human cognition. I include it because it shows a standard of **explanatory integration** that I would like mechanistic studies of modern ANNs to approach: one coherent mechanism connecting multiple phenomena, rather than an ever-expanding catalog of interesting internal patterns.

## 7. Where does this leave mechanistic interpretability?

The field has developed increasingly sophisticated ways to observe and manipulate neural networks. In some settings, we can identify meaningful features, map partial computational circuits, and intervene on internal variables with impressive specificity. Those achievements are real.

But a science of mechanisms should aspire to more than describing the contents of trained networks. Three distinctions seem particularly important to me:

**Description versus explanation.** A geometric regularity or interpretable feature is a finding. Explaining why it arises, which computation it supports, and what follows from it is a further step.

**Local mechanisms versus general principles.** A component-level explanation for one model and prompt may be correct yet highly implementation-specific. More general science asks what remains stable across tasks, learning conditions, architectures, and populations—and why.

**Artificial cognition versus human cognition.** A mechanism found in an ANN can establish computational possibility, suggest an alternative cognitive theory, or motivate a new experiment. It cannot automatically become a claim about the human brain.

My current view is that MI's deepest scientific opportunity lies in moving toward **the principles of learned computation**: under what circumstances do learning systems develop particular representations and algorithms, what functions do those structures support, and which of these principles can help explain cognition beyond a single model?

That is a more demanding agenda than identifying a neuron, feature, or circuit. It is also, to me, a more compelling one.

We may eventually find that some aspects of intelligence are realized through stable, general computational principles, while others are highly contingent on architecture, data, and training. Either outcome would be scientifically interesting—provided we can formulate predictions that distinguish them.

The original question therefore remains open:

> **Can we use the mechanisms discovered inside artificial neural networks to build explanations of cognition that are more precise, predictive, and general than the theories we have today?**

I do not think we have a complete answer yet. But studying the internal computations of learned systems gives us an unusually direct way to investigate it.

---

## Papers, visual explanations, and code

The following resources are especially useful for readers who want to explore the examples or reproduce parts of the analysis. Figures in this post were adapted from the original lecture slides; credit for each published research figure appears in its caption.

| Topic | Read | Explore / code |
|---|---|---|
| Neuron selectivity | [Network Dissection (Bau et al., 2017)](https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html) | [NetDissect GitHub](https://github.com/CSAILVision/NetDissect) |
| Feature visualization | [Feature Visualization (Distill, 2017)](https://distill.pub/2017/feature-visualization/) | Interactive figures in the article |
| Distributed features | [Mapping the Mind (Anthropic, 2024)](https://www.anthropic.com/research/mapping-mind-language-model) | [SAELens GitHub](https://github.com/decoderesearch/SAELens) |
| Circuit analysis | [Indirect Object Identification (Wang et al., 2023)](https://arxiv.org/abs/2211.00593) | [TransformerLens tutorials](https://transformerlensorg.github.io/TransformerLens/content/tutorials.html) |
| Population geometry | [Hu, Niu, & Varma (2026)](https://arxiv.org/abs/2607.04525) | Paper and supplementary analyses |
| Dynamical systems | [Sussillo & Barak (2013)](https://direct.mit.edu/neco/article/25/3/626/7854/Opening-the-Black-Box-Low-Dimensional-Dynamics-in) | Fixed-point and state-space analyses in the paper |
| World representations | [Othello-GPT paper](https://arxiv.org/abs/2210.13382) · [Illustrated explanation](https://thegradient.pub/othello/) | [Othello World GitHub](https://github.com/likenneth/othello_world) |
| Planning in language models | [On the Biology of a Large Language Model (2025)](https://www.transformer-circuits.pub/2025/attribution-graphs/biology.html) | Interactive attribution graphs in the article |
| Belief tracking | [Lookbacks project website](https://belief.baulab.info/) · [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d1941f73833eb107939f62d323d6cc0-Abstract-Conference.html) | [Belief tracking GitHub](https://github.com/Nix07/belief_tracking) |
| Semantic development | [Saxe, McClelland, & Ganguli (2019)](https://web.stanford.edu/~jlmcc/papers/SaxeMcCGanguli19MathTheorySemCog.pdf) | Derivations and supplementary material in the paper |

For hands-on experiments with activation caching, patching, and interventions in language models, [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) is a useful starting point. For sparse-feature analysis, see [SAELens](https://github.com/decoderesearch/SAELens). These are research tools, not substitutes for deciding what scientific question an experiment is meant to answer.
