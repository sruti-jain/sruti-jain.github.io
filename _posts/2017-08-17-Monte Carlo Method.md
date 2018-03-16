---
layout: post
title: First post!
image: /img/hello_world.jpeg
---

**Monte Carlo Method** is a computational mathematical technique that allows us to interpret all probable outcomes of our decisions allowing better decision making strategies under uncertainty. The essential idea behind this method is using randomness to solve problems that might be deterministic in principle.

During a Monte Carlo simulation, values are sampled at random from the input probability distributions.  Each set of samples is called an iteration, and the resulting outcome from that sample is recorded.  Monte Carlo simulation does these hundreds or thousands of times, and the result is a probability distribution of possible outcomes.  In this way, Monte Carlo simulation provides a much more comprehensive view of what may happen.  It tells you not only what could happen, but how likely it is to happen. The larger the sample size, the more accurate the estimation of the output distributions.

**Monte Carlo method in Data analysis**:

**Sampling**: Here the objective is to gather information about a random object by observing many realizations of it. An example is simulation modeling, where a random process mimics the behavior of some real-life system, such as a production line or telecommunications network. Another example is found in Bayesian statistics, where Markov chain Monte Carlo (MCMC) is often used to sample from a posterior distribution.

**Estimation:** In this case, the emphasis is on estimating certain numerical quantities related to a simulation model. An example in the natural setting of Monte Carlo techniques is the estimation of the expected throughput in a production line. An example in the artificial context is the evaluation of multi-dimensional integrals via Monte Carlo techniques by writing the integral as the expectation of a random variable.

**Optimization**. The Monte Carlo Method is a powerful tool for the optimization of complicated objective functions. In many applications, these functions are deterministic and randomness is introduced artificially in order to more efficiently search the domain of the objective function. Monte Carlo techniques are also used to optimize noisy functions, where the function itself is random — for example, the result of a Monte Carlo simulation.

**Advantages of Monte Carlo Simulation**:  Monte Carlo simulation provides a number of advantages over deterministic, or “single-point estimate” analysis:

**Probabilistic Results:** Results show not only what could happen, but how likely each outcome is.

**Graphical Results:** Because of the data Monte Carlo simulation generates, it’s easy to create graphs of different outcomes and their chances of occurrence.  This is important for communicating findings to other stakeholders.

**Correlation of Inputs: **In Monte Carlo simulation, it’s possible to model interdependent relationships between input variables.  It’s important for accuracy to represent how, in reality, when some factors go up, others go up or down accordingly.

**Sensitivity Analysis:** With just a few cases, deterministic analysis makes it difficult to see which variables impact the outcome the most.  In Monte Carlo simulation, it’s easy to see which inputs had the biggest effect on bottom-line results.

**Scenario Analysis:** In deterministic models, it’s very difficult to model different combinations of values for different inputs to see the effects of truly different scenarios.  Using Monte Carlo simulation, analysts can see exactly which inputs had which values together when certain outcomes occurred.  This is invaluable for pursuing further analysis.

**Two real time application of Monte Carlo method: **are implemented and the github links to those are as given below:

1. Determining the value of pi: *Github Link*
2. Risk Analysis for stock market: *Github Link*

**References and further reading**:

1. Gábor Belvárdi, “[Monte Carlo simulation based performance analysis of Supply Chains](http://airccse.org/journal/mvsc/papers/3212ijmvsc01.pdf)“,  (IJMVSC) Vol. 3, No. 2
2. Dirk P. Kroese, “[Why the Monte Carlo Method is so important today](https://people.smp.uq.edu.au/DirkKroese/ps/whyMCM_fin.pdf)“, The University of Queensland
3. Mike Giles, “[Research on Monte Carlo Methods](https://people.maths.ox.ac.uk/gilesm/talks/nomura.pdf)“, Mathematical and Computational Finance Group, Nomura, Tokyo
4. “[Palisade Monte Carlo simulation](http://www.palisade.com/risk/monte_carlo_simulation.asp)“, Monte Carlo Simulation
5. Pamela Paxton, Patrick J. Curran, Kenneth A. Bollen, Jim Kirby, Feinian Chen, “[Monte Carlo Experiments: Design and Implementation](http://www.unc.edu/~curran/pdfs/Paxton,Curran,Bollen,Kirby%26Chen(2001).pdf)“,  288-312 Paxton Et Al.
6. Book: R. Y. Rubinstein, “Simulation and the Monte Carlo Method”, John Wiley and Sons, New York (1981).