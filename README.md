# ICASSP 2022 Conference Paper Code

Title: Online Learning for Latent Yule-Simon Processes

Authors: Asher A. Hensley and Petar M. Djuric

Abstract: Yule-Simon processes are one of the most commonly occurring natural processes you’ve never heard of. These processes generate power laws using a preferential attachment mechanism which can describe a variety of data distributions such as word frequencies, scientific citations, journal publications, income, node connections in complex networks, biological genera, and bosons in quantum states. Much of the work in this area has focused on modeling the properties of observable quantities such as these. In this work we focus on learning the properties of {\it unobservable} Yule-Simon processes which control the dynamics of sequential sensor measurements. This is motivated by the fact Yule-Simon processes have a varying memory length which offer a more general framework for data modeling than hidden Markov models. In this paper we present an approximate online inference procedure based on multiple hypothesis pruning which is shown to reach 0.5dB of the posterior Cramer-Rao lower bound.

Keywords: Bayesian filtering, regime switching, real time machine learning, generative model, time series, state estimation, power law

Execute RUN_EXPERIMENT.m to reproduce the results of the paper.
