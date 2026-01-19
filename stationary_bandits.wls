#!/usr/bin/env wolframscript
(* ::Package:: *)

(* 10-armed testbed experiment: compares epsilon-greedy methods with different exploration rates *)

nBandits = 10;
timeSteps = 1000;
runs = 2000;
\[Epsilon]List = {0, 0.01, 0.1};
bias = 0;


Monitor[Do[ensembleAverageRewards[\[Epsilon]] = ConstantArray[0, timeSteps];
ensembleOptimalFrequency[\[Epsilon]] = ConstantArray[0, timeSteps];
Do[(* Each bandit has reward drawn from N(q*(a), 1) where q*(a) ~ N(0, 1) *)
bandits = RandomVariate[NormalDistribution[0, 1], nBandits];
estimatedMeans = ConstantArray[bias, nBandits];
actionCounts = ConstantArray[0, nBandits];
optimalBandit = FirstPosition[bandits, Max[bandits]][[1]];
{rewards, optimalAction} = Reap[Do[greedyAction = FirstPosition[estimatedMeans, Max[estimatedMeans]][[1]];
(* Epsilon-greedy action selection: explore with probability \[CurlyEpsilon], otherwise exploit *)
action = If[RandomReal[] < \[Epsilon], RandomInteger[{1, nBandits}], greedyAction];
actionCounts[[action]] += 1;
reward = bandits[[action]] + RandomVariate[NormalDistribution[]];
estimatedMeans[[action]] = estimatedMeans[[action]] + (reward - estimatedMeans[[action]]) / actionCounts[[action]]; (* Sample average *)
Sow[reward, "reward"]; (* Cumulative average reward *)
Sow[If[action == optimalBandit, 1, 0], "optimal"], (* Fraction of optimal actions *)
{t, timeSteps}], {"reward", "optimal"}][[2, All, 1]];
ensembleAverageRewards[\[Epsilon]] += rewards;
ensembleOptimalFrequency[\[Epsilon]] += optimalAction, {r, runs}];
(* Average performance across all runs for this epsilon value *)
ensembleAverageRewards[\[Epsilon]] /= runs;
ensembleOptimalFrequency[\[Epsilon]] /= runs, {\[Epsilon], \[Epsilon]List}], {\[Epsilon], r}]


ListPlot[Table[{t, ensembleAverageRewards[#][[t]]}, {t, timeSteps}] & /@ \[Epsilon]List]
ListPlot[Table[{t, ensembleOptimalFrequency[#][[t]]}, {t, timeSteps}] & /@ \[Epsilon]List]




