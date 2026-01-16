#!/usr/bin/env wolframscript
(* ::Package:: *)

nBandits = 10;
timeSteps = 10000;
runs = 2000;
\[Epsilon] = 0.1;
meanDrift = 0.01;
\[Alpha] = 0.1;


(* ::Section:: *)
(*Sample average*)


Monitor[sampleAverageRewards = ConstantArray[0, timeSteps];
sampleOptimalRatio = ConstantArray[0, timeSteps];
Do[bandits = ConstantArray[0, nBandits];
estimatedMeans = ConstantArray[0, nBandits];
actionCounts = ConstantArray[0, nBandits];
optimalActionCount = 0;
runningTotal = 0;
{averageReward, optimalRatio} = Reap[Do[greedyAction = FirstPosition[estimatedMeans, Max[estimatedMeans]][[1]];
action = If[RandomReal[] < \[Epsilon], RandomInteger[{1, nBandits}], greedyAction];
actionCounts[[action]] += 1;
reward = bandits[[action]] + RandomVariate[NormalDistribution[]];
runningTotal += reward;
estimatedMeans[[action]] = estimatedMeans[[action]] + (reward - estimatedMeans[[action]]) / actionCounts[[action]];
Sow[runningTotal / t, "average"];
optimalBandit = FirstPosition[bandits, Max[bandits]][[1]];
If[action == optimalBandit, optimalActionCount += 1];
Sow[optimalActionCount / t, "optimal"];
bandits += RandomVariate[NormalDistribution[0, meanDrift], nBandits], {t, timeSteps}], {"average", "optimal"}][[2, All, 1]];
sampleAverageRewards += averageReward;
sampleOptimalRatio += optimalRatio, {r, runs}];
sampleAverageRewards /= runs;
sampleOptimalRatio /= runs, {r, t}]


(* ::Section:: *)
(*Exponential recency-weighted average*)


Monitor[exponentialAverageRewards = ConstantArray[0, timeSteps];
exponentialOptimalRatio = ConstantArray[0, timeSteps];
Do[bandits = ConstantArray[0, nBandits];
estimatedMeans = ConstantArray[0, nBandits];
actionCounts = ConstantArray[0, nBandits];
optimalActionCount = 0;
runningTotal = 0;
{averageReward, optimalRatio} = Reap[Do[greedyAction = FirstPosition[estimatedMeans, Max[estimatedMeans]][[1]];
action = If[RandomReal[] < \[Epsilon], RandomInteger[{1, nBandits}], greedyAction];
actionCounts[[action]] += 1;
reward = bandits[[action]] + RandomVariate[NormalDistribution[]];
runningTotal += reward;
estimatedMeans[[action]] = estimatedMeans[[action]] + \[Alpha] (reward - estimatedMeans[[action]]);
Sow[runningTotal / t, "average"];
optimalBandit = FirstPosition[bandits, Max[bandits]][[1]];
If[action == optimalBandit, optimalActionCount += 1];
Sow[optimalActionCount / t, "optimal"];
bandits += RandomVariate[NormalDistribution[0, meanDrift], nBandits], {t, timeSteps}], {"average", "optimal"}][[2, All, 1]];
exponentialAverageRewards += averageReward;
exponentialOptimalRatio += optimalRatio, {r, runs}];
exponentialAverageRewards /= runs;
exponentialOptimalRatio /= runs, {r, t}]


(* ::Section:: *)
(*Performance comparison*)


ListPlot[{Table[{t, sampleAverageRewards[[t]]}, {t, timeSteps}],Table[{t, exponentialAverageRewards[[t]]}, {t, timeSteps}]}]
ListPlot[{Table[{t, sampleOptimalRatio[[t]]}, {t, 2, timeSteps}],Table[{t, exponentialOptimalRatio[[t]]}, {t, 2, timeSteps}]}]




