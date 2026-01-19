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
sampleOptimalFrequency = ConstantArray[0, timeSteps];
Do[bandits = ConstantArray[0, nBandits];
estimatedMeans = ConstantArray[0, nBandits];
actionCounts = ConstantArray[0, nBandits];
{rewards, optimalAction} = Reap[Do[greedyAction = FirstPosition[estimatedMeans, Max[estimatedMeans]][[1]];
action = If[RandomReal[] < \[Epsilon], RandomInteger[{1, nBandits}], greedyAction];
actionCounts[[action]] += 1;
reward = bandits[[action]] + RandomVariate[NormalDistribution[]];
estimatedMeans[[action]] = estimatedMeans[[action]] + (reward - estimatedMeans[[action]]) / actionCounts[[action]];
Sow[reward, "reward"];
optimalBandit = FirstPosition[bandits, Max[bandits]][[1]];
Sow[If[action == optimalBandit, 1, 0], "optimal"];
bandits += RandomVariate[NormalDistribution[0, meanDrift], nBandits], {t, timeSteps}], {"reward", "optimal"}][[2, All, 1]];
sampleAverageRewards += rewards;
sampleOptimalFrequency += optimalAction, {r, runs}];
sampleAverageRewards /= runs;
sampleOptimalFrequency /= runs, {r}];


(* ::Section:: *)
(*Exponential recency-weighted average*)


Monitor[exponentialAverageRewards = ConstantArray[0, timeSteps];
exponentialOptimalFrequency = ConstantArray[0, timeSteps];
Do[bandits = ConstantArray[0, nBandits];
estimatedMeans = ConstantArray[0, nBandits];
actionCounts = ConstantArray[0, nBandits];
{rewards, optimalAction} = Reap[Do[greedyAction = FirstPosition[estimatedMeans, Max[estimatedMeans]][[1]];
action = If[RandomReal[] < \[Epsilon], RandomInteger[{1, nBandits}], greedyAction];
actionCounts[[action]] += 1;
reward = bandits[[action]] + RandomVariate[NormalDistribution[]];
estimatedMeans[[action]] = estimatedMeans[[action]] + \[Alpha] (reward - estimatedMeans[[action]]);
Sow[reward, "reward"];
optimalBandit = FirstPosition[bandits, Max[bandits]][[1]];
Sow[If[action == optimalBandit, 1, 0], "optimal"];
bandits += RandomVariate[NormalDistribution[0, meanDrift], nBandits], {t, timeSteps}], {"reward", "optimal"}][[2, All, 1]];
exponentialAverageRewards += rewards;
exponentialOptimalFrequency += optimalAction, {r, runs}];
exponentialAverageRewards /= runs;
exponentialOptimalFrequency /= runs, {r}];


(* ::Section:: *)
(*Performance comparison*)


ListPlot[{Table[{t, sampleAverageRewards[[t]]}, {t, timeSteps}],Table[{t, exponentialAverageRewards[[t]]}, {t, timeSteps}]}]
ListPlot[{Table[{t, sampleOptimalFrequency[[t]]}, {t, 2, timeSteps}],Table[{t, exponentialOptimalFrequency[[t]]}, {t, 2, timeSteps}]}]



