#!/usr/bin/env wolframscript
(* ::Package:: *)

nBandits = 10;
timeSteps = 10000;
runs = 100;
\[Epsilon] = 0.1;
meanDrift = 0.01;


Monitor[ensembleAverageRewards = ConstantArray[0, timeSteps];
ensembleOptimalRatio = ConstantArray[0, timeSteps];
Do[bandits = ConstantArray[0, nBandits];
estimatedMeans = ConstantArray[0, nBandits];
totalRewards = ConstantArray[0, nBandits];
actionCounts = ConstantArray[0, nBandits];
optimalActionCount = 0;
runningTotal = 0;
{averageReward, optimalRatio} = Reap[Do[greedyAction = FirstPosition[estimatedMeans, Max[estimatedMeans]][[1]];
action = If[RandomReal[] < \[Epsilon], RandomInteger[{1, nBandits}], greedyAction];
actionCounts[[action]] += 1;
reward = bandits[[action]] + RandomVariate[NormalDistribution[]];
totalRewards[[action]] += reward;
runningTotal += reward;
estimatedMeans[[action]] = totalRewards[[action]] / actionCounts[[action]];
Sow[runningTotal / t, "average"];
optimalBandit = FirstPosition[bandits, Max[bandits]][[1]];
If[action == optimalBandit, optimalActionCount += 1];
Sow[optimalActionCount / t, "optimal"];
bandits += RandomVariate[NormalDistribution[0, meanDrift], nBandits], {t, timeSteps}], {"average", "optimal"}][[2, All, 1]];
ensembleAverageRewards += averageReward;
ensembleOptimalRatio += optimalRatio, {r, runs}];
ensembleAverageRewards /= runs;
ensembleOptimalRatio /= runs, {r, t}]


ListPlot[Table[{t, ensembleAverageRewards[[t]]}, {t, timeSteps}]]
ListPlot[Table[{t, ensembleOptimalRatio[[t]]}, {t, 2, timeSteps}]]



