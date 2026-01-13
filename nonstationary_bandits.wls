#!/usr/bin/env wolframscript
(* ::Package:: *)

nBandits = 10;
timeSteps = 10000;
runs = 2000;
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
{averageReward, optimalRatio} = Reap[Do[greedyChoice = FirstPosition[estimatedMeans, Max[estimatedMeans]];
choice = If[RandomReal[] < \[Epsilon], RandomInteger[{1, nBandits}], greedyChoice];
actionCounts[[choice]] += 1;
reward = bandits[[choice]] + RandomVariate[NormalDistribution[]];
totalRewards[[choice]] += reward;
runningTotal += reward;
estimatedMeans[[choice]] = totalRewards[[choice]] / actionCounts[[choice]];
Sow[runningTotal / t, "average"];
optimalBandit = FirstPosition[bandits, Max[bandits]][[1]];
If[choice == optimalBandit, optimalActionCount += 1];
Sow[optimalActionCount / t, "optimal"];
bandits += RandomVariate[NormalDistribution[0, meanDrift], nBandits], {t, timeSteps}], {"average", "optimal"}][[2, All, 1]];
ensembleAverageRewards += averageReward;
ensembleOptimalRatio += optimalRatio, {r, runs}];
ensembleAverageRewards /= runs;
ensembleOptimalRatio /= runs, {r, t}]


ListPlot[Table[{t, ensembleAverageRewards[[t]]}, {t, timeSteps}]]
ListPlot[Table[{t, ensembleOptimalRatio[[t]]}, {t, timeSteps}]]
