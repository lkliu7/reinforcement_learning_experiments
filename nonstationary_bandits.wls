#!/usr/bin/env wolframscript
(* ::Package:: *)

nBandits = 10;
timeSteps = 10000;
runs = 2000;
\[Epsilon] = 0.1;
meanDrift = 0.01;


bandits = ConstantArray[0, nBandits];
estimatedMeans = ConstantArray[0, nBandits];
totalRewards = ConstantArray[0, nBandits];
actionCounts = ConstantArray[0, nBandits];
{averageReward, optimalRatio} = Reap[Do[greedyChoice = FirstPosition[estimatedMeans, Max[estimatedMeans]];
choice = If[RandomReal[] < \[Epsilon], RandomInteger[{1, nBandits}], greedyChoice];
actionCounts[[choice]] += 1;
reward = bandits[[choice]] + RandomVariate[NormalDistribution[]];
totalRewards[[choice]] += reward;
estimatedMeans[[choice]] = totalRewards[[choice]] / actionCounts[[choice]];
Sow[Total[totalRewards] / t, "average"];
optimalBandit = FirstPosition[bandits, Max[bandits]][[1]];
Sow[actionCounts[[optimalBandit]] / t, "optimal"];
bandits += RandomVariate[NormalDistribution[0, meanDrift], nBandits], {t, timeSteps}], {"average", "optimal"}][[2, All, 1]];

