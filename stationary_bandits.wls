#!/usr/bin/env wolframscript
(* ::Package:: *)

(* 10-armed testbed experiment: compares epsilon-greedy methods with different exploration rates *)

nBandits=10;
timeSteps=1000;
runs=2000;
\[Epsilon]List={0,0.01,0.1};


Monitor[Do[ensembleAverageRewards[\[Epsilon]]=ConstantArray[0,timeSteps];
ensembleOptimalRatio[\[Epsilon]]=ConstantArray[0,timeSteps];
Do[(* Each bandit has reward drawn from N(q*(a), 1) where q*(a) ~ N(0, 1) *)
banditMeans=RandomVariate[NormalDistribution[0,1],nBandits];
bandits=NormalDistribution[#,1]&/@banditMeans;
estimatedMeans=ConstantArray[0,nBandits];
totalRewards=ConstantArray[0,nBandits];
actionCounts=ConstantArray[0,nBandits];
optimalBandit=FirstPosition[banditMeans,Max[banditMeans]][[1]];
{averageReward,optimalRatio}=Reap[Do[greedyChoice=FirstPosition[estimatedMeans,Max[estimatedMeans]][[1]];
(* Epsilon-greedy action selection: explore with probability ε, otherwise exploit *)
choice=If[RandomReal[]<\[Epsilon],RandomInteger[{1,nBandits}],greedyChoice];
actionCounts[[choice]]+=1;
reward=RandomVariate[bandits[[choice]]];
totalRewards[[choice]]+=reward;
estimatedMeans[[choice]]=totalRewards[[choice]]/actionCounts[[choice]]; (* Sample average *)
Sow[Total[totalRewards]/t,"average"]; (* Cumulative average reward *)
Sow[actionCounts[[optimalBandit]]/t,"optimal"], (* Fraction of optimal actions *)
{t,timeSteps}],{"average","optimal"}][[2,All,1]];
ensembleAverageRewards[\[Epsilon]]+=averageReward;
ensembleOptimalRatio[\[Epsilon]]+=optimalRatio,{r,runs}];
(* Average performance across all runs for this epsilon value *)
ensembleAverageRewards[\[Epsilon]]/=runs;
ensembleOptimalRatio[\[Epsilon]]/=runs,{\[Epsilon],\[Epsilon]List}],{\[Epsilon],r,t}]


ListPlot[Table[{t,ensembleAverageRewards[#][[t]]},{t,timeSteps}]&/@\[Epsilon]List]
ListPlot[Table[{t,ensembleOptimalRatio[#][[t]]},{t,timeSteps}]&/@\[Epsilon]List]
