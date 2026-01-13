#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*nBandits=10;*)
(*steps=10000;*)
(*runs=2000;*)
(*\[Epsilon]List={0,0.01,0.1};*)
(*meanDrift=0.01*)


(* ::Input:: *)
(*banditMeans=ConstantArray[0,nBandits];*)
(*bandit[i_]:=NormalDistribution[banditMeans[[i]],1]*)
(*Do[banditMeans+=RandomVariate[NormalDistribution[0,meanDrift],nBandits],{t,steps}]*)
