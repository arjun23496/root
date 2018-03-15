#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"
#include "TMVA/DNN/Architectures/Cpu.h"

using namespace TMVA;

int test_file(){

	std::cout << "Start GAN Training";

	// // Read data
	// TFile *input(0);
	// TString fname = "./tmva_class_example.root";
	// if (!gSystem->AccessPathName( fname )) {
	// 	input = TFile::Open( fname ); // check if file in local directory exists
	// }
	// else {
	// 	TFile::SetCacheFileDir(".");
	// 	input = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD");
	// }
	// if (!input) {
	// 	std::cout << "ERROR: could not open data file" << std::endl;
	// 	exit(1);
	// }
	// std::cout << "--- GAN: Using input file: " << input->GetName() << std::endl;

	// // Register train and test trees
	// TTree *signalTree     = (TTree*)input->Get("TreeS");
	// TTree *background     = (TTree*)input->Get("TreeB");

	// // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
	// TString outfileName( "TMVA.root" );
	// TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

	// TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");
	// TMVA::Factory *factory = new TMVA::Factory( "TMVAGAN", outputFile,
    //                                         	"!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );

	// // Define input variables for training
	// dataloader->AddVariable( "myvar1 := var1+var2", 'F' );
	// dataloader->AddVariable( "myvar2 := var1-var2", "Expression 2", "", 'F' );
	// dataloader->AddVariable( "var3",                "Variable 3", "units", 'F' );
	// dataloader->AddVariable( "var4",                "Variable 4", "units", 'F' );

	// // global event weights per tree (see below for setting event-wise weights)
	// Double_t signalWeight     = 1.0;
	// Double_t backgroundWeight = 1.0;

	// // You can add an arbitrary number of signal or background trees
	// dataloader->AddSignalTree    ( signalTree,     signalWeight );
	// dataloader->AddBackgroundTree( background, backgroundWeight );

	// dataloader->SetBackgroundWeightExpression( "weight" );

	// // Cuts for trees(train test cuts)
	// TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
	// TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

	// dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,
    //                                     "nTrain_Signal=1000:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V" );

	// TString layoutString ("Layout=TANH|128,TANH|128,TANH|128,LINEAR");

	// // Training strategies.
	// TString training("LearningRate=1e-1,Momentum=0.9,Repetitions=1,"
	// 					"ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
	// 					"WeightDecay=1e-4,Regularization=L2,"
	// 					"DropConfig=0.0+0.5+0.5+0.5, Multithreading=True");
	// TString trainingStrategyString ("TrainingStrategy=");
	// trainingStrategyString += training;

	// TString dnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:"
    //                       "WeightInitialization=XAVIERUNIFORM");
	
	// dnnOptions.Append (":"); dnnOptions.Append (layoutString);
	// dnnOptions.Append (":"); dnnOptions.Append (trainingStrategyString);

	// TString cpuOptions = dnnOptions + ":Architecture=CPU";
    // factory->BookMethod(dataloader, TMVA::Types::kDNN, "DNN_CPU", cpuOptions);

	// factory->TrainAllMethods();

	// Simple GAN for taking values
	
	// Layer 1 definitions
	DNN::EInitialization init = DNN::EInitialization::kGauss;
	DNN::EActivationFunction afunc = DNN::EActivationFunction::kSigmoid;
	DNN::ERegularization reg = DNN::ERegularization::kNone;

	std::cout << "GAN for taking values for generating normal distribution";

	// DNN::TDenseLayer input_layer = DNN::TDenseLayer();

	DNN::TDeepNet<DNN::TCpu<Double_t>> network;

	return 0;
}