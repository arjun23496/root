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

#include "TMVA/DNN/DeepNet.h"

int test_file(){

	std::cout << "Start GAN Training"

	// Read data
	TFile *input(0);
	TString fname = "./tmva_class_example.root";
	if (!gSystem->AccessPathName( fname )) {
		input = TFile::Open( fname ); // check if file in local directory exists
	}
	else {
		TFile::SetCacheFileDir(".");
		input = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD");
	}
	if (!input) {
		std::cout << "ERROR: could not open data file" << std::endl;
		exit(1);
	}
	std::cout << "--- GAN: Using input file: " << input->GetName() << std::endl;

	return 0;
}