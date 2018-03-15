#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <random>
#include <thread>
#include <atomic>

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

// void print_layer_weights(DNN::TDeepNet<DNN::TCpu<Double_t>> &network){
// 	std::vector<DNN::VGeneralLayer<DNN::TCpu<Double_t>>> layers = network.GetLayers();

// 	for(int i=0; i<layers.size(); i++)
// 	{
// 		std::vector<DNN::TCpuMatrix<double>> layer_weights = layers[i].GetWeights();

// 		std::cout << "k: " << layer_weights.size();
// 	}
// }

struct shared_space{
	std::atomic<bool> gen_output_available;
	std::atomic<bool> dis_eval_available;

	DNN::TCpuMatrix<double> disc_prediction;
};

void train_generator(struct shared_space *shared){
	// Global parameters
	int batch_size = 10;
	int steps = 1000;
	int minibatch = 10;
	int discriminator_train = 10;
	int generator_train = 1;
	int nele = minibatch*1;
	double learning_rate = 0.00001;
	bool real_train = true;

	// Initialize pointers
	double *g_input_ptr = NULL;
	double *g_output_ptr = NULL;
	double *g_target_ptr = NULL;
	double *weights_ptr = NULL;

	// Set random number generators
	std::default_random_engine rand_generator;
	std::uniform_real_distribution<double> source_distribution(0.0,1.0);
	std::normal_distribution<double> target_distribution(1.0,1.0);

	std::cout << "\ngenerator training started";

	// Layer definitions
	DNN::EInitialization init = DNN::EInitialization::kGauss;
	DNN::EActivationFunction afunc = DNN::EActivationFunction::kSigmoid;
	DNN::EActivationFunction fact = DNN::EActivationFunction::kTanh;
	DNN::ERegularization reg = DNN::ERegularization::kNone;
	DNN::ELossFunction loss = DNN::ELossFunction::kMeanSquaredError;

	// START generator network

	// Layer definitions
	DNN::TDenseLayer<DNN::TCpu<Double_t>> g_input_layer = DNN::TDenseLayer<DNN::TCpu<Double_t>>(batch_size, 1, 10, init, 0, afunc, reg, 0);
	DNN::TDenseLayer<DNN::TCpu<Double_t>> g_output_layer = DNN::TDenseLayer<DNN::TCpu<Double_t>>(batch_size, 10, 1, init, 0, fact, reg, 0);	

	// Initialize network
	DNN::TDeepNet<DNN::TCpu<Double_t>> generator;

	// Add dense layer to network
	generator.AddDenseLayer(&g_input_layer);
	generator.AddDenseLayer(&g_output_layer);
	generator.SetLossFunction(loss);

	// END generator network

	// Initialize networks
	generator.Initialize();

	// Initialize input arrays
	DNN::TCpuMatrix<double> g_input = DNN::TCpuMatrix<double>(minibatch, 1);
	DNN::TCpuMatrix<double> g_output = DNN::TCpuMatrix<double>(minibatch, 1);
	DNN::TCpuMatrix<double> g_target = DNN::TCpuMatrix<double>(minibatch, 1);
	DNN::TCpuMatrix<double> weights = DNN::TCpuMatrix<double>(minibatch, 1);

	// create input vectors
	std::vector<DNN::TCpuMatrix<double>> g_input_v = { g_input };
	std::vector<DNN::TCpuMatrix<double>> g_output_v = { g_output };

	g_input_ptr = g_input.GetRawDataPointer();
	g_output_ptr = g_output.GetRawDataPointer();
	g_target_ptr = g_target.GetRawDataPointer();
	weights_ptr = weights.GetRawDataPointer();

	// Intialize constant vectors
	for(int i=0; i<nele; i++)
	{
		// Initialize constant target pointers
		weights_ptr[i] = 1.0;
	}

	// Training
	for(int step=1; step<=steps; step++)
	{
		// cout<<"\n Network at "<<step;
		// print_network(discriminator);
		// print_gradients(discriminator);

		if(real_train){
			// Intialize array with values
			for(int i=0; i<nele; i++)
			{
				// g_input_ptr[i] = target_distribution(rand_generator);
				g_input_ptr[i] = i;
			}

			generator.Forward(g_input_v);
			
			std::cout << "\n g_pred: "<< shared->disc_prediction(0, 0);

			generator.Backward(g_input_v, shared->disc_prediction, weights);
			generator.Update(learning_rate);
		}
	}
}

void train_discriminator(struct shared_space *shared){
	// Global parameters
	int batch_size = 10;
	int steps = 1000;
	int minibatch = 10;
	int discriminator_train = 10;
	int generator_train = 1;
	int nele = minibatch*1;
	double learning_rate = 0.0001;
	bool real_train = true;

	// Initialize pointers
	double *d_input_ptr = NULL;
	double *d_output_ptr = NULL;
	double *d_target_real_ptr = NULL;
	double *d_target_fake_ptr = NULL;
	double *weights_ptr = NULL;

	// Set random number generators
	std::default_random_engine rand_generator;
	std::uniform_real_distribution<double> source_distribution(0.0,1.0);
	std::normal_distribution<double> target_distribution(1.0,1.0);

	std::cout << "\nDiscriminator Training started";

	// Layer definitions
	DNN::EInitialization init = DNN::EInitialization::kGauss;
	DNN::EActivationFunction afunc = DNN::EActivationFunction::kSigmoid;
	DNN::EActivationFunction fact = DNN::EActivationFunction::kTanh;
	DNN::ERegularization reg = DNN::ERegularization::kNone;
	DNN::ELossFunction loss = DNN::ELossFunction::kMeanSquaredError;

	// START Discriminator Network

	// Layer definitions
	DNN::TDenseLayer<DNN::TCpu<Double_t>> d_input_layer = DNN::TDenseLayer<DNN::TCpu<Double_t>>(batch_size, 1, 50, init, 0, afunc, reg, 0);
	DNN::TDenseLayer<DNN::TCpu<Double_t>> d_output_layer = DNN::TDenseLayer<DNN::TCpu<Double_t>>(batch_size, 50, 1, init, 0, afunc, reg, 0);	

	// Initialize network
	DNN::TDeepNet<DNN::TCpu<Double_t>> discriminator;

	// Add dense layer to network
	discriminator.AddDenseLayer(&d_input_layer);
	discriminator.AddDenseLayer(&d_output_layer);
	discriminator.SetLossFunction(loss);

	// END Discriminator Network

	// Initialize networks
	discriminator.Initialize();

	// discriminator.Print();

	// Initialize input arrays
	DNN::TCpuMatrix<double> d_input = DNN::TCpuMatrix<double>(minibatch, 1);
	DNN::TCpuMatrix<double> d_output = DNN::TCpuMatrix<double>(minibatch, 1);
	DNN::TCpuMatrix<double> g_input = DNN::TCpuMatrix<double>(minibatch, 1);
	DNN::TCpuMatrix<double> g_output = DNN::TCpuMatrix<double>(minibatch, 1);
	DNN::TCpuMatrix<double> d_target_real = DNN::TCpuMatrix<double>(minibatch, 1);
	DNN::TCpuMatrix<double> d_target_fake = DNN::TCpuMatrix<double>(minibatch, 1);
	DNN::TCpuMatrix<double> g_target = DNN::TCpuMatrix<double>(minibatch, 1);
	DNN::TCpuMatrix<double> weights = DNN::TCpuMatrix<double>(minibatch, 1);

	// create input vectors
	std::vector<DNN::TCpuMatrix<double>> d_input_v = { d_input };
	std::vector<DNN::TCpuMatrix<double>> d_output_v = { d_output };
	std::vector<DNN::TCpuMatrix<double>> g_input_v = { g_input };
	std::vector<DNN::TCpuMatrix<double>> g_output_v = { g_output };

	d_input_ptr = d_input.GetRawDataPointer();
	d_output_ptr = d_output.GetRawDataPointer();
	d_target_real_ptr = d_target_real.GetRawDataPointer();
	d_target_fake_ptr = d_target_fake.GetRawDataPointer();
	weights_ptr = weights.GetRawDataPointer();

	// Intialize constant vectors
	for(int i=0; i<nele; i++)
	{
		// Initialize constant target pointers
		d_target_real_ptr[i] = 1;
		d_target_fake_ptr[i] = 0;
		weights_ptr[i] = 1;
	}

	// Training
	for(int step=1; step<=steps; step++)
	{
		// Intialize array with values
		for(int i=0; i<nele; i++)
		{
			d_input_ptr[i] = target_distribution(rand_generator);
			// d_input_ptr[i] = 1;
		}

		discriminator.Forward(d_input_v);
		
		discriminator.Backward(d_input_v, d_target_real, weights);
		discriminator.Update(learning_rate);

		discriminator.Prediction(shared->disc_prediction, DNN::EOutputFunction::kIdentity);

		std::cout << "\n d_pred: "<< shared->disc_prediction(0,0);
	}
}

int test_file(){
	pthread_t generator_thread, discriminator_thread;
	struct shared_space *shared = (struct shared_space*)malloc(sizeof(struct shared_space));
	std::thread disc_thread;
	std::thread gen_thread;

	shared->dis_eval_available = false;
	shared->gen_output_available = false;
	shared->disc_prediction = DNN::TCpuMatrix<double>(10, 1);

	std::cout << "\nStarting threads";

	disc_thread = std::thread(train_discriminator, shared);
	gen_thread = std::thread(train_generator, shared);

	disc_thread.join();
	std::cout << "\nDiscriminator thread done";

	gen_thread.join();
	std::cout << "\nGenerator thread done";

	return 0;
}