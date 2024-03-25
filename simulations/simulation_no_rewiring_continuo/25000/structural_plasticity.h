#ifndef STRUCTURAL_PLASTICITY_H
#define STRUCTURAL_PLASTICITY_H

#include <random>
#include <vector>


float my_erfinvf (float a);

double erfm1(double x);

class simulation
{
  // connection rule
  int connection_rule;
  // use discrete rates or lognormal distribution
  bool lognormal_rate;
  // multapses can be allowed or forbidden
  bool allow_multapses;
  // step (in n. of patterns) for connection recombination (0: no recombination)
  int r;
  // to save memory, patterns can be generated on the fly from their index
  // without storing the whole set in memory
  bool generate_patterns_on_the_fly;
  
  // number of training patterns
  int T;
  // number of test patterns
  int n_test;
  // connections per layer-2-neuron, i.e. indegree (double needed)
  double C;
  // probability of high rate for layer 1
  double alpha1;
  // probability of high rate for layer 2
  double alpha2;
  // number of neurons in pop 1
  int N1;
  // number of neurons in pop 2
  int N2;  
  // baseline weight
  double Wb;
  // consolidated weight
  double Ws;
  // low rate for layer 1 [Hz]
  double nu_l_1;
  // high rate for layer 1 [Hz]
  double nu_h_1;
  // low rate for layer 2 [Hz]
  double nu_l_2;
  // high rate for layer 2 [Hz]
  double nu_h_2;
  // master seed for random number generation
  uint_least32_t master_seed;
  // seed offset for random number generation
  int seed_offset;
  // perform training
  bool train_flag;
  // perform test
  bool test_flag;
  // load network from file
  bool load_network;
  // save network to file after training
  bool save_network;
  // number of training patterns between saving network and status
  int train_block_size;
  // number of test patterns between saving the status
  int test_block_size;
  // random number generator (Mersenne Twister MT 19937)
  //std::mt19937 rnd_gen;
  // random number generators for different parts of the simulation
  // separated for reproducibility purpose in simulation rounds
  std::mt19937 rnd_gen_network;
  std::mt19937 rnd_gen_train;
  std::mt19937 rnd_gen_test;
  // Arbitrary offsets, fixed for reproducibility in simulation rounds.
  // Must be larger than 9999. No need to change them
  // they are added to the master seed and to seed_offset
  int seed_offset_network;
  int seed_offset_train_set;
  int seed_offset_train;
  int seed_offset_test;
  static const int max_file_name_size = 1000;
  // network file name
  char network_file_name[max_file_name_size];
  // status file name
  char status_file_name[max_file_name_size];
  // firing rate of layer-1-neurons in training patterns
  std::vector <std::vector<double> > rate_L1_train_set;
  // firing rate of layer-2-neurons in training patterns
  std::vector <std::vector<double> > rate_L2_train_set;
  // firing rate of layer-1-neurons in test patterns
  std::vector <std::vector<double> > rate_L1_test_set;
  // firing rate of layer-2-neurons in test patterns
  std::vector <std::vector<double> > rate_L2_test_set;
  // Same for single pattern
  // firing rate of layer-1-neurons in single pattern
  std::vector<double> rate_L1_pattern;
  // firing rate of layer-2-neurons in training pattern
  std::vector<double> rate_L2_pattern;
  // test pattern indexes
  std::vector<int> ie_test_arr;
  // test pattern indexes can be ordered sequentially or randomly 
  bool random_test_order;
  ///////////////////////////////////////////
  // theoretical values of model quantities
  ///////////////////////////////////////////
  // probability of having consolidated the
  // connection for at least one instance
  double p;
  // complement of alpha1
  double q1;
  // average rate layer 1
  double rm1;
  // complement of alpha2
  double q2;
  // average rate layer 2
  double rm2;

  // rate lognormal distribution parameters for layer 1
  double sigma_ln1;
  double mu_ln1;
  double yt_ln1;
  double rt1;

  // rate lognormal distribution parameters for layer 2
  double sigma_ln2;
  double mu_ln2;
  double yt_ln2;
  double rt2;
  
  // average num. of consolidated connections
  double k;
  // <r^2> for layer 1 (we do not need it for layer 2)
  double rsq1;
  // rate variance layer 1
  double var_r1;
  // <k^2> and variance of k
  double k2;
  double var_k;
  
  // theoretical values of Sb, S2 and sigma^2 Sb
  double Sbt;
  double S2t;
  double S2t_chc;
  double var_St;
  
  // Connection index vector
  // conn_index[i2][ic] = i1 = index of the neuron of pop 1 connected
  // to i2 through connection ic of neuron i2 of pop 2
  std::vector<std::vector<int> > conn_index;

  // Connection weight vector
  // w[i2][ic] = weight of the connection ic of neuron i2
  std::vector<std::vector<double> > w;
  // maximum number of connections per target neuron
  int iC_reserve;
  // array of number of connections per target neuron
  std::vector<int> n_conn_2; 

  // output file
  FILE *fp_out;
  // header file
  FILE *fp_head;
  char file_name_head[max_file_name_size];
  // epsilon (margin for numerical compatibility)
  const double eps = 1.0e-6;

  // range of pattern indexes for training
  int ie0_train;
  int ie1_train;
  // index of trainig pattern group in case simulation must be splitted 
  int j_train;
  // range of pattern indexes for test
  int ie0_test;
  int ie1_test;
  // index of test pattern group in case simulation must be splitted 
  int j_test;

public:
  // constructor
  simulation();
  // read command line arguments
  int readArg(int argc, char *argv[]);
  // initialize simulation
  int init(int argc, char *argv[]);
  // evaluate and print theoretical values of relevant quantities
  int evalTheoreticalValues();
  // read parameters from file
  int readParams(char *filename);
  // generate random training set
  int generateRandomSet();
  // generate random training pattern
  int generateRandomPattern(double *rate_L1, double *rate_L2, int ie);
  
  // copy train set to test set
  int copyTrainToTest();
  // generate indexes for test patterns
  int extractTestPatternIndexes();
  // load simulation status  
  int loadStatus();
  // save simulation status  
  int saveStatus(std::string s, int j);
  // save network connections in a file
  int saveNetwork();
  // load network connections from file
  int loadNetwork();
  // allocate arrays for network
  int allocateNetwork();
  // create network
  int createNetwork();
  // destroy and create non-consolidated connections
  int rewireConnections();
  // train network with training set
  int train();
  // evaluate output with test set
  int test();
  // run simulation
  int run();
};


#endif
