#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#define THREAD_MAXNUM omp_get_max_threads()
#define THREAD_IDX omp_get_thread_num()
#else
#define THREAD_MAXNUM 1
#define THREAD_IDX 0
#endif

#include "structural_plasticity.h"

double erfm1(double x)
{
  return sqrt(2.0)*my_erfinvf((float)(2.0*x - 1.0));
}

int randomGeneratorFromInt(std::mt19937 &rnd_gen, int seed, int value)
{
  // seed probability distribution generators
  rnd_gen.seed(seed + value);
  // Advances the internal state by z notches,
  // as if operator() was called z times, but without generating
  // any numbers in the process.
  rnd_gen.discard(5);
  
  return 0;
}



enum CONNECTION_RULE{FIXED_INDEGREE=0, POISSON_INDEGREE, FIXED_TOTAL_NUMBER,
  POISSON_TOTAL_NUMBER};
std::string connection_rule_name[] = {"fixed_indegree", "poisson_indegree",
  "fixed_total_number", "poisson_total_number"};

simulation::simulation()
{
  connection_rule = POISSON_INDEGREE;
  // use discrete rates or lognormal distribution
  lognormal_rate = true;
  // multapses can be allowed or forbidden
  allow_multapses = true;
  // step (in n. of patterns) for connection recombination (0: no recombination)
  change_conn_step = 100;
  // to save memory, patterns can be generated on the fly from their index
  // without storing the whole set in memory
  generate_patterns_on_the_fly = true;
  // number of training patterns
  T = 1000;
  // n. of test patterns
  n_test = 1000;
  // connections per layer-2-neuron, i.e. indegree (double needed)
  C = 5000.0;
  // probability of high rate for layer 1
  p1 = 1.0e-3;
  // probability of high rate for layer 2
  p2 = 1.0e-3;
  // number of neurons in pop 1
  N1 = 100000;
  // number of neurons in pop 2
  N2 = 100000;  
  // baseline weight
  W0 = 0.1;
  // consolidated weight
  Wc = 1.0;
  // low rate for layer 1 [Hz]
  rl1 = 2.0;
  // high rate for layer 1 [Hz]
  rh1 = 50.0;
  // low rate for layer 2 [Hz]
  rl2 = 2.0;
  // high rate for layer 2 [Hz]
  rh2 = 50.0;
  // master seed for random number generation
  master_seed = 123456;
  // Arbitrary offsets, fixed for reproducibility in simulation rounds.
  // Must be larger than 9999 and smaller than 990000. No need to change them.
  // They are added to the master seed and to seed_offset
  seed_offset_network = 100000;
  seed_offset_train_set = 200000;
  seed_offset_train = 300000;
  seed_offset_test = 400000;

  // perform training
  train_flag = true;
  // perform test
  test_flag = true;
  // load network from file
  load_network = false;
  // save network to file after training
  save_network = true;
  // by default test pattern indexes are ordered randomly 
  random_test_order = true;
  // number of training patterns between saving network and status
  train_block_size = 5000;
  // number of test patterns between saving the status
  test_block_size = 5000;
  // seed offset for random number generation
  seed_offset = 0;
}  

int simulation::readArg(int argc, char *argv[])
{  
  if (argc==2 || argc==3) {
    if(strspn(argv[1], "0123456789") != strlen(argv[1])) {
      printf("Error: first argument must be a number\n");
      printf("Usage: %s rnd_seed_offset [parameter_file]\n", argv[0]);
      exit(-1);
    }
    sscanf(argv[1], "%d", &seed_offset);
  }
  else {
    printf("Usage: %s rnd_seed_offset [parameter_file]\n", argv[0]);
    exit(0);
  }

  return 0;
}

int simulation::init(int argc, char *argv[])
{
  readArg(argc, argv);
  if (argc==3) {
    readParams(argv[2]);
  }
  
  if (!allow_multapses && change_conn_step!=0) {
    std::cerr << "Connection recombination allowed only "
      "with multapses allowed\n";
    exit(-1);
  }
  if (!allow_multapses && connection_rule!=FIXED_INDEGREE) {
    std::cerr << "Multapses can be forbidden only "
      "with fixed_indegree connection rule\n";
    exit(-1);
  }
  //if (connection_rule!=FIXED_INDEGREE && change_conn_step!=0) {
  //  std::cerr << "Connection recombination currently allowed only "
  //    "with fixed_indegree connection rule\n";
  //  exit(-1);
  //}

  sprintf(file_name_head, "mem_head_%04d.dat", seed_offset);

  sprintf(status_file_name, "status_%04d.dat", seed_offset);

  return 0;
}

int simulation::evalTheoreticalValues()
{
  // probability of having consolidated the
  // connection at least for one instance
  p = 1.0 - pow(1.0 - p1*p2, T);

  // complement of p1
  q1 = 1.0 - p1;
  // average rate layer 1
  rm1 = p1*rh1 + q1*rl1;

  // complement of p2
  q2 = 1.0 - p2;
  // average rate layer 2
  rm2 = p2*rh2 + q2*rl2;

  // rate lognormal distribution parameters for layer 1
  sigma_ln1 = erfm1(q1) - erfm1(q1*rl1/rm1);
  mu_ln1 = log(rm1) - sigma_ln1*sigma_ln1/2.0;
  yt_ln1 = erfm1(q1)*sigma_ln1 + mu_ln1;
  if(lognormal_rate) {
    rt1 = exp(yt_ln1);
  }
  else {
    rt1 = (rh1 + rl1) / 2.0;
  }

  // rate lognormal distribution parameters for layer 2
  sigma_ln2 = erfm1(q2) - erfm1(q2*rl2/rm2);
  mu_ln2 = log(rm2) - sigma_ln2*sigma_ln2/2.0;
  yt_ln2 = erfm1(q2)*sigma_ln2 + mu_ln2;
  if(lognormal_rate) {
    rt2 = exp(yt_ln2);
  }
  else {
    rt2 = (rh2 + rl2) / 2.0;
  }
  
  // average consolidated connections
  k = p*C;
  // <r^2> for layer 1 (we do not need it for layer 2)
  rsq1 = p1*rh1*rh1 + (1.0 - p1)*rl1*rl1;
  // rate variance layer 1
  if (lognormal_rate) {
    var_r1 = (exp(sigma_ln1*sigma_ln1) -1.0)
      * exp(2.0*mu_ln1 + sigma_ln1*sigma_ln1);
  }
  else {
    var_r1 = rsq1 - rm1*rm1;
  }
  // calculation for variance of k
  k2 = C*(C - 1)*pow(1.0 - (2.0 - p1)*p1*p2, T)
    - C*(2*C - 1)*pow(1.0 - p1*p2, T) + C*C;
  var_k = k2 - k*k;
  
  // theoretical estimation of Sb, S2 and sigma^2 Sb
  Sbt = Wc*k*rm1 + W0*(C-k)*rm1;
  S2t = rh1*Wc*p1*C + rl1*(1.0-p1)*(W0*C + (Wc - W0)*k);
  S2t_chc = rh1*Wc*p1*C + rm1*(1.0-p1)*(W0*C + (Wc - W0)*k);
  var_St = (Wc*Wc*k + W0*W0*(C-k))*var_r1
    + (Wc - W0)*(Wc - W0)*rm1*rm1*var_k;

  // Variable n. of connections per target neuron (Poisson distribution)
  double C_m = C;
  double var_C = C;
  double C2_m = C*(C + 1.0);
  double eta = pow(1.0 - p1*p2, T);
  double csi = pow(1.0 - (2.0 - p1)*p1*p2, T);
  
  double var_S_poiss = pow(W0 + p*(Wc - W0), 2.0)*rm1*rm1*var_C
    + C_m*(p*Wc*Wc + eta*W0*W0)*var_r1
    + pow(Wc - W0, 2)*rm1*rm1*((C2_m - C_m)*csi + C_m*eta - C2_m*eta*eta);

  printf("Number of openmp threads: %d\n", THREAD_MAXNUM);
  // print of theoretical estimations
  printf("p: %.9lf\n", p);
  printf("sigma2r layer 1 (theoretical): %.4lf\n", var_r1);
  printf("sigma2k (theoretical): %.4lf\n", var_k);
  printf("S2 (theoretical): %.4lf\n", S2t);
  printf("S2 with connection recombination (theoretical): %.4lf\n", S2t_chc);
  printf("Sb (theoretical):  %.4lf\n", Sbt);
  printf("sigma2S (theoretical): %.4lf\n", var_St);
  printf("sigma2S with Poisson indegree (theoretical): %.4lf\n", var_S_poiss);
  fflush(stdout);
  //std::cout << (Wc*Wc*k + W0*W0*(C-k))*sigma2r << "\n";
  //std::cout << (Wc - W0)*(Wc - W0)*r*r*sigma2k << "\n";

  // same but saved in the header file
  fp_head = fopen(file_name_head, "wt");
  fprintf(fp_head, "Number of openmp threads: %d\n", THREAD_MAXNUM);
  fprintf(fp_head, "p: %.9lf\n", p);
  fprintf(fp_head, "sigma2r layer 1 (theoretical): %.4lf\n", var_r1);
  fprintf(fp_head, "sigma2k (theoretical): %.4lf\n", var_k);
  fprintf(fp_head, "S2 (theoretical): %.4lf\n", S2t);
  fprintf(fp_head, "S2 with connection recombination (theoretical): %.4lf\n",
	  S2t_chc);
  fprintf(fp_head, "Sb (theoretical):  %.4lf\n", Sbt);
  fprintf(fp_head, "sigma2S (theoretical): %.4lf\n", var_St);
  fprintf(fp_head, "sigma2S with Poisson indegree (theoretical): %.4lf\n",
	  var_S_poiss);
  fclose(fp_head);

  return 0;
}

int simulation::copyTrainToTest()
{
  std::cout << "Copying train set to test set...\n" << std::flush;
  // training set and test set are the same for now

  //rate_L1_test_set = rate_L1_train_set;
  //rate_L2_test_set = rate_L2_train_set;
  rate_L1_test_set.resize(n_test);
  rate_L2_test_set.resize(n_test);
  
  for (int ie=0; ie<n_test; ie++) {
    if (ie%100 == 0) {
      std::cout << "Generating test pattern n. " << ie + 1 << " / " << n_test << "\n" << std::flush;
    }
    rate_L1_test_set[ie] = rate_L1_train_set[ie];
    rate_L2_test_set[ie] = rate_L2_train_set[ie];
  }
  std::cout << "Done copying train set to test set.\n" << std::flush;
  
  return 0;
}

int simulation::run()
{
  // evaluate and print theoretical values of relevant quantities
  evalTheoreticalValues();
  
  rate_L1_pattern.resize(N1);
  rate_L2_pattern.resize(N2);
  // generate random training set
  if (!generate_patterns_on_the_fly) {
    generateRandomSet();
    // copy train set to test set
    copyTrainToTest();
  }
  
  ie0_train = 0;
  ie1_train = T;
  j_train = 0;
  ie0_test = 0;
  ie1_test = n_test;
  j_test = 0;

  if (T>train_block_size || n_test>test_block_size) {
    loadStatus();
  }
  
  allocateNetwork();
  if (load_network) {
    int j = 0;
    if (T>train_block_size) {
      if (train_flag && j_train>=1) {
	j = j_train - 1;
      }
      else if (test_flag) {
	j = (T + train_block_size - 1) / train_block_size - 1;
      }
    }
    sprintf(network_file_name, "network_%04d_%04d.dat", seed_offset, j);
    loadNetwork();
  }
  else {
    createNetwork();
  }
  // train network with training set
  if (train_flag) {
    train();
  }
  // evaluate output with test set
  if (test_flag) {
    extractTestPatternIndexes();
    test();
  }
  
  return 0;
}

// train network with training set
int simulation::train()
{
  std::cout << "Training...\n" << std::flush;

  // seed RNGs
  rnd_gen_train.seed(master_seed + seed_offset_train + seed_offset
		     + j_train*1000000);

  // loop over the T patterns
  for (int ie=ie0_train; ie<ie1_train; ie++) {
    if (ie%100 == 0) {
      std::cout << "\tTraining pattern n. " << ie + 1 << " / " << T << "\n";
    }
    double *rate_L1;
    double *rate_L2;
    if (generate_patterns_on_the_fly) {
      rate_L1 = &rate_L1_pattern[0];
      rate_L2 = &rate_L2_pattern[0];
      generateRandomPattern(rate_L1, rate_L2, ie);
    }
    else {
      rate_L1 = &rate_L1_train_set[ie][0];
      rate_L2 = &rate_L2_train_set[ie][0];
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i2=0; i2<N2; i2++) {
      if (rate_L2[i2] > rt2) {
	for (int ic=0; ic<n_conn_2[i2]; ic++) {
	  int i1 = conn_index[i2][ic];
	  if (rate_L1[i1] > rt1) {
	    // synaptic consolidation
	    w[i2][ic] = Wc;
	  }
	}
      }
    }
    
    if (change_conn_step!=0 && ((ie+1)%change_conn_step==0)
	&& allow_multapses) {
      //std::cout << "Training pattern n. " << ie + 1 << " / " << T << "\n";
      rewireConnections();
    }
  }
  sprintf(network_file_name, "network_%04d_%04d.dat", seed_offset,
	  j_train);
  if (T>train_block_size || n_test>test_block_size) {
    saveNetwork();
    saveStatus("train", j_train+1);
    if (!save_network && j_train>=1) {
      sprintf(network_file_name, "network_%04d_%04d.dat", seed_offset,
	      j_train-1);
      remove(network_file_name);
    }
  }
  else if (save_network) {
    saveNetwork();
  }
  std::cout << "Done training.\n" << std::flush;

  return 0;
}


int simulation::test()
{
  std::cout << "Testing...\n" << std::flush;

  // seed RNGs
  rnd_gen_test.seed(master_seed + seed_offset_test + seed_offset
		    + (j_test+10000)*1000000);

  // Test phase
  char file_name_out[] = "mem_out_xxxx_yyyy.dat";
  sprintf(file_name_out, "mem_out_%04d_%04d.dat", seed_offset, j_test);
  fp_out = fopen(file_name_out, "wt");

  printf("Simulated\n");
  printf("ie\tSb\tS2\tsigma2S\n");
  fflush(stdout);
  
  int P2_arr[THREAD_MAXNUM];
  double S2_sum_arr[THREAD_MAXNUM];
  double Sb_sum_arr[THREAD_MAXNUM];
  double Sb_square_sum_arr[THREAD_MAXNUM];

  double *rate_L1;
  double *rate_L2;
  // test over the patterns previously shown
  for (int ie1 = ie0_test; ie1<ie1_test; ie1++) {
    int ie = ie_test_arr[ie1];
    if (generate_patterns_on_the_fly) {
      rate_L1 = &rate_L1_pattern[0];
      rate_L2 = &rate_L2_pattern[0];
      generateRandomPattern(rate_L1, rate_L2, ie);
    }
    else {
      rate_L1 = &rate_L1_test_set[ie][0];
      rate_L2 = &rate_L2_test_set[ie][0];
    }
    
    double S2_sum = 0.0;
    // for mean calculation - counter of neurons at rh in pop 2
    int P2 = 0;
    double Sb_sum = 0.0;
    double Sb_square_sum = 0.0;
    
    for (int ith=0; ith<THREAD_MAXNUM; ith++) {
      P2_arr[ith] = 0.0;
      S2_sum_arr[ith] = 0.0;
      Sb_sum_arr[ith] = 0.0;
      Sb_square_sum_arr[ith] = 0.0;
    }
      
    // loop over pop 2 neurons
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i2=0; i2<N2; i2++) {
      // signal of neurons representing the item
      double S2 = 0.0;
      // signal of neurons not representing the item (i.e. background)
      double Sb = 0.0;
      // consider neurons of pop 2 at high rate (S2)
      if (rate_L2[i2] > rt2) {
      	P2_arr[THREAD_IDX]++;
        // loop for each connection of neuron of pop 2
        for (int ic=0; ic<n_conn_2[i2]; ic++) {
          // look at the neuron of pop 1 connected through the connection
          int i1 = conn_index[i2][ic];
          // compute its contribution to input signal, i.e. r*W
          double s = rate_L1[i1]*w[i2][ic];
          // add the contribution to input signal
          S2 += s;
        }
      }
      // consider neurons of pop 2 at low rate (Sb)
      else {
        for (int ic=0; ic<n_conn_2[i2]; ic++) {
          int i1 = conn_index[i2][ic];
          // compute contribution
          double s = rate_L1[i1]*w[i2][ic];
          // add it to background signal
          Sb += s;
        }
      }
      // sum of input signal over the N2 neurons of pop 2
      Sb_sum_arr[THREAD_IDX] += Sb;
      S2_sum_arr[THREAD_IDX] += S2;
      // needed for computing the variance of the bkg
      Sb_square_sum_arr[THREAD_IDX] += Sb*Sb;
    }
//end of openmp parallel loop
    for (int ith=0; ith<THREAD_MAXNUM; ith++) {
      P2 += P2_arr[ith];
      S2_sum += S2_sum_arr[ith];
      Sb_sum += Sb_sum_arr[ith];
      Sb_square_sum += Sb_square_sum_arr[ith];
    }

    // average of S2 input over the pop 2 neurons that code for the item
    double S2_mean = S2_sum / P2;
    // same for background input (i.e. Sb)
    double Sb_mean = Sb_sum / (N2 - P2);
    // needed for bkg variance calculation <Sb^2> - Sb^2
    double Sb_square_mean = Sb_square_sum / (N2 - P2);
    printf("%d\t%.1lf\t%.1lf\t%.1lf\n", ie, Sb_mean, S2_mean,
	   Sb_square_mean - Sb_mean*Sb_mean); 
    fprintf(fp_out, "%d\t%.1lf\t%.1lf\t%.1lf\n", ie, Sb_mean, S2_mean,
	    Sb_square_mean - Sb_mean*Sb_mean);
    fflush(fp_out);
  }
  if (T>train_block_size || n_test>test_block_size) {
    if ((j_test+1)*test_block_size>=n_test) {
      saveStatus("done", j_test+1);
      if (save_network==false) {
	int j = 0;
	if (T>train_block_size) {
	  j = (T + train_block_size - 1) / train_block_size - 1;
	}
	sprintf(network_file_name, "network_%04d_%04d.dat", seed_offset, j);
	int res = remove(network_file_name);
	if (res != 0) {
	  std::cerr << "Cannot remove network file.\n";
	}
      }
    }
    else {
      saveStatus("test", j_test+1);
    }
  }

  fclose(fp_out);

  std::cout << "Done testing.\n" << std::flush;

  return 0;
}



int simulation::readParams(char *filename)
{
  std::ifstream ifs;
  ifs.open(filename, std::ios::in);
  if (ifs.fail()) {
    std::cerr << "Error opening parameter file\n";
    exit(-1);
  }

  std::string str;
  int line_num = 0;
  while(std::getline(ifs, str)) {
    line_num++;
    std::istringstream ss(str);
    std::string s;
    ss >> s;
    if (s[0]=='#' || s=="") {
      continue;
    }
    if (s=="connection_rule") {
      ss >> connection_rule;
      std::cout << "connection_rule: "
		<< connection_rule_name[connection_rule] << "\n";
    }
    else if (s=="lognormal_rate") {
      ss >> lognormal_rate;
      std::cout  << std::boolalpha
		 << "lognormal_rate: " << lognormal_rate << "\n";
    }
    else if (s=="allow_multapses") {
      ss >> allow_multapses;
      std::cout  << std::boolalpha
		 << "allow_multapses: " << allow_multapses << "\n";
    }
    else if (s=="change_conn_step") {
      ss >> change_conn_step;
      std::cout << "change_conn_step: " << change_conn_step << "\n";
    }
    else if (s=="T") {
      ss >> T;
      std::cout << "T: " << T << "\n";
    }
    else if (s=="n_test") {
      ss >> n_test;
      std::cout << "n_test: " << n_test << "\n";
    }
    else if (s=="C") {
      ss >> C;
      std::cout << "C: " << C << "\n";
    }
    else if (s=="p1") {
      ss >> p1;
      std::cout << "p1: " << p1 << "\n";
    }
    else if (s=="p2") {
      ss >> p2;
      std::cout << "p2: " << p2 << "\n";
    }
    else if (s=="N1") {
      ss >> N1;
      std::cout << "N1: " << N1 << "\n";
    }
    else if (s=="N2") {
      ss >> N2;
      std::cout << "N2: " << N2 << "\n";
    }
    else if (s=="W0") {
      ss >> W0;
      std::cout << "W0: " << W0 << "\n";
    }
    else if (s=="Wc") {
      ss >> Wc;
      std::cout << "Wc: " << Wc << "\n";
    }
    else if (s=="rl1") {
      ss >> rl1;
      std::cout << "rl1: " << rl1 << "\n";
    }
    else if (s=="rh1") {
      ss >> rh1;
      std::cout << "rh1: " << rh1 << "\n";
    }
    else if (s=="rl2") {
      ss >> rl2;
      std::cout << "rl2: " << rl2 << "\n";
    }
    else if (s=="rh2") {
      ss >> rh2;
      std::cout << "rh2: " << rh2 << "\n";
    }
    else if (s=="master_seed") {
      ss >> master_seed;
      std::cout << "master_seed: " << master_seed << "\n";
    }
    else if (s=="save_network") {
      ss >> save_network;
      std::cout  << std::boolalpha
		 << "save_network: " << save_network << "\n";
    }
    else if (s=="load_network") {
      ss >> load_network;
      std::cout  << std::boolalpha
		 << "load_network: " << load_network << "\n";
    }
    else if (s=="train_flag") {
      ss >> train_flag;
      std::cout  << std::boolalpha
		 << "train_flag: " << train_flag << "\n";
    }
    else if (s=="test_flag") {
      ss >> test_flag;
      std::cout  << std::boolalpha
		 << "test_flag: " << test_flag << "\n";
    }
    else if (s=="random_test_order") {
      ss >> random_test_order;
      std::cout  << std::boolalpha
		 << "random_test_order: " << random_test_order << "\n";
    }
    else if (s=="train_block_size") {
      ss >> train_block_size;
      std::cout << "train_block_size: " << train_block_size << "\n";
    }
    else if (s=="test_block_size") {
      ss >> test_block_size;
      std::cout << "test_block_size: " << test_block_size << "\n";
    }
    else {
      std::cerr << "Error in line " << line_num << " of parameter file\n";
      std::cerr << str << "\n";
      std::cerr << "Unrecognized parameter " << s << "\n";
      exit(-1);
    }
    if (ss >> s && s[0]!='#') {
      std::cerr << "Error in line " << line_num << " of parameter file\n";
      std::cerr << str << "\n";
      exit(-1);
    }
  }
    
  return 0;
}


int simulation::generateRandomSet() 
{
  std::cout << "Generating random train set...\n" << std::flush;
  rate_L1_train_set.resize(T);
  rate_L2_train_set.resize(T);
  
  for (int ie=0; ie<T; ie++) {
    rate_L1_train_set[ie].resize(N1);
    rate_L2_train_set[ie].resize(N2);
    if (ie%100 == 0) {
      std::cout << "Training pattern n. " << ie + 1 << " / " << T << "\n" << std::flush;
    }
    generateRandomPattern(&rate_L1_train_set[ie][0],
			  &rate_L2_train_set[ie][0], ie);
  }
  std::cout << "Done generating random train set.\n" << std::flush;
  
  return 0;
}

int simulation::generateRandomPattern(double *rate_L1, double *rate_L2,
				      int ie)  
{
  // initialize random generator from integer (ie)
  std::mt19937 rnd_gen;
  randomGeneratorFromInt(rnd_gen,
			 master_seed + seed_offset_train_set + seed_offset,
			 ie);
  
  // probability distribution generators
  std::uniform_real_distribution<> rnd_uniform(0.0, 1.0);
  std::normal_distribution<double> rnd_normal1(mu_ln1, sigma_ln1);
  std::normal_distribution<double> rnd_normal2(mu_ln2, sigma_ln2);
  
  // extract rate for pop 1 neurons
  //std::vector<double> rate1;
  for (int i1=0; i1<N1; i1++) {
    if (lognormal_rate) {
      //rate1.push_back(exp(rnd_normal1(rnd_gen)));
      rate_L1[i1] = exp(rnd_normal1(rnd_gen));
    }
    else {
      if (rnd_uniform(rnd_gen) < p1) {
	//rate1.push_back(rh1);
	rate_L1[i1] = rh1;
      }
      else {
	//rate1.push_back(rl1);
	rate_L1[i1] = rl1;
      }
    }
  }
  //rate_L1.push_back(rate1);

  // extract rate for pop 2 neurons
  //std::vector<double> rate2;
  for (int i2=0; i2<N2; i2++) {
    if (lognormal_rate) {
      //rate2.push_back(exp(rnd_normal2(rnd_gen)));
      rate_L2[i2] = exp(rnd_normal2(rnd_gen));
    }
    else {
      if (rnd_uniform(rnd_gen) < p2) {
	//rate2.push_back(rh2);
	rate_L2[i2] = rh2;
      }
      else {
	//rate2.push_back(rl2);
	rate_L2[i2] = rl2;
      }
    }
  }
  //rate_L2.push_back(rate2);

  return 0;
}

int simulation::loadStatus()
{
  std::cout << "Loading simulation status...\n" << std::flush;

  std::ifstream ifs;
  ifs.open(status_file_name, std::ios::in);
  if (ifs.fail()) {
    if (train_flag) {
      test_flag = false;
      ie1_train = std::min(train_block_size, T);
    }
    std::cout << "Done loading simulation status.\n" << std::flush;
    return 0;
  }

  load_network = true;
  
  std::string s;
  ifs >> s;
  if (s == "train") {
    test_flag = false;
    ifs >> j_train;
    ifs >> s;
    if (s != "end") {
      std::cerr << "Error reading status file\n";
      exit(-1);
    }
    ie0_train = j_train*train_block_size;
    ie1_train = std::min(ie0_train+train_block_size, T);
  }
  else if (s == "test") {
    train_flag = false;
    ifs >> j_test;
    ifs >> s;
    if (s != "end") {
      std::cerr << "Error reading status file\n";
      exit(-1);
    }
    ie0_test = j_test*test_block_size;
    ie1_test = std::min(ie0_test+test_block_size, n_test);
    if (ie0_test >= n_test) {
      std::cerr << "Error: ";
      std::cerr << "complete simulation data already stored in this folder\n";
      std::cerr << "If you really want to overwrite the simulation files\n";
      std::cerr << "remove the " << status_file_name << " file\n";
      exit(-1);
    }
  }
  else if (s == "done") {
    std::cerr << "Error: ";
    std::cerr << "complete simulation data already stored in this folder\n";
    std::cerr << "If you really want to overwrite the simulation files\n";
    std::cerr << "remove the " << status_file_name << " file\n";
    exit(-1);
  }
  
  else {
    std::cerr << "Error reading status file\n";
    exit(-1);
  }
  
  std::cout << "Done loading simulation status.\n" << std::flush;
  return 0;
}


int simulation::saveStatus(std::string s, int j)
{
  std::cout << "Saving simulation status...\n" << std::flush;
  
  if (s=="train" && (j*train_block_size>=T)) {
    s = "test";
    j = 0;
  }
  std::ofstream ofs;
  ofs.open(status_file_name, std::ios::out);
  if (ofs.fail()) {
    std::cerr << "Error writing status file\n";
    exit(-1);
  }
  
  ofs << s << " " << j << " end\n";
  ofs.close();
  
  std::cout << "Done saving simulation status.\n" << std::flush;

  return 0;
}
		 
int simulation::saveNetwork()
{
  std::cout << "Saving network on a file...\n" << std::flush;

  std::ofstream ofs;
  ofs.open(network_file_name, std::ios::out | std::ios::binary);
  if (ofs.fail()) {
    std::cerr << "Error writing network file\n";
    exit(-1);
  }
  uint N2_tmp = conn_index.size();
  ofs.write(reinterpret_cast<const char*>(&N2_tmp), sizeof(int));
  for (uint i2=0; i2<N2_tmp; i2++) {
    if (i2%10000==0) {
      std::cout << "Saving network " << i2 << " / " << N2_tmp << "\n";
    }
    uint nc = n_conn_2[i2];
    ofs.write(reinterpret_cast<const char*>(&nc), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&conn_index[i2][0]),
	      nc*sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&w[i2][0]),
	      nc*sizeof(double));
  }
  ofs.close();

  std::cout << "Done saving network on a file.\n" << std::flush;

  return 0;
}

int simulation::allocateNetwork()
{
  std::cout << "Allocating arrays for network...\n" << std::flush;
  
  int iC = (int)round(C);
  if (connection_rule==FIXED_INDEGREE) {
    iC_reserve = iC;
  }
  else {
    iC_reserve = (int)round(C + 10.0*sqrt(C));
  }
  //conn_index.clear();
  //w.clear();
  conn_index.resize(N2);
  w.resize(N2);
  n_conn_2.resize(N2);
  for (int i2=0; i2<N2; i2++) {
    //conn_index[i2].reserve(iC_reserve);
    //w[i2].reserve(iC_reserve);
    conn_index[i2].resize(iC_reserve);
    w[i2].resize(iC_reserve);
    n_conn_2[i2] = 0;
  }
  std::cout << "\tDone.\n" << std::flush;

  return 0;
}

int simulation::createNetwork()
{
  std::cout << "Creating network...\n" << std::flush;
  
  rnd_gen_network.seed(master_seed + seed_offset_network + seed_offset);
  int iC = (int)round(C);
  
  // uniform distribution for connectivity purpose
  std::uniform_int_distribution<> rnd_int1(0, N1-1);
  std::uniform_int_distribution<> rnd_int2(0, N2-1);
  if (connection_rule==FIXED_TOTAL_NUMBER
      || connection_rule==POISSON_TOTAL_NUMBER) {
    std::cout << "\tUsing fixed_total_number or poisson_total_number rule...\n"
	      << std::flush;
    int total_number;
    if (connection_rule==FIXED_TOTAL_NUMBER) {
      total_number = (int)round(C*N2);
    }
    else {
      std::poisson_distribution<> rnd_poiss_total_num(C*N2);
      total_number = rnd_poiss_total_num(rnd_gen_network);
    }
    for (int ic=0; ic<total_number; ic++) {
      if (ic%10000000 == 0) {
	std::cout << "\tCreate network " << ic << " / " << total_number << "\n" << std::flush;
      }
      int i2;
      int ic1;
      do {
	i2 = rnd_int2(rnd_gen_network);
	ic1 = n_conn_2[i2];
      } while (ic1 >= iC_reserve);
      int i1 = rnd_int1(rnd_gen_network);
      //conn_index[i2].push_back(i1);
      //w[i2].push_back(W0);
      conn_index[i2][ic1] = i1;
      w[i2][ic1] = W0;
      n_conn_2[i2]++;
    }
    std::cout << "\tDone.\n" << std::flush;
  }
  else if (connection_rule==FIXED_INDEGREE
	   || connection_rule==POISSON_INDEGREE) {
    std::cout << "\tUsing fixed_indegree or poisson_indegree rule...\n"
	      << std::flush;
    std::vector<int> int_range;
    if (!allow_multapses) { 
      for (int i=0; i<N1; i++) {
	int_range.push_back(i);
      }
    }
    
    for (int i2=0; i2<N2; i2++) {
      if (i2%10000 == 0) {
	std::cout << "\tCreate network " << i2 << " / " << N2 << "\n";
      }
      
      if (allow_multapses) {
	int iC1;
	if (connection_rule == FIXED_INDEGREE) {
	  iC1 = iC;
	}
	else {
	  std::poisson_distribution<> rnd_poiss(C);
	  do {
	    iC1 = rnd_poiss(rnd_gen_network);
	  } while (iC1 >= iC_reserve);
	}
	n_conn_2[i2] = iC1;
	for (int ic=0; ic<iC1; ic++) {
	  //conn_index[i2].push_back(rnd_int1(rnd_gen_network));
	  //w[i2].push_back(W0);
	  conn_index[i2][ic] = rnd_int1(rnd_gen_network);
	  w[i2][ic] = W0;
	}
      }
      else {
	n_conn_2[i2] = iC;
	for (int ic=0; ic<iC; ic++) {
	  std::uniform_int_distribution<> rnd_j1(ic, N1-1);
	  int j1 = rnd_j1(rnd_gen_network);
	  std::swap(int_range[ic], int_range[j1]);
	  //conn_index[i2].push_back(int_range[ic]);
	  //w[i2].push_back(W0);
	  conn_index[i2][ic] = int_range[ic];
	  w[i2][ic] = W0;
	}
      }
    }
    std::cout << "\tDone.\n" << std::flush;
  }
  else {
    std::cerr << "Unknown connection rule\n";
    exit(-1);
  }
  std::cout << "Done creating network.\n" << std::flush;
  
  return 0;
}  
  

int simulation::loadNetwork()
{
  std::cout << "Loading network from file...\n" << std::flush;

  std::ifstream ifs;
  ifs.open(network_file_name, std::ios::in | std::ios::binary);
  if (ifs.fail()) {
    std::cerr << "Error reading network file\n";
    exit(-1);
  }
  int N2_tmp;
  ifs.read(reinterpret_cast<char*>(&N2_tmp), sizeof(int));
  if (N2_tmp != N2) {
    std::cerr << "Inconsistent value if N2 in network file\n";
    exit(-1);
  }
  
  for (int i2=0; i2<N2; i2++) {
    if (i2%10000 == 0) {
      std::cout << "Load network " << i2 << " / " << N2 << "\n";
    }
    uint nc;
    ifs.read(reinterpret_cast<char*>(&nc), sizeof(int));
    n_conn_2[i2] = nc;
    //std::vector<int> ci(nc);
    //std::vector<double> wi2(nc);
    ifs.read(reinterpret_cast<char*>(&conn_index[i2][0]),
	     nc*sizeof(int));
    ifs.read(reinterpret_cast<char*>(&w[i2][0]),
	     nc*sizeof(double));
    //conn_index.push_back(ci);
    //w.push_back(wi2);
  }
  ifs.close();

  std::cout << "Done loading network from file.\n" << std::flush;

  return 0;
}

int simulation::rewireConnections()
{
  std::cout << "Rewiring connections...\n" << std::flush;

  // uniform distribution for connectivity purpose
  std::uniform_int_distribution<> rnd_int1(0, N1-1);
  std::uniform_int_distribution<> rnd_int2(0, N2-1);

  if (connection_rule==FIXED_INDEGREE) {
    // uniform distribution for connectivity purpose
    std::uniform_int_distribution<> rnd_int(0, N1-1);

    // for fixed_indegree connection rule
    // move (i.e. destroy and recreate) non-consolidated connections
    for (int i2=0; i2<N2; i2++) {
      for (int ic=0; ic<n_conn_2[i2]; ic++) {
	if (w[i2][ic] < W0 + eps) { // non-consolidated connection
	  conn_index[i2][ic] = rnd_int(rnd_gen_train);
	}
      }
    }
  }
  // for other rules just destroy and recreate non-consolidated connections
  else {
    // first identify consolidated connections and reindex them
    // to the beginning of the connection vector
    int consolidated_num = 0;
    for (int i2=0; i2<N2; i2++) {
      int k = 0;
      for (int ic=0; ic<n_conn_2[i2]; ic++) {
	if (w[i2][ic] > Wc - eps) { // consolidated connection
	  conn_index[i2][k] = conn_index[i2][ic];
	  w[i2][k] = w[i2][ic];
	  k++;
	  consolidated_num++;
	}
      }
      n_conn_2[i2] = k;
      //conn_index[i2].resize(k);
      //w[i2].resize(k);
    }
    if (connection_rule==POISSON_INDEGREE) {
      // uniform and poisson  distributions for connectivity purpose
      std::uniform_int_distribution<> rnd_int(0, N1-1);
      std::poisson_distribution<> rnd_poiss(C);
      int iC1;
      for (int i2=0; i2<N2; i2++) {
	int k = n_conn_2[i2];
	do {
	  iC1 = rnd_poiss(rnd_gen_train);
	} while (iC1 >= iC_reserve || iC1 < k);
	n_conn_2[i2] = iC1;
	for (int ic=k; ic<n_conn_2[i2]; ic++) {
	  conn_index[i2][ic] = rnd_int(rnd_gen_train);
	  w[i2][ic] = W0;
	}
      }
    }
    else {
      int total_number;
      if (connection_rule==FIXED_TOTAL_NUMBER) {
	total_number = (int)round(C*N2);
      }
      else {
	std::poisson_distribution<> rnd_poiss_total_num(C*N2);
	total_number = rnd_poiss_total_num(rnd_gen_train);
      }
      if (total_number<=consolidated_num) {
	return 0;
      }
      for (int ic=0; ic<total_number-consolidated_num; ic++) {
	//if (ic%10000000 == 0) {
	//std::cout << "Rewire connections " << ic << " / "
	//	  << total_number-consolidated_num << "\n";
	//}
	int i2;
	int ic1;
	do {
	  i2 = rnd_int2(rnd_gen_train);
	  ic1 = n_conn_2[i2];
	} while (ic1 >= iC_reserve);
	int i1 = rnd_int1(rnd_gen_train);
	//conn_index[i2].push_back(i1);
	//w[i2].push_back(W0);
	conn_index[i2][ic1] = i1;
	w[i2][ic1] = W0;
	n_conn_2[i2]++;
	//int i1 = rnd_int1(rnd_gen_train);
	//int i2 = rnd_int2(rnd_gen_train);
	//conn_index[i2].push_back(i1);
	//w[i2].push_back(W0);
      }
    }
  }
  
  std::cout << "Done rewiring connections.\n" << std::flush;

  return 0;
}

int simulation::extractTestPatternIndexes()
{
  std::cout << "Extracting test pattern indexes...\n" << std::flush;

  // must be created for the whole training set indexes, not just the test
  ie_test_arr.resize(T);
  for (int ie=0; ie<T; ie++) {
    ie_test_arr[ie] = ie;
  }
  if (random_test_order) {
    std::mt19937 rnd_gen;
    rnd_gen.seed(master_seed + seed_offset_test + seed_offset);
    for (int ie=0; ie<T-1; ie++) {
      std::uniform_int_distribution<> rnd_ie(ie, T-1);
      int ie1 = rnd_ie(rnd_gen);
      std::swap(ie_test_arr[ie], ie_test_arr[ie1]);
    }
  }
  std::cout << "Done.\n" << std::flush;
  
  return 0;
}
    