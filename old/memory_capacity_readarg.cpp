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

float my_erfinvf (float a);

double erfm1(double x)
{
  return sqrt(2.0)*my_erfinvf((float)(2.0*x - 1.0));
}

namespace sim_params
{
  // use discrete rates or lognormal distribution
  bool lognormal_rate = false;
  // multapses can be allowed or forbidden
  bool allow_multapses = true;
  // step (in n. of examples) for connection recombination (0: no recombination)
  int change_conn_step = 20;
  
  // number of training examples
  int T = 1000;
  // connections per layer-2-neuron, i.e. indegree (double needed)
  double C = 10000.0;
  // probability of high rate for layer 1
  double p1 = 1.0e-3;
  // probability of high rate for layer 2
  double p2 = 1.0e-3;
  // number of neurons in pop 1
  int N1 = 100000;
  // number of neurons in pop 2
  int N2 = 100000;  
  // baseline weight
  double W0 = 0.1;
  // consolidated weight
  double Wc = 1.0;
  // low rate for layer 1 [Hz]
  double rl1 = 2.0;
  // high rate for layer 1 [Hz]
  double rh1 = 50.0;
  // low rate for layer 2 [Hz]
  double rl2 = 2.0;
  // high rate for layer 2 [Hz]
  double rh2 = 50.0;
  uint_least32_t master_seed = 123456;
  bool train_flag = true;
  bool load_network = false;
  bool save_network = false;

  int train_block_size = 10000;
  int test_block_size = 10000;
  std::mt19937 rnd_gen;
  char network_file_name[] = "network_xxxx.dat";
  char status_file_name[] = "status_xxxx.dat";  
  
  int readParams(char *filename);
}

int generateRandomTrainingSet(double **rate_L1_train, double **rate_L2_train,
			      double mu_ln1, double sigma_ln1,
			      double mu_ln2, double sigma_ln2);

int loadStatus(int &ie0_train, int &ie1_train, int &j_train,
	       int &ie0_test, int &ie1_test, int &j_test,
	       int n_test, bool &train_flag);

int saveStatus(std::string s, int j);

int saveNetwork(std::vector<std::vector<int> > &conn_index,
		std::vector<std::vector<double> > &w);

int loadNetwork(std::vector<std::vector<int> > &conn_index,
		std::vector<std::vector<double> > &w);

int createNetwork(std::vector<std::vector<int> > &conn_index,
		  std::vector<std::vector<double> > &w);

int main(int argc, char *argv[])
{
  using namespace sim_params;

  int seed_offset = 0;
  
  if (argc==2 || argc==3) {
    sscanf(argv[1], "%d", &seed_offset);
  }
  else {
    printf("Usage: %s rnd_seed_offset [parameter_file]\n", argv[0]);
    exit(0);
  }
  if (argc==3) {
    readParams(argv[2]);
  }
  if (!allow_multapses && change_conn_step!=0) {
    std::cerr << "Connection recombination allowed only "
      "with multapses allowed\n";
    exit(-1);
  }
  
  // seed RNGs
  rnd_gen.seed(master_seed + seed_offset);
  
  // output file
  FILE *fp_out;
  // header file
  FILE *fp_head;

  char file_name_head[] = "mem_head_xxxx.dat";
  sprintf(file_name_head, "mem_head_%04d.dat", seed_offset);
  fp_head = fopen(file_name_head, "wt");


  sprintf(network_file_name, "network_%04d.dat", seed_offset);

  sprintf(status_file_name, "status_%04d.dat", seed_offset);

  // epsilon (for equivalence condition)
  double eps = 1.0e-6;

  // probability of having consolidated the
  // connection at least for one instance
  double p = 1.0 - pow(1.0 - p1*p2, T);

  double q1 = 1.0 - p1;
  // average rate layer 1
  double rm1 = p1*rh1 + q1*rl1;

  double q2 = 1.0 - p2;
  // average rate layer 2
  double rm2 = p2*rh2 + q2*rl2;

  // rate lognormal distribution parameters for layer 1
  double sigma_ln1 = erfm1(q1) - erfm1(q1*rl1/rm1);
  double mu_ln1 = log(rm1) - sigma_ln1*sigma_ln1/2.0;
  double yt_ln1 = erfm1(q1)*sigma_ln1 + mu_ln1;
  double rt1;
  if(lognormal_rate) {
    rt1 = exp(yt_ln1);
  }
  else {
    rt1 = (rh1 + rl1) / 2.0;
  }

  // rate lognormal distribution parameters for layer 2
  double sigma_ln2 = erfm1(q2) - erfm1(q2*rl2/rm2);
  double mu_ln2 = log(rm2) - sigma_ln2*sigma_ln2/2.0;
  double yt_ln2 = erfm1(q2)*sigma_ln2 + mu_ln2;
  double rt2;
  if(lognormal_rate) {
    rt2 = exp(yt_ln2);
  }
  else {
    rt2 = (rh2 + rl2) / 2.0;
  }
  
  // average consolidated connections
  double k = p*C;
  // <r^2> for layer 1 (we do not need it for layer 2)
  double rsq1 = p1*rh1*rh1 + (1.0 - p1)*rl1*rl1;
  // rate variance layer 1
  double var_r1;
  if (lognormal_rate) {
    var_r1 = (exp(sigma_ln1*sigma_ln1) -1.0)
      * exp(2.0*mu_ln1 + sigma_ln1*sigma_ln1);
  }
  else {
    var_r1 = rsq1 - rm1*rm1;
  }
  // calculation for variance of k
  double k2 = C*(C - 1)*pow(1.0 - (2.0 - p1)*p1*p2, T)
    - C*(2*C - 1)*pow(1.0 - p1*p2, T) + C*C;
  double var_k = k2 - k*k;
  
  // theoretical estimation of Sb, S2 and sigma^2 Sb
  double Sbt = Wc*k*rm1 + W0*(C-k)*rm1;
  double S2t = rh1*Wc*p1*C + rl1*(1.0-p1)*(W0*C + (Wc - W0)*k);
  double S2t_chc = rh1*Wc*p1*C + rm1*(1.0-p1)*(W0*C + (Wc - W0)*k);
  double var_St = (Wc*Wc*k + W0*W0*(C-k))*var_r1
    + (Wc - W0)*(Wc - W0)*rm1*rm1*var_k;

  // print of theoretical estimations
  printf("p: %.9lf\n", p);
  printf("sigma2r layer 1 (theoretical): %.4lf\n", var_r1);
  printf("sigma2k (theoretical): %.4lf\n", var_k);
  printf("S2 (theoretical): %.4lf\n", S2t);
  printf("S2 with connection recombination (theoretical): %.4lf\n", S2t_chc);
  printf("Sb (theoretical):  %.4lf\n", Sbt);
  printf("sigma2S (theoretical): %.4lf\n", var_St);
  //std::cout << (Wc*Wc*k + W0*W0*(C-k))*sigma2r << "\n";
  //std::cout << (Wc - W0)*(Wc - W0)*r*r*sigma2k << "\n";

  // same but saved in the header file
  fprintf(fp_head, "p: %.9lf\n", p);
  fprintf(fp_head, "sigma2r layer 1 (theoretical): %.4lf\n", var_r1);
  fprintf(fp_head, "sigma2k (theoretical): %.4lf\n", var_k);
  fprintf(fp_head, "S2 (theoretical): %.4lf\n", S2t);
  fprintf(fp_head, "S2 with connection recombination (theoretical): %.4lf\n",
	  S2t_chc);
  fprintf(fp_head, "Sb (theoretical):  %.4lf\n", Sbt);
  fprintf(fp_head, "sigma2S (theoretical): %.4lf\n", var_St);
  fclose(fp_head);
  
  // Connection index vector
  // conn_index[i2][ic] = i1 = index of the neuron of pop 1 connected
  // to i2 through connection ic of neuron i2 of pop 2
  std::vector<std::vector<int> > conn_index;

  // Connection weight vector
  // w[i2][ic] = weight of the connection ic of neuron i2
  std::vector<std::vector<double> > w;
  
  double **rate_L1_train = new double*[T];
  double **rate_L2_train = new double*[T];
  for (int ie=0; ie<T; ie++) {
    // for each example I allocate the rate of pop 1 and pop 2 neurons
    rate_L1_train[ie] = new double[N1];
    rate_L2_train[ie] = new double[N2];
  }
  generateRandomTrainingSet(rate_L1_train, rate_L2_train, mu_ln1, sigma_ln1,
			    mu_ln2, sigma_ln2);
  
  // training set and test set are the same for now
  int n_test = T;
  double **rate_L1_test = rate_L1_train;
  double **rate_L2_test = rate_L2_train;

  int ie0_train = 0;
  int ie1_train = T;
  int j_train = 0;
  int ie0_test = 0;
  int ie1_test = n_test;
  int j_test = 0;

  if (T>train_block_size || n_test>test_block_size) {
    loadStatus(ie0_train, ie1_train, j_train, ie0_test, ie1_test, j_test,
	       n_test, train_flag);
  }
  if (load_network) {
    loadNetwork(conn_index, w);
  }
  else {
    createNetwork(conn_index, w);
  }
  
  // Training phase
  if (train_flag) {
    // loop over the T examples
    for (int ie=ie0_train; ie<ie1_train; ie++) {
      //std::cout << "Training example n. " << ie + 1 << " / " << T << "\n";
      for (int i2=0; i2<N2; i2++) {
	if (rate_L2_train[ie][i2] > rt2) {
	  for (uint ic=0; ic<conn_index[i2].size(); ic++) {
	    int i1 = conn_index[i2][ic];
	    if (rate_L1_train[ie][i1] > rt1) {
	      // synaptic consolidation
	      w[i2][ic] = Wc;
	    }
	  }
	}
      }
    
      if (change_conn_step!=0 && ((ie+1)%change_conn_step==0)
	  && allow_multapses) {
	std::cout << "Training example n. " << ie + 1 << " / " << T << "\n";
	// uniform distribution for connectivity purpose
	std::uniform_int_distribution<> rnd_int(0, N1-1);

	// move (i.e. destroy and recreate) non-consolidated connections
	for (int i2=0; i2<N2; i2++) {
	  for (uint ic=0; ic<conn_index[i2].size(); ic++) {
	    if (w[i2][ic] < W0 + eps) { // non-consolidated connection
	      conn_index[i2][ic] = rnd_int(rnd_gen);
	    }
	  }
	}
      }
    }
    if (save_network) {
      saveNetwork(conn_index, w);
    }
    if (T>train_block_size || n_test>test_block_size) {
      saveStatus("train", j_train+1);
      return 0;
    }
  }

  // Test phase
  if (T>train_block_size || n_test>test_block_size) {
    char file_name_out[] = "mem_out_xxxx_yyyy.dat";
    sprintf(file_name_out, "mem_out_%04d_%04d.dat", seed_offset, j_test);
    fp_out = fopen(file_name_out, "wt");
  }
  else {
    char file_name_out[] = "mem_out_xxxx.dat";
    sprintf(file_name_out, "mem_out_%04d.dat", seed_offset);
    fp_out = fopen(file_name_out, "wt");
  }

  printf("Simulated\n");
  printf("ie\tSb\tS2\tsigma2S\n");
  
  // test over the examples previously shown
  for (int ie = ie0_test; ie<ie1_test; ie++) {
    double S2_sum = 0.0;
    // for mean calculation - counter of neurons at rh in pop 2
    int P2 = 0;
    double Sb_sum = 0.0;
    double Sb_square_sum = 0.0;
    
    // loop over pop 2 neurons
    for (int i2=0; i2<N2; i2++) {
      // signal of neurons representing the item
      double S2 = 0.0;
      // signal of neurons not representing the item (i.e. background)
      double Sb = 0.0;
      // consider neurons of pop 2 at high rate (S2)
      if (rate_L2_test[ie][i2] > rt2) {
      	P2++;
        // loop for each connection of neuron of pop 2
        for (uint ic=0; ic<conn_index[i2].size(); ic++) {
          // look at the neuron of pop 1 connected through the connection
          int i1 = conn_index[i2][ic];
          // compute its contribution to input signal, i.e. r*W
          double s = rate_L1_test[ie][i1]*w[i2][ic];
          // add the contribution to input signal
          S2 += s;
        }
      }
      // consider neurons of pop 2 at low rate (Sb)
      else {
        for (uint ic=0; ic<conn_index[i2].size(); ic++) {
          int i1 = conn_index[i2][ic];
          // compute contribution
          double s = rate_L1_test[ie][i1]*w[i2][ic];
          // add it to background signal
          Sb += s;
        }
      }
      // sum of input signal over the N2 neurons of pop 2
      Sb_sum += Sb;
      S2_sum += S2;
      // needed for computing the variance of the bkg
      Sb_square_sum += Sb*Sb;
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
    saveStatus("test", j_test+1);
    if ((j_test+1)*test_block_size>=n_test && save_network==false) {
      int res = remove(network_file_name);
      if (res != 0) {
	std::cerr << "Cannot remove network file.\n";
      }
    } 
  }

  fclose(fp_out);
  
  return 0;
}


namespace sim_params {
  int readParams(char *filename)
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
      if (s=="lognormal_rate") {
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
}  



int generateRandomTrainingSet(double **rate_L1_train, double **rate_L2_train,
			      double mu_ln1, double sigma_ln1,
			      double mu_ln2, double sigma_ln2) 
{
  using namespace sim_params;

  // probability distribution generators
  std::uniform_real_distribution<> rnd_uniform(0.0, 1.0);
  std::normal_distribution<double> rnd_normal1(mu_ln1, sigma_ln1);
  std::normal_distribution<double> rnd_normal2(mu_ln2, sigma_ln2);
  
  for (int ie=0; ie<T; ie++) {
    //std::cout << "Training example n. " << ie + 1 << " / " << T << "\n";
    // extract rate for pop 1 neurons
    for (int i1=0; i1<N1; i1++) {
      if (lognormal_rate) {
	rate_L1_train[ie][i1] = exp(rnd_normal1(rnd_gen));
      }
      else {
	if (rnd_uniform(rnd_gen) < p1) {
	  rate_L1_train[ie][i1] = rh1;
	}
	else {
	  rate_L1_train[ie][i1] = rl1;
	}
      }
    }
    
    // extract rate and select between rh and rl for pop 2 neurons
    for (int i2=0; i2<N2; i2++) {
      if (lognormal_rate) {
	rate_L2_train[ie][i2] =  exp(rnd_normal2(rnd_gen));
      }
      else {
	if (rnd_uniform(rnd_gen) < p2) {
	  rate_L2_train[ie][i2] = rh2;
	}
	else {
	  rate_L2_train[ie][i2] = rl2;
	}
      }
    }
  }

  return 0;
}

int loadStatus(int &ie0_train, int &ie1_train, int &j_train,
	       int &ie0_test, int &ie1_test, int &j_test,
	       int n_test, bool &train_flag)
{
  using namespace sim_params;

  std::ifstream ifs;
  ifs.open(status_file_name, std::ios::in);
  if (ifs.fail()) {
    ie1_train = std::min(train_block_size, T);
    save_network = true;
    return 0;
  }

  load_network = true;
  
  std::string s;
  ifs >> s;
  if (s == "train") {
    save_network = true;
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
  else {
    std::cerr << "Error reading status file\n";
    exit(-1);
  }
  
  return 0;
}


int saveStatus(std::string s, int j)
{
  using namespace sim_params;
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

  return 0;
}
		 
int saveNetwork(std::vector<std::vector<int> > &conn_index,
		std::vector<std::vector<double> > &w)
{
  using namespace sim_params;
  
  std::ofstream ofs;
  ofs.open(network_file_name, std::ios::out | std::ios::binary);
  if (ofs.fail()) {
    std::cerr << "Error writing network file\n";
    exit(-1);
  }
  uint N2 = conn_index.size();
  ofs.write(reinterpret_cast<const char*>(&N2), sizeof(int));
  for (uint i2=0; i2<N2; i2++) {
    if (i2%10000==0) {
      std::cout << "Saving network " << i2 << " / " << N2 << "\n";
    }
    uint nc = conn_index[i2].size();
    ofs.write(reinterpret_cast<const char*>(&nc), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&conn_index[i2][0]),
	      nc*sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&w[i2][0]),
	      nc*sizeof(double));
  }
  ofs.close();

  return 0;
}

int createNetwork(std::vector<std::vector<int> > &conn_index,
		  std::vector<std::vector<double> > &w)
{
  using namespace sim_params;
  int iC = (int)round(C);
  // uniform distribution for connectivity purpose
  std::uniform_int_distribution<> rnd_int(0, N1-1);

  std::vector<int> int_range;
  if (!allow_multapses) { 
    for (int i=0; i<N1; i++) {
      int_range.push_back(i);
    }
  }
  
  for (int i2=0; i2<N2; i2++) {
    if (i2%10000 == 0) {
      std::cout << "Create network " << i2 << " / " << N2 << "\n";
    }
    std::vector<int> ci;
    std::vector<double> wi2;
    
    if (allow_multapses) {
      for (int ic=0; ic<iC; ic++) {
	ci.push_back(rnd_int(rnd_gen));
	wi2.push_back(W0);
      }
    }
    else {
      for (int ic=0; ic<iC; ic++) {
	std::uniform_int_distribution<> rnd_j1(ic, N1-1);
	int j1 = rnd_j1(rnd_gen);
	std::swap(int_range[ic], int_range[j1]);
	ci.push_back(int_range[ic]);
	wi2.push_back(W0);
      }
    }
    conn_index.push_back(ci);
    w.push_back(wi2);
  }


  return 0;
}

int loadNetwork(std::vector<std::vector<int> > &conn_index,
		std::vector<std::vector<double> > &w)
{
  using namespace sim_params;

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
    std::vector<int> ci(nc);
    std::vector<double> wi2(nc);
    ifs.read(reinterpret_cast<char*>(&ci[0]),
	     nc*sizeof(int));
    ifs.read(reinterpret_cast<char*>(&wi2[0]),
	     nc*sizeof(double));
    conn_index.push_back(ci);
    w.push_back(wi2);
  }
  ifs.close();

  return 0;
}
