#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>
#include <vector>
#include "erfinv.cpp"

float my_erfinvf (float a);

double erfm1(double x)
{
  return sqrt(2.0)*my_erfinvf((float)(2.0*x - 1.0));
}

int main(int argc, char *argv[])
{
  uint_least32_t master_seed = 123456;
  int seed_offset = 0;
  bool allow_multapses = true;
  
  if (argc==2 || argc==3) {
    sscanf(argv[1], "%d", &seed_offset);
  }
  else {
    printf("Usage: %s rnd_seed_offset [-nm]\n", argv[0]);
    exit(0);
  }
  if (argc==3) {
    if (strcmp(argv[2], "-nm")==0) {
      allow_multapses = false;
      std::cout << "Multapses disabled.\n";
    }
    else {
      printf("Unrecognized argument. Usage: %s rnd_seed_offset [-nm]\n",
	     argv[0]);
      exit(0);
    }
  }

  // generate RNGs
  std::mt19937 rnd_gen(master_seed + seed_offset);
  std::uniform_real_distribution<> rnd_uniform(0.0, 1.0);
  
  // output file
  FILE *fp_out;
  // header file
  FILE *fp_head;
  char file_name_out[] = "mem_out_xxxx.dat";
  char file_name_head[] = "mem_head_xxxx.dat";
  sprintf(file_name_out, "mem_out_%04d.dat", seed_offset);
  sprintf(file_name_head, "mem_head_%04d.dat", seed_offset);
  fp_out = fopen(file_name_out, "wt");
  fp_head = fopen(file_name_head, "wt");

  // number of tests
  int T = 1000;
  // connections (both int and double needed)
  int iC = 1000;
  double C = iC;
  // probability of high rate
  double alpha1 = 1.0e-3;
  // probability of low rate
  double alpha2 = 1.0e-3;
  // number of neurons in pop 1
  int N1 = 10000;
  // number of neurons in pop 2
  int N2 = 10000;

  // uniform distribution for connectivity purpose
  std::uniform_int_distribution<> rnd_int(0, N1-1);
  
  // baseline weight
  double Wb = 0.1;
  // consolidated weight
  double Ws = 1.0;

  // low rate [Hz]
  double rl = 2.0;
  // high rate [Hz]
  double rh = 50.0;
  // epsilon (for equivalence consition)
  //double eps = 1.0e-6;

  // probability of having consolidated the
  // connection at least for one instance
  double p = 1.0 - pow(1.0 - alpha1*alpha2, T);
  // rate
  double rm1 = alpha1*rh + (1.0 - alpha1)*rl;

  double q1 = 1.0 - alpha1;
  double sigma_ln1 = erfm1(q1) - erfm1(q1*rl/rm1);
  double mu_ln1 = log(rm1) - sigma_ln1*sigma_ln1/2.0;
  double yt_ln1 = erfm1(q1)*sigma_ln1 + mu_ln1;
  double rt1 = exp(yt_ln1);

  std::normal_distribution<double> rnd_normal(mu_ln1, sigma_ln1);
 
  // average consolidated connections
  double k = p*C;
  // <r^2>
  //double r2 = alpha1*rh*rh + (1.0 - alpha1)*rl*rl;
  // rate variance
  //double sigma2r = r2 - rm1*rm1;
  double sigma2r = (exp(sigma_ln1*sigma_ln1) -1.0)
    * exp(2.0*mu_ln1 + sigma_ln1*sigma_ln1);
  // calculation for variance og k
  double k2 = C*(C - 1)*pow(1.0 - (2.0 - alpha1)*alpha1*alpha2, T)
    - C*(2*C - 1)*pow(1.0 - alpha1*alpha2, T) + C*C;
  double sigma2k = k2 - k*k;
  
  // theoretical estimation of Sb, S2 and sigma^2 Sb
  double Sbt = Ws*k*rm1 + Wb*(C-k)*rm1;
  double S2t = rh*Ws*alpha1*C + rl*(1.0-alpha1)*(Wb*C + (Ws - Wb)*k);
  double sigma2St = (Ws*Ws*k + Wb*Wb*(C-k))*sigma2r
    + (Ws - Wb)*(Ws - Wb)*rm1*rm1*sigma2k;

  // print of theoretical estimations
  printf("p: %.9lf\n", p);
  printf("sigma2r (theoretical): %.4lf\n", sigma2r);
  printf("sigma2k (theoretical): %.4lf\n", sigma2k);
  printf("S2 (theoretical): %.4lf\n", S2t);
  printf("Sb (theoretical):  %.4lf\n", Sbt);
  printf("sigma2S (theoretical): %.4lf\n", sigma2St);
  //std::cout << (Ws*Ws*k + Wb*Wb*(C-k))*sigma2r << "\n";
  //std::cout << (Ws - Wb)*(Ws - Wb)*r*r*sigma2k << "\n";

  // same but saved in the header file
  fprintf(fp_head, "p: %.9lf\n", p);
  fprintf(fp_head, "sigma2r (theoretical): %.4lf\n", sigma2r);
  fprintf(fp_head, "sigma2k (theoretical): %.4lf\n", sigma2k);
  fprintf(fp_head, "S2 (theoretical): %.4lf\n", S2t);
  fprintf(fp_head, "Sb (theoretical):  %.4lf\n", Sbt);
  fprintf(fp_head, "sigma2S (theoretical): %.4lf\n", sigma2St);
  
  // Inizialization of connection and weight arrays
  // connection array (N2, iC)
  // conn_index[i2][iC] = i1 = index of the neuron of pop 1 connected
  // to i2 through connection iC of neuron i2 of pop 2
  //int **conn_index = new int*[N2];
  std::vector<std::vector<int> > conn_index;
  std::vector<int> int_range;
  if (!allow_multapses) { 
    for (int i=0; i<N1; i++) {
      int_range.push_back(i);
    }
  }
  
  // w[i2][iC] = weight of the connection iC of neuron i2
  double **w = new double*[N2];
  for (int i2=0; i2<N2; i2++) {
    if (i2%10000 == 0) {
      std::cout << i2 << " / " << N2 << "\n";
    }
    std::vector<int> ci;
    //conn_index[i2] = new int[iC];
    w[i2] = new double[iC];
    if (allow_multapses) {
      for (int ic=0; ic<iC; ic++) {
	ci.push_back(rnd_int(rnd_gen));
	w[i2][ic] = Wb;
      }
    }
    else {
      for (int ic=0; ic<iC; ic++) {
	std::uniform_int_distribution<> rnd_j1(ic, N1-1);
	int j1 = rnd_j1(rnd_gen);
	std::swap(int_range[ic], int_range[j1]);
	ci.push_back(int_range[ic]);
	w[i2][ic] = Wb;
      }
    }
    conn_index.push_back(ci);
  }
 
  // Training phase

  double **rate_L1 = new double*[T];
  double **rate_L2 = new double*[T];

  // loop over the T examples
  for (int ie=0; ie<T; ie++) {
    // for each example I allocate the rate of pop 1 and pop 2 neurons
    rate_L1[ie] = new double[N1];
    rate_L2[ie] = new double[N2];
    // extract rate and select between rh and rl for pop 1 neurons
    for (int i1=0; i1<N1; i1++) {
      rate_L1[ie][i1] = exp(rnd_normal(rnd_gen));
    }
    
    // extract rate and select between rh and rl for pop 2 neurons
    for (int i2=0; i2<N2; i2++) {
      rate_L2[ie][i2] =  exp(rnd_normal(rnd_gen));
      if (rate_L2[ie][i2] > rt1) {
	for (int ic=0; ic<iC; ic++) {
	  int i1 = conn_index[i2][ic];
	  // if r > rt for this neuron of pop 2
	  if (rate_L1[ie][i1] > rt1) {
	    // synaptic consolidation
	    w[i2][ic] = Ws;
	  }
	}
      }
    }
  }

  // Test phase

  printf("Simulated\n");
  printf("ie\tSb\tS2\tsigma2S\n");

  fprintf(fp_head, "Simulated\n");
  fprintf(fp_head, "ie\tSb\tS2\tsigma2S\n");
  fclose(fp_head);
  
  // test over the examples previously shown
  for (int ie = 0; ie<T; ie++) {
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
      if (rate_L2[ie][i2] > rt1) {
      	P2++;
        // loop for each connection of neuron of pop 2
        for (int ic=0; ic<iC; ic++) {
          // look at the neuron of pop 1 connected through the connection
          int i1 = conn_index[i2][ic];
          // compute its contribution to input signal, i.e. r*W
          double s = rate_L1[ie][i1]*w[i2][ic];
          // add the contribution to input signal
          S2 += s;
        }
      }
      // consider neurons of pop 2 at low rate (Sb)
      else {
        for (int ic=0; ic<iC; ic++) {
          int i1 = conn_index[i2][ic];
          // compute contribution
          double s = rate_L1[ie][i1]*w[i2][ic];
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
  
  fclose(fp_out);
  
  return 0;
}
