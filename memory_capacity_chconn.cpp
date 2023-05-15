#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>
#include <vector>

int main(int argc, char *argv[])
{
  uint_least32_t master_seed = 123456;
  int seed_offset = 0;
  bool allow_multapses = true;
  bool change_conn = true;
  
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
  int iC = 10000;
  double C = iC;
  // probability of high rate
  double p1 = 1.0e-3;
  // probability of low rate
  double p2 = 1.0e-3;
  // number of neurons in pop 1
  int N1 = 100000;
  // number of neurons in pop 2
  int N2 = 100000;

  // uniform distribution for connectivity purpose
  std::uniform_int_distribution<> rnd_int(0, N1-1);
  
  // baseline weight
  double W0 = 0.1;
  // consolidated weight
  double Wc = 1.0;

  // low rate [Hz]
  double rl = 2.0;
  // high rate [Hz]
  double rh = 50.0;
  // epsilon (for equivalence consition)
  double eps = 1.0e-6;

  // probability of having consolidated the
  // connection at least for one instance
  double p = 1.0 - pow(1.0 - p1*p2, T);
  // rate
  double r = p1*rh + (1.0 - p1)*rl;
  // average consolidated connections
  double k = p*C;
  // <r^2>
  double r2 = p1*rh*rh + (1.0 - p1)*rl*rl;
  // rate variance
  double sigma2r = r2 - r*r;
  // calculation for variance og k
  double k2 = C*(C - 1)*pow(1.0 - (2.0 - p1)*p1*p2, T)
    - C*(2*C - 1)*pow(1.0 - p1*p2, T) + C*C;
  double sigma2k = k2 - k*k;
  
  // theoretical estimation of Sb, S2 and sigma^2 Sb
  double Sbt = Wc*k*r + W0*(C-k)*r;
  double S2t = rh*Wc*p1*C + rl*(1.0-p1)*(W0*C + (Wc - W0)*k);
  double S2t_chc = rh*Wc*p1*C + r*(1.0-p1)*(W0*C + (Wc - W0)*k);
  double sigma2St = (Wc*Wc*k + W0*W0*(C-k))*sigma2r
    + (Wc - W0)*(Wc - W0)*r*r*sigma2k;

  // print of theoretical estimations
  printf("p: %.9lf\n", p);
  printf("sigma2r (theoretical): %.4lf\n", sigma2r);
  printf("sigma2k (theoretical): %.4lf\n", sigma2k);
  printf("S2 (theoretical): %.4lf\n", S2t);
  printf("S2 with connection recombination (theoretical): %.4lf\n", S2t_chc);
  printf("Sb (theoretical):  %.4lf\n", Sbt);
  printf("sigma2S (theoretical): %.4lf\n", sigma2St);
  //std::cout << (Wc*Wc*k + W0*W0*(C-k))*sigma2r << "\n";
  //std::cout << (Wc - W0)*(Wc - W0)*r*r*sigma2k << "\n";

  // same but saved in the header file
  fprintf(fp_head, "p: %.9lf\n", p);
  fprintf(fp_head, "sigma2r (theoretical): %.4lf\n", sigma2r);
  fprintf(fp_head, "sigma2k (theoretical): %.4lf\n", sigma2k);
  fprintf(fp_head, "S2 (theoretical): %.4lf\n", S2t);
  fprintf(fp_head, "S2 with connection recombination (theoretical): %.4lf\n",
	  S2t_chc);
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
	w[i2][ic] = W0;
      }
    }
    else {
      for (int ic=0; ic<iC; ic++) {
	std::uniform_int_distribution<> rnd_j1(ic, N1-1);
	int j1 = rnd_j1(rnd_gen);
	std::swap(int_range[ic], int_range[j1]);
	ci.push_back(int_range[ic]);
	w[i2][ic] = W0;
      }
    }
    conn_index.push_back(ci);
  }
 
  // Training phase

  double **rate_L1 = new double*[T];
  double **rate_L2 = new double*[T];

  // loop over the T examples
  for (int ie=0; ie<T; ie++) {
    //std::cout << "Training example n. " << ie + 1 << " / " << T << "\n";
    // for each example I allocate the rate of pop 1 and pop 2 neurons
    rate_L1[ie] = new double[N1];
    rate_L2[ie] = new double[N2];
    // extract rate and select between rh and rl for pop 1 neurons
    for (int i1=0; i1<N1; i1++) {
      if (rnd_uniform(rnd_gen) < p1) {
	rate_L1[ie][i1] = rh;
      }
      else {
	rate_L1[ie][i1] = rl;
      }
    }
    
    // extract rate and select between rh and rl for pop 2 neurons
    for (int i2=0; i2<N2; i2++) {
      if (rnd_uniform(rnd_gen) < p2) {
	rate_L2[ie][i2] = rh;
	for (int ic=0; ic<iC; ic++) {
	  int i1 = conn_index[i2][ic];
    // if r = rh for this neuron of pop 2
	  if (rate_L1[ie][i1] > rh - eps) {
      // synaptic consolidation
	    w[i2][ic] = Wc;
	  }
	}
      }
      else {
	rate_L2[ie][i2] = rl;
      }
    }
    
    if (((ie+1)%5 == 0) && allow_multapses && change_conn) {
      std::cout << "Training example n. " << ie + 1 << " / " << T << "\n";
      // move (i.e. destroy and recreate) non-consolidated connections
      for (int i2=0; i2<N2; i2++) {
	for (int ic=0; ic<iC; ic++) {
	  if (w[i2][ic] < W0 + eps) { // non-consolidated connection
	    conn_index[i2][ic] = rnd_int(rnd_gen);
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
      if (rate_L2[ie][i2] > rh - eps) {
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
