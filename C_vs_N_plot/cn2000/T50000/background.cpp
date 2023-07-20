#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <random>

int main(int argc, char *argv[])
{
  uint_least32_t master_seed = 123456;
  int seed_offset = 0;
  if (argc==2) {
    sscanf(argv[1], "%d", &seed_offset);
  }
  else {
    printf("Usage: %s rnd_seed_offset\n", argv[0]);
    exit(0);
  }

  // generate RNGs
  std::mt19937 rnd_gen(master_seed + seed_offset);
  std::uniform_real_distribution<> rnd_uniform(0.0, 1.0);

  // output file
  FILE *fp_out;
  // header file
  FILE *fp_head;
  char file_name_out[] = "bkg_out_xxxx.dat";
  char file_name_head[] = "bkg_head_xxxx.dat";
  sprintf(file_name_out, "bkg_out_%04d.dat", seed_offset);
  sprintf(file_name_head, "bkg_head_%04d.dat", seed_offset);
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
  double k = p*C;
  double r2 = p1*rh*rh + (1.0 - p1)*rl*rl;
  double sigma2r = r2 - r*r;
  double k2 = C*(C - 1)*pow(1.0 - (2.0 - p1)*p1*p2, T)
    - C*(2*C - 1)*pow(1.0 - p1*p2, T) + C*C;
  double sigma2k = k2 - k*k;
  
  double Sbt = Wc*k*r + W0*(C-k)*r;
  double S2t = rh*Wc*p1*C + rl*(1.0-p1)*(W0*C + (Wc - W0)*k);
  double sigma2St = (Wc*Wc*k + W0*W0*(C-k))*sigma2r
    + (Wc - W0)*(Wc - W0)*r*r*sigma2k;

  // print of theoretical estimations
  printf("p: %.9lf\n", p);
  printf("sigma2r (theoretical): %.4lf\n", sigma2r);
  printf("sigma2k (theoretical): %.4lf\n", sigma2k);
  printf("S2 (theoretical): %.4lf\n", S2t);
  printf("Sb (theoretical):  %.4lf\n", Sbt);
  printf("sigma2S (theoretical): %.4lf\n", sigma2St);
  //std::cout << (Wc*Wc*k + W0*W0*(C-k))*sigma2r << "\n";
  //std::cout << (Wc - W0)*(Wc - W0)*r*r*sigma2k << "\n";

  // same but saved in the header file
  fprintf(fp_head, "p: %.9lf\n", p);
  fprintf(fp_head, "sigma2r (theoretical): %.4lf\n", sigma2r);
  fprintf(fp_head, "sigma2k (theoretical): %.4lf\n", sigma2k);
  fprintf(fp_head, "S2 (theoretical): %.4lf\n", S2t);
  fprintf(fp_head, "Sb (theoretical):  %.4lf\n", Sbt);
  fprintf(fp_head, "sigma2S (theoretical): %.4lf\n", sigma2St);
  
  double w[iC];
  double **rate_L1 = new double*[T];
  //double rate_L2[T];
  double *rate_L1_test = new double[iC];

  for (int ie=0; ie<T; ie++) {
    rate_L1[ie] = new double[iC];
  }

  int N = 100000;
  printf("Simulated\n");
  printf("i\tSb\n");

  fprintf(fp_head, "Simulated\n");
  fprintf(fp_head, "i\tSb\n");
  fclose(fp_head);
  
  for (int i=0; i<N; i++) {
    for (int i1=0; i1<iC; i1++) {
      w[i1] = W0;
    }

    for (int ie=0; ie<T; ie++) {
      for (int i1=0; i1<iC; i1++) {
	if (rnd_uniform(rnd_gen) < p1) {
	  rate_L1[ie][i1] = rh;
	}
	else {
	  rate_L1[ie][i1] = rl;
	}
      }
      
      if (rnd_uniform(rnd_gen) < p2) {
	//rate_L2[ie] = rh;
	for (int i1=0; i1<iC; i1++) {
	  if (rate_L1[ie][i1] > rh - eps) {
	    w[i1] = Wc;
	  }
	}
      }
      //else {
      //	rate_L2[ie] = rl;
      //}
    }
    
    for (int i1=0; i1<iC; i1++) {
      if (rnd_uniform(rnd_gen) < p1) {
	rate_L1_test[i1] = rh;
      }
      else {
	rate_L1_test[i1] = rl;
      }
    }
    
    //double Sb_sum = 0.0;
    //double Sb_square_sum = 0.0;
    
    double Sb = 0.0;
    for (int i1=0; i1<iC; i1++) {
      double s1 = rate_L1_test[i1]*w[i1];
      Sb += s1;
    }
    printf("%d\t%.1lf\n", i, Sb);
    fprintf(fp_out, "%d\t%.1lf\n", i, Sb);
    fflush(fp_out);
  }

  fclose(fp_out);
  
  return 0;
}
