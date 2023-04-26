#include <iostream>
#include <stdio.h>
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

  std::mt19937 rnd_gen(master_seed + seed_offset);
  std::uniform_real_distribution<> rnd_uniform(0.0, 1.0);
  
  FILE *fp_out;
  FILE *fp_head;
  char file_name_out[] = "mem_out_xxxx.dat";
  char file_name_head[] = "mem_head_xxxx.dat";
  sprintf(file_name_out, "mem_out_%04d.dat", seed_offset);
  sprintf(file_name_head, "mem_head_%04d.dat", seed_offset);
  fp_out = fopen(file_name_out, "wt");
  fp_head = fopen(file_name_head, "wt");
    
  int T = 1000;
  int iC = 10000;
  double C = iC;
  double p1 = 1.0e-3;
  double p2 = 1.0e-3;
  int N1 = 100000;
  int N2 = 100000;

  std::uniform_int_distribution<> rnd_int(0, N1-1);
    
  double W0 = 0.1;
  double Wc = 1.0;

  double rl = 2.0;
  double rh = 50.0;
  double eps = 1.0e-6;

  double p = 1.0 - pow(1.0 - p1*p2, T);
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

  printf("p: %.9lf\n", p);
  printf("sigma2r (theoretical): %.4lf\n", sigma2r);
  printf("sigma2k (theoretical): %.4lf\n", sigma2k);
  printf("S2 (theoretical): %.4lf\n", S2t);
  printf("Sb (theoretical):  %.4lf\n", Sbt);
  printf("sigma2S (theoretical): %.4lf\n", sigma2St);
  //std::cout << (Wc*Wc*k + W0*W0*(C-k))*sigma2r << "\n";
  //std::cout << (Wc - W0)*(Wc - W0)*r*r*sigma2k << "\n";

  fprintf(fp_head, "p: %.9lf\n", p);
  fprintf(fp_head, "sigma2r (theoretical): %.4lf\n", sigma2r);
  fprintf(fp_head, "sigma2k (theoretical): %.4lf\n", sigma2k);
  fprintf(fp_head, "S2 (theoretical): %.4lf\n", S2t);
  fprintf(fp_head, "Sb (theoretical):  %.4lf\n", Sbt);
  fprintf(fp_head, "sigma2S (theoretical): %.4lf\n", sigma2St);
  
  int **conn_index = new int*[N2];
  double **w = new double*[N2];
  for (int i2=0; i2<N2; i2++) {
    conn_index[i2] = new int[iC];
    w[i2] = new double[iC];
    for (int ic=0; ic<iC; ic++) {
      conn_index[i2][ic] = rnd_int(rnd_gen);
      w[i2][ic] = W0;
    }
  }
  
  double **rate_L1 = new double*[T];
  double **rate_L2 = new double*[T];

  for (int ie=0; ie<T; ie++) {
    rate_L1[ie] = new double[N1];
    rate_L2[ie] = new double[N2];
    for (int i1=0; i1<N1; i1++) {
      if (rnd_uniform(rnd_gen) < p1) {
	rate_L1[ie][i1] = rh;
      }
      else {
	rate_L1[ie][i1] = rl;
      }
    }
    
    for (int i2=0; i2<N2; i2++) {
      if (rnd_uniform(rnd_gen) < p2) {
	rate_L2[ie][i2] = rh;
	for (int ic=0; ic<iC; ic++) {
	  int i1 = conn_index[i2][ic];
	  if (rate_L1[ie][i1] > rh - eps) {
	    w[i2][ic] = Wc;
	  }
	}
      }
      else {
	rate_L2[ie][i2] = rl;
      }
    }
  }


  printf("Simulated\n");
  printf("ie\tSb\tS2\tsigma2S\n");

  fprintf(fp_head, "Simulated\n");
  fprintf(fp_head, "ie\tSb\tS2\tsigma2S\n");
  fclose(fp_head);
  
  for (int ie = 0; ie<T; ie++) {
    double S2_sum = 0.0;
    int P2 = 0;
    double Sb_sum = 0.0;
    double Sb_square_sum = 0.0;
    
    for (int i2=0; i2<N2; i2++) {
      double S2 = 0.0;
      double Sb = 0.0;
      if (rate_L2[ie][i2] > rh - eps) {
	P2++;

	for (int ic=0; ic<iC; ic++) {
	  int i1 = conn_index[i2][ic];
	  double s = rate_L1[ie][i1]*w[i2][ic];
	  S2 += s;
	}
      }
      else {
	for (int ic=0; ic<iC; ic++) {
	  int i1 = conn_index[i2][ic];
	  double s = rate_L1[ie][i1]*w[i2][ic];
	  Sb += s;
	}
      }
      Sb_sum += Sb;
      S2_sum += S2;
      Sb_square_sum += Sb*Sb;
    }
    double S2_mean = S2_sum / P2;
    double Sb_mean = Sb_sum / (N2 - P2);
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
