#include <iostream>

#include "structural_plasticity.h"

int main(int argc, char *argv[])
{
  simulation sim;
  sim.init(argc, argv);
  sim.run();
  
  return 0;
}
