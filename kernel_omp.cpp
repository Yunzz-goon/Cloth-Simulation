#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "./cloth_code_omp_block.h"
#include "./cloth_param.h"
#include "simple_papi.h"
#include <omp.h>


int main(int argc, char **argv) {
  int i, iter;
  double pe, ke, te;
  // assess input flags

  for (i = 1; i < argc; i += 2) {
    if (argv[i][0] == '-') {
      switch (argv[i][1]) {
      case 'n':
        n = atoi(argv[i + 1]);
        break;
      case 's':
        sep = atof(argv[i + 1]);
        break;
      case 'm':
        mass = atof(argv[i + 1]);
        break;
      case 'f':
        fcon = atof(argv[i + 1]);
        break;
      case 'd':
        delta = atoi(argv[i + 1]);
        break;
      case 'g':
        grav = atof(argv[i + 1]);
        break;
      case 'b':
        rball = atof(argv[i + 1]);
        break;
      case 'o':
        offset = atof(argv[i + 1]);
        break;
      case 't':
        dt = atof(argv[i + 1]);
        break;
      case 'i':
        maxiter = atoi(argv[i + 1]);
        break;
      case 'p':
        p = atoi(argv[i + 1]);
        break;
      default:
        printf(" %s\n"
               "Nodes_per_dimension:             -n int \n"
               "Grid_separation:                 -s float \n"
               "Mass_of_node:                    -m float \n"
               "Force_constant:                  -f float \n"
               "Node_interaction_level:          -d int \n"
               "Gravity:                         -g float \n"
               "Radius_of_ball:                  -b float \n"
               "offset_of_falling_cloth:         -o float \n"
               "timestep:                        -t float \n"
               "num iterations:                  -i int \n",
               argv[0]);
        return -1;
      }
    } else {
      printf(" %s\n"
             "Nodes_per_dimension:             -n int \n"
             "Grid_separation:                 -s float \n"
             "Mass_of_node:                    -m float \n"
             "Force_constant:                  -f float \n"
             "Node_interaction_level:          -d int \n"
             "Gravity:                         -g float \n"
             "Radius_of_ball:                  -b float \n"
             "offset_of_falling_cloth:         -o float \n"
             "timestep:                        -t float \n"
             "num iterations:                  -i int \n",
             argv[0]);
      return -1;
    }
  }

  // print out values to be used in the program
  printf("____________________________________________________\n"
         "_____ COMP3320 Assignment 2 - Cloth Simulation _____\n"
         "____________________________________________________\n"
         "Number of nodes per dimension:  %d\n"
         "Grid separation:                %lf\n"
         "Mass of node:                   %lf\n"
         "Force constant                  %lf\n"
         "Node Interaction Level (delta): %d\n"
         "Gravity:                        %lf\n"
         "Radius of Ball:                 %lf\n"
         "Offset of falling cloth:        %lf\n"
         "Timestep:                       %lf\n",
         n, sep, mass, fcon, delta, grav, rball, offset, dt);

  initMatrix(n, mass, fcon, delta, grav, sep, rball, offset, dt, &x, &y, &z,
             &cpx, &cpy, &cpz, &fx, &fy, &fz, &vx, &vy, &vz, &oldfx, &oldfy,
             &oldfz);

  InitPapi();
  long long StartTime;
  long long StopTime;
  StartTime = PapiStartCounters();
  for (iter = 0; iter < maxiter; iter++) {
    loopcode(n, mass, fcon, delta, grav, sep, rball, xball, yball, zball, dt, x,
             y, z, fx, fy, fz, vx, vy, vz, oldfx, oldfy, oldfz, &pe, &ke, &te, p);
    printf("Iteration %10d PE %10.5f  KE %10.5f  TE %10.5f \n ", iter, pe, ke,
           te);
  }
  StopTime  = PapiStopCounters ();
  PrintPapiResults( "Kernel_xx:", StartTime, StopTime);

  return 0;
}
