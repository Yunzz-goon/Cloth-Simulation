#include "./cloth_code_vect_omp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
// #include "simple_papi.h"

void initMatrix(int n, double UNUSED(mass), double UNUSED(fcon),
                int UNUSED(delta), double UNUSED(grav), double sep,
                double rball, double offset, double UNUSED(dt), double **x,
                double **y, double **z, double **cpx, double **cpy,
                double **cpz, double **fx, double **fy, double **fz,
                double **vx, double **vy, double **vz, double **oldfx,
                double **oldfy, double **oldfz) {
  int i, nx, ny;

  // Free any existing
  free(*x);
  free(*y);
  free(*z);
  free(*cpx);
  free(*cpy);
  free(*cpz);

  // allocate arrays to hold locations of nodes
  *x = (double *)malloc(n * n * sizeof(double));
  *y = (double *)malloc(n * n * sizeof(double));
  *z = (double *)malloc(n * n * sizeof(double));
  // This is for opengl stuff
  *cpx = (double *)malloc(n * n * sizeof(double));
  *cpy = (double *)malloc(n * n * sizeof(double));
  *cpz = (double *)malloc(n * n * sizeof(double));

  // initialize coordinates of cloth
  for (nx = 0; nx < n; nx++) {
    for (ny = 0; ny < n; ny++) {
      (*x)[n * nx + ny] = nx * sep - (n - 1) * sep * 0.5 + offset;
      (*z)[n * nx + ny] = rball + 1;
      (*y)[n * nx + ny] = ny * sep - (n - 1) * sep * 0.5 + offset;
      (*cpx)[n * nx + ny] = 0;
      (*cpz)[n * nx + ny] = 1;
      (*cpy)[n * nx + ny] = 0;
    }
  }

  // Throw away existing arrays
  free(*fx);
  free(*fy);
  free(*fz);
  free(*vx);
  free(*vy);
  free(*vz);
  free(*oldfx);
  free(*oldfy);
  free(*oldfz);
  // Alloc new
  *fx = (double *)malloc(n * n * sizeof(double));
  *fy = (double *)malloc(n * n * sizeof(double));
  *fz = (double *)malloc(n * n * sizeof(double));
  *vx = (double *)malloc(n * n * sizeof(double));
  *vy = (double *)malloc(n * n * sizeof(double));
  *vz = (double *)malloc(n * n * sizeof(double));
  *oldfx = (double *)malloc(n * n * sizeof(double));
  *oldfy = (double *)malloc(n * n * sizeof(double));
  *oldfz = (double *)malloc(n * n * sizeof(double));
  for (i = 0; i < n * n; i++) {
    (*vx)[i] = 0.0;
    (*vy)[i] = 0.0;
    (*vz)[i] = 0.0;
    (*fx)[i] = 0.0;
    (*fy)[i] = 0.0;
    (*fz)[i] = 0.0;
  }
}
void loopcode(int n, double mass, double fcon, int delta, double grav,
              double sep, double rball, double xball, double yball,
              double zball, double dt, double *x, double *y, double *z,
              double *fx, double *fy, double *fz, double *vx, double *vy,
              double *vz, double *oldfx, double *oldfy, double *oldfz,
              double *pe, double *ke, double *te) {
  int k;
  double xdiff, ydiff, zdiff, vmag, damp, velDotDiff;

  double update_const = dt * 0.5 / mass;
  double update_v_const;
  // update position as per MD simulation
  #pragma omp simd
  for (k = 0; k < n*n; k++) {
    x[k] += dt * (vx[k] + fx[k] * update_const);
    oldfx[k] = fx[k];
    y[k] += dt * (vy[k] + fy[k] * update_const);
    oldfy[k] = fy[k];
    z[k] += dt * (vz[k] + fz[k] * update_const);
    oldfz[k] = fz[k];

//	apply constraints - push cloth outside of ball
    xdiff = xball - x[k] ;
    ydiff = yball - y[k] ;
    zdiff = zball - z[k] ;
    vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
    if (vmag < rball) {
      // #pragma omp declare simd inbranch
      x[k] = xball - xdiff * rball / vmag;
      y[k] = yball - ydiff * rball / vmag;
      z[k] = zball - zdiff * rball / vmag;
      velDotDiff = vx[k] * xdiff + vy[k] * ydiff + vz[k] * zdiff;
      update_v_const = velDotDiff / (vmag * vmag);
      vx[k] = (vx[k] - xdiff * update_v_const) * 0.1;
      vy[k] = (vy[k] - ydiff * update_v_const) * 0.1;
      vz[k] = (vz[k] - zdiff * update_v_const) * 0.1;
    }
  }


  *pe = eval_pef(n, delta, mass, grav, sep, fcon, x, y, z, fx, fy, fz);

  // Add a damping factor to eventually set velocity to zero
  damp = 0.995;
  *ke = 0.0;
  double tmp_ke;
  tmp_ke = 0.0;
  #pragma omp simd reduction(+:tmp_ke) 
  for (k = 0; k < n*n; k++) {
    vx[k] = (vx[k] +
                      (fx[k] + oldfx[k]) * update_const) *
                    damp;
    vy[k] = (vy[k] +
                      (fy[k] + oldfy[k]) * update_const) *
                    damp;
    vz[k] = (vz[k] +
                      (fz[k] + oldfz[k]) * update_const) *
                    damp;
    tmp_ke += vx[k] * vx[k] + vy[k] * vy[k] +
            vz[k] * vz[k];
  }
  *ke = tmp_ke / 2.0;
  *te = *pe + *ke;
}

double eval_pef(int n, int delta, double mass, double grav, double sep,
                double fcon, double *x, double *y, double *z, double *fx,
                double *fy, double *fz) {
  double pe, rlen, xdiff, ydiff, zdiff, vmag;
  int nx, ny, dx, dy, k;
  double update_f_cons;
  // FILE* fp;
  // fp = fopen("/home/444/yz4016/COMP3320/assessments/comp3320-2022-assignment-2/debugger/debug_opt.txt", "w");
  #pragma omp simd
  for (k = 0; k < n*n; k++) {
    fx[k] = 0.0;
    fy[k] = 0.0;
    fz[k] = -mass * grav;
  }
  pe = 0.0;
  // loop over particles
  double nxdx;
  int nxn, nxny,dxn, dxdy;
  // #pragma omp simd reduction(+:fx[nxny], fy[nxny], fz[nxny], pe)
  for (nx = 0; nx < n; nx++) {
    nxn = nx*n;
    for (ny = 0; ny < n; ny++) {
      nxny = nxn+ny;
      // loop over displacements
      for (dx = nx; dx < MIN(nx + delta + 1, n); dx++){
        nxdx = (nx - dx) * (nx - dx);
        dxn = dx*n;
        #pragma omp simd reduction(+:fx[nxny], fy[nxny], fz[nxny], pe)
        for (dy = MIN(ny + 1, n); dy < MIN(ny + delta + 1, n); dy++)
        {
          dxdy = dxn+dy;
          // compute reference distance
          rlen =
              sqrt(nxdx + (ny - dy) * (ny - dy)) *
              sep;
          // compute actual distance
          xdiff = x[dxdy] - x[nxny];
          ydiff = y[dxdy] - y[nxny];
          zdiff = z[dxdy] - z[nxny];
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          // potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen);
          update_f_cons = fcon  * (vmag - rlen) / vmag;

          fx[nxny] += xdiff * update_f_cons;
          fy[nxny] += ydiff * update_f_cons;
          fz[nxny] += zdiff * update_f_cons;

          fx[dxdy] += -xdiff * update_f_cons;
          fy[dxdy] += -ydiff * update_f_cons;
          fz[dxdy] += -zdiff * update_f_cons;
          // refer to the formula given in the readme.md file instead of referring the lecture notes.
        }
        }
      for (dx = MIN(nx+1, n); dx < MIN(nx + delta + 1, n); dx++) {
        nxdx = (nx - dx) * (nx - dx);
        dxn = dx*n;
        #pragma omp simd reduction(+:fx[nxny], fy[nxny], fz[nxny], pe)
        for (dy = MAX(ny - delta, 0); dy < ny+1; dy++){
          dxdy = dxn+dy;
          // compute reference distance
          rlen =
              sqrt(nxdx  + (ny - dy) * (ny - dy)) *
              sep;
          // compute actual distance
          xdiff = x[dxdy] - x[nxny];
          ydiff = y[dxdy] - y[nxny];
          zdiff = z[dxdy] - z[nxny];
          // fprintf(fp, "%f  \n", vmag);
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          // potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen); 
          update_f_cons = fcon  * (vmag - rlen) / vmag;
          fx[nxny] += xdiff * update_f_cons;
          fy[nxny] += ydiff * update_f_cons;
          fz[nxny] += zdiff * update_f_cons;

          fx[dxdy] += -xdiff * update_f_cons;
          fy[dxdy] += -ydiff * update_f_cons;
          fz[dxdy] += -zdiff * update_f_cons;
          // refer to the formula given in the readme.md file instead of referring the lecture notes.
        }
      }
    }
  }
  // fclose(fp);
  return pe;
}

