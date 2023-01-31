#include "./cloth_code_omp_block.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "simple_papi.h"
#include <immintrin.h>
#include <omp.h>


void initMatrix(int n, double UNUSED(mass), double UNUSED(fcon),
                int UNUSED(delta), double UNUSED(grav), double sep,
                double rball, double offset, double dt, double **x,
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

void loopcode(int n, const double mass, double fcon, int delta, double grav,
              double sep, double const rball, const double xball, const double yball,
              const double zball, const double dt, double * __restrict__ x, double * __restrict__ y, double * __restrict__ z,
              double * __restrict__ fx, double * __restrict__ fy, double * __restrict__ fz, double * __restrict__ vx, double * __restrict__ vy,
              double * __restrict__ vz, double * __restrict__ oldfx, double * __restrict__ oldfy, double * __restrict__ oldfz,
              double * __restrict__ pe, double * __restrict__ ke, double * __restrict__ te, int p) {
  __declspec(align(8))
  const int n_square = n * n;
  int k;
  double xdiff, ydiff, zdiff, vmag, damp, velDotDiff;
  double update_const = dt * 0.5 / mass;
  omp_set_num_threads(p);
  // sse for update position as per MD simulation
  double zpo = 0.1;
  const __m256d sse_dt = _mm256_set1_pd(dt);
  
  const __m256d sse_update_const = _mm256_set1_pd(update_const);
  const __m256d sse_xball = _mm256_set1_pd(xball);
  const __m256d sse_yball = _mm256_set1_pd(yball);
  const __m256d sse_zball = _mm256_set1_pd(zball);
  const __m256d sse_rball = _mm256_set1_pd(rball);
    
    #pragma omp parallel
    {
      #pragma omp for
      for (k = 0; k < (n_square/4)*4; k+= 4)
      {
        __m256d sse_zpo = _mm256_set1_pd(zpo);
        // notice here in order to prevent the high expansive of divide, we use * 1/mass instead
        __m256d sse_vx = _mm256_loadu_pd(&vx[k]);
        __m256d sse_fx = _mm256_loadu_pd(&fx[k]);
        __m256d sse_x = _mm256_loadu_pd(&x[k]);
                sse_x = _mm256_add_pd(sse_x, 
                        _mm256_mul_pd(sse_dt, 
                        _mm256_add_pd(sse_vx,
                        _mm256_mul_pd(sse_fx, sse_update_const))));
        __m256d sse_oldfx = _mm256_loadu_pd(&fx[k]);
        __m256d sse_vy = _mm256_loadu_pd(&vy[k]);
        __m256d sse_fy = _mm256_loadu_pd(&fy[k]);
        __m256d sse_y = _mm256_loadu_pd(&y[k]);
                sse_y = _mm256_add_pd(sse_y, 
                        _mm256_mul_pd(sse_dt, 
                        _mm256_add_pd(sse_vy,
                        _mm256_mul_pd(sse_fy, sse_update_const))));
        __m256d sse_oldfy = _mm256_loadu_pd(&fy[k]);
        __m256d sse_vz = _mm256_loadu_pd(&vz[k]);
        __m256d sse_fz = _mm256_loadu_pd(&fz[k]);
        __m256d sse_z = _mm256_loadu_pd(&z[k]);
                sse_z = _mm256_add_pd(sse_z, 
        _mm256_mul_pd(sse_dt, 
        _mm256_add_pd(sse_vz,
        _mm256_mul_pd(sse_fz, sse_update_const))));
        __m256d sse_oldfz = _mm256_loadu_pd(&fz[k]);

        _mm256_storeu_pd(&oldfx[k], sse_oldfx);
        _mm256_storeu_pd(&oldfy[k], sse_oldfy);
        _mm256_storeu_pd(&oldfz[k], sse_oldfz);

        // sse version	apply constraints - push cloth outside of ball
        __m256d sse_xdiff = _mm256_sub_pd(sse_xball, sse_x);
        __m256d sse_ydiff = _mm256_sub_pd(sse_yball, sse_y);
        __m256d sse_zdiff = _mm256_sub_pd(sse_zball, sse_z);
        __m256d sse_vmag = _mm256_sqrt_pd(
                            _mm256_add_pd(
                            _mm256_add_pd(
                            _mm256_mul_pd(sse_xdiff, sse_xdiff),
                            _mm256_mul_pd(sse_ydiff, sse_ydiff)), 
                            _mm256_mul_pd(sse_zdiff, sse_zdiff)));

        __m256d sse_judge_vmag = _mm256_cmp_pd(sse_vmag, sse_rball, 1);

        __m256d sse_update_pos_const = _mm256_div_pd(sse_rball, sse_vmag); // general
        // update position inside if-statement.
        __m256d sse_x_anti_pre = _mm256_andnot_pd(sse_judge_vmag, sse_x);
        sse_x = _mm256_and_pd(sse_judge_vmag, _mm256_sub_pd(sse_xball, _mm256_mul_pd(sse_xdiff, sse_update_pos_const)));
        sse_x = _mm256_add_pd(sse_x, sse_x_anti_pre);
        __m256d sse_y_anti_pre = _mm256_andnot_pd(sse_judge_vmag, sse_y);
        sse_y = _mm256_and_pd(sse_judge_vmag, _mm256_sub_pd(sse_yball, _mm256_mul_pd(sse_ydiff, sse_update_pos_const)));
        sse_y = _mm256_add_pd(sse_y, sse_y_anti_pre);
        __m256d sse_z_anti_pre = _mm256_andnot_pd(sse_judge_vmag, sse_z);
        sse_z = _mm256_and_pd(sse_judge_vmag, _mm256_sub_pd(sse_zball, _mm256_mul_pd(sse_zdiff, sse_update_pos_const)));
        sse_z = _mm256_add_pd(sse_z, sse_z_anti_pre);

        // calculate velDotDiff
        __m256d sse_velDotDiff = _mm256_add_pd(
                _mm256_add_pd(
                _mm256_mul_pd(sse_vx, sse_xdiff),
                _mm256_mul_pd(sse_vy, sse_ydiff)),
                _mm256_mul_pd(sse_vz, sse_zdiff)); // general
        __m256d sse_v_update_const = _mm256_div_pd(sse_velDotDiff, _mm256_mul_pd(sse_vmag, sse_vmag)); // general
        __m256d sse_vx_anti = _mm256_andnot_pd(sse_judge_vmag, sse_vx);
        // x:
        sse_vx = _mm256_mul_pd(
          _mm256_sub_pd(sse_vx, 
          _mm256_mul_pd(sse_xdiff, sse_v_update_const)), sse_zpo);
        sse_vx = _mm256_and_pd(sse_judge_vmag, sse_vx);
        sse_vx = _mm256_add_pd(sse_vx, sse_vx_anti);
        // y:
        __m256d sse_vy_anti = _mm256_andnot_pd(sse_judge_vmag, sse_vy);
        sse_vy = _mm256_mul_pd(
          _mm256_sub_pd(sse_vy, 
          _mm256_mul_pd(sse_ydiff, sse_v_update_const)), sse_zpo);
        sse_vy = _mm256_and_pd(sse_judge_vmag, sse_vy);
        sse_vy = _mm256_add_pd(sse_vy, sse_vy_anti);
        // z:
        __m256d sse_vz_anti = _mm256_andnot_pd(sse_judge_vmag, sse_vz);
        sse_vz = _mm256_mul_pd(
          _mm256_sub_pd(sse_vz, 
          _mm256_mul_pd(sse_zdiff, sse_v_update_const)), sse_zpo);
        sse_vz = _mm256_and_pd(sse_judge_vmag, sse_vz);
        sse_vz = _mm256_add_pd(sse_vz, sse_vz_anti);

        _mm256_storeu_pd(&x[k], sse_x);
        _mm256_storeu_pd(&y[k], sse_y);
        _mm256_storeu_pd(&z[k], sse_z);
        _mm256_storeu_pd(&vx[k], sse_vx);
        _mm256_storeu_pd(&vy[k], sse_vy);
        _mm256_storeu_pd(&vz[k], sse_vz);        
      }
        // we have to treat these elements separately if the size of the array is not divisable by 4
      #pragma omp for
      for (k = (n_square / 4) * 4; k < n_square; k++){
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
          x[k] = xball - xdiff * rball / vmag;
          y[k] = yball - ydiff * rball / vmag;
          z[k] = zball - zdiff * rball / vmag;

          velDotDiff = vx[k] * xdiff + vy[k] * ydiff + vz[k] * zdiff;
          vx[k] = (vx[k] - velDotDiff / vmag * xdiff / vmag) * 0.1;
          vy[k] = (vy[k] - velDotDiff / vmag * ydiff / vmag) * 0.1;
          vz[k] = (vz[k] - velDotDiff / vmag * zdiff / vmag) * 0.1;
        }
      }
    } 
    // Add a damping factor to eventually set velocity to zero
  *pe = eval_pef(n, delta, mass, grav, sep, fcon, x, y, z, fx, fy, fz, p);

  // Add a damping factor to eventually set velocity to zero
  damp = 0.995;
  *ke = 0.0;
  __m256d sse_damp = _mm256_set1_pd(damp);
  __m256d sse_ke_part = _mm256_set1_pd(0.0);
  double tmp_ke = 0.0;
  double ke_part[4] = {0.0, 0.0, 0.0, 0.0};

  #pragma omp parallel private(k, ke_part) reduction(+:tmp_ke) num_threads(p)
  {
    #pragma omp for
    for (k = 0; k < (n_square / 4) * 4; k += 4)
    {
      // x:
      __m256d sse_vx = _mm256_loadu_pd(&vx[k]);
      __m256d sse_oldfx = _mm256_loadu_pd(&oldfx[k]);
      __m256d sse_fx = _mm256_loadu_pd(&fx[k]);
      sse_vx = _mm256_mul_pd(sse_damp,
                            _mm256_add_pd(sse_vx,
                                          _mm256_mul_pd(sse_update_const,
                                                        _mm256_add_pd(sse_fx, sse_oldfx))));
      // y:
      __m256d sse_vy = _mm256_loadu_pd(&vy[k]);
      __m256d sse_oldfy = _mm256_loadu_pd(&oldfy[k]);
      __m256d sse_fy = _mm256_loadu_pd(&fy[k]);
      sse_vy = _mm256_mul_pd(sse_damp,
                            _mm256_add_pd(sse_vy,
                                          _mm256_mul_pd(sse_update_const,
                                                        _mm256_add_pd(sse_fy, sse_oldfy))));
      // z:
      __m256d sse_vz = _mm256_loadu_pd(&vz[k]);
      __m256d sse_oldfz = _mm256_loadu_pd(&oldfz[k]);
      __m256d sse_fz = _mm256_loadu_pd(&fz[k]);
      sse_vz = _mm256_mul_pd(sse_damp,
                            _mm256_add_pd(sse_vz,
                                          _mm256_mul_pd(sse_update_const,
                                                        _mm256_add_pd(sse_fz, sse_oldfz))));
      _mm256_storeu_pd(&vx[k], sse_vx);
      _mm256_storeu_pd(&vy[k], sse_vy);
      _mm256_storeu_pd(&vz[k], sse_vz);

      sse_ke_part = _mm256_add_pd(
      _mm256_add_pd(
      _mm256_mul_pd(sse_vx, sse_vx), 
      _mm256_mul_pd(sse_vy, sse_vy)), 
      _mm256_mul_pd(sse_vz, sse_vz));
      _mm256_storeu_pd(ke_part, sse_ke_part);
      tmp_ke += ke_part[0] + ke_part[1] + ke_part[2] + ke_part[3];

    }
    // _mm256_storeu_pd(ke_part, sse_ke_part);
    // *ke = ke_part[0] + ke_part[1] + ke_part[2] + ke_part[3];
    // #pragma omp for
    for (k = (n_square / 4) * 4; k < n_square; k++){
      vx[k] = (vx[k] + (fx[k] + oldfx[k]) * update_const) * damp;
      vy[k] = (vy[k] + (fy[k] + oldfy[k]) * update_const) * damp;
      vz[k] = (vz[k] + (fz[k] + oldfz[k]) * update_const) * damp;
      tmp_ke += vx[k] * vx[k] + vy[k] * vy[k] + vz[k] * vz[k];
    }
  }
  *ke = tmp_ke / 2.0;
  *te = *pe + *ke;
}

double eval_pef(int n, int delta, double mass, double grav, double sep,
                double fcon, double * __restrict__ x, double * __restrict__ y, double * __restrict__ z, double * __restrict__ fx,
                double * __restrict__ fy, double * __restrict__ fz, int p) {
  double pe, rlen, xdiff, ydiff, zdiff, vmag;
  int nx, ny, dx, dy, i, j, k;
  double update_f_cons;
  __m256d sse_fxfy = _mm256_set1_pd(0.0);
  __m256d sse_z = _mm256_set1_pd((double)-mass*grav);
  #pragma omp parallel for
  #pragma ivdep
  for (k = 0; k < (n * n / 4) * 4; k+=4)
  {
    _mm256_storeu_pd(&fx[k], sse_fxfy);
    _mm256_storeu_pd(&fy[k], sse_fxfy);
    _mm256_storeu_pd(&fz[k], sse_z);
  }
  #pragma vector aligned
  for (k = (n * n / 4) * 4; k< n*n; k++) {
      fx[k] = 0.0;
      fy[k] = 0.0;
      fz[k] = -mass * grav;
  }
  pe = 0.0;
  // loop over particles
  // for (k = 0; k < (n_square/4)*4; k+= 4)
  __m256d sse_sep = _mm256_set1_pd(sep);
  __m256d sse_fcon = _mm256_set1_pd(fcon);
  double fx_add_part_n[4];
  double fy_add_part_n[4];
  double fz_add_part_n[4];
  double pe_sum[4] = {0.0, 0.0, 0.0, 0.0};
  int block_x, block_y, block_num, block_size;
  // block_num = (int) sqrt(p);
  // block_size = (int) n/block_num;
  block_num = (int) n/(8+delta);
  block_size = (int) n/block_num;

  // Plan A
  // first implement a way without parallel
  int block_size_psedu = block_size - delta;
  int move_block = block_size + delta;
  int delta2p1 = 2*delta+1;
  #pragma omp parallel for \
  private(pe_sum,fx_add_part_n, fy_add_part_n, fz_add_part_n,\
  block_x, block_y, i, j, nx, ny, dx, dy, rlen, xdiff, ydiff, zdiff, vmag, update_f_cons)\
  reduction(+:pe) collapse(2) num_threads(p) schedule(static, 1)
  for (block_x=0; block_x < n; block_x+= move_block){
    for (block_y=0; block_y < n; block_y+= move_block){
      #pragma prefetch x:3:5
      #pragma prefetch y:3:5
      #pragma prefetch z:3:5
      #pragma prefetch fx:3:5
      #pragma prefetch fy:3:5
      #pragma prefetch fz:3:5
      #pragma prefetch x:3:3999
      #pragma prefetch y:3:3999
      #pragma prefetch z:3:3999
      #pragma prefetch fx:3:3999
      #pragma prefetch fy:3:3999
      #pragma prefetch fz:3:3999
      // printf("the thread now is %d \n: ", omp_get_thread_num());
      for (i=0; i<delta2p1; i++){
        for (j=0; j<delta2p1; j++){
          for (nx = block_x+i; nx < MIN(block_x + block_size_psedu, n); nx+=delta2p1)
          {
            for (ny = block_y+j; ny < MIN(block_y+block_size_psedu, n); ny+=delta2p1) {
              int nxny = nx * n + ny;
              __builtin_prefetch(&fx[nxny], 0, 2);
              __builtin_prefetch(&fy[nxny], 0, 2);
              __builtin_prefetch(&fz[nxny], 0, 2);

              __builtin_prefetch(&x[nxny], 0, 2);
              __builtin_prefetch(&y[nxny], 0, 2);
              __builtin_prefetch(&z[nxny], 0, 2);
              __m256d sse_ny = _mm256_set1_pd((double)ny);
              __m256d sse_x_n = _mm256_set1_pd(x[nx*n+ny]);
              __m256d sse_y_n = _mm256_set1_pd(y[nx*n+ny]);
              __m256d sse_z_n = _mm256_set1_pd(z[nx*n+ny]);
              for (int tmp = 0; tmp < 4; tmp++){
                fx_add_part_n[tmp] = 0.0;
                fy_add_part_n[tmp] = 0.0;
                fz_add_part_n[tmp] = 0.0;
              }
              __m256d sse_pe_sum = _mm256_set1_pd(0.0);
              __m256d sse_fx_add_part_n = _mm256_set1_pd(0.0);
              __m256d sse_fy_add_part_n = _mm256_set1_pd(0.0);
              __m256d sse_fz_add_part_n = _mm256_set1_pd(0.0);

              // loop over displacements
              dx = nx;
              // __m256d sse_dx = _mm256_set1_pd((double) dx);
              double nx_m_dx_sqare = (nx-dx)*(nx-dx);
              int dxn = dx * n;
              __m256d sse_nx_m_dx_sqare = _mm256_set1_pd(nx_m_dx_sqare);
              #pragma ivdep
              for (dy = MIN(ny+1, n); MIN(ny + delta + 1, n)-4 >= dy; dy+=4){
                int dxdy = dxn + dy;
                __builtin_prefetch(&fx[dxdy], 0, 2);
                __builtin_prefetch(&fy[dxdy], 0, 2);
                __builtin_prefetch(&fz[dxdy], 0, 2);

                __builtin_prefetch(&x[dxdy], 0, 2);
                __builtin_prefetch(&y[dxdy], 0, 2);
                __builtin_prefetch(&z[dxdy], 0, 2);

                __builtin_prefetch(&fx[dxdy], 1, 2);
                __builtin_prefetch(&fy[dxdy], 1, 2);
                __builtin_prefetch(&fz[dxdy], 1, 2);
                __m256d sse_fx_add_part = _mm256_set1_pd(0.0);
                __m256d sse_fy_add_part = _mm256_set1_pd(0.0);
                __m256d sse_fz_add_part = _mm256_set1_pd(0.0);
                __m256d sse_dy = _mm256_set_pd((double) dy+3.0, dy+2.0, dy+1.0, dy+0.0);
                __m256d sse_ny_m_dy_sqare = _mm256_mul_pd(
                                              _mm256_sub_pd(sse_dy, sse_ny), _mm256_sub_pd(sse_dy, sse_ny));
                // compute reference distance
                __m256d sse_rlen = _mm256_mul_pd(sse_sep,
                                                  _mm256_sqrt_pd(
                                                  _mm256_add_pd(sse_nx_m_dx_sqare, sse_ny_m_dy_sqare
                                                  )));

                // compute actual distance
                __m256d sse_x_d = _mm256_loadu_pd(&x[dxdy]);
                __m256d sse_xdiff = _mm256_sub_pd(sse_x_d, sse_x_n);
                __m256d sse_y_d = _mm256_loadu_pd(&y[dxdy]);
                __m256d sse_ydiff = _mm256_sub_pd(sse_y_d, sse_y_n);
                __m256d sse_z_d = _mm256_loadu_pd(&z[dxdy]);
                __m256d sse_zdiff = _mm256_sub_pd(sse_z_d, sse_z_n);
                __m256d sse_vmag = _mm256_sqrt_pd(
                                      _mm256_add_pd(
                                        _mm256_mul_pd(sse_xdiff, sse_xdiff),
                                          _mm256_add_pd( 
                                            _mm256_mul_pd(sse_ydiff, sse_ydiff),
                                              _mm256_mul_pd(sse_zdiff, sse_zdiff))));

                // potential energy and force
                sse_pe_sum = _mm256_add_pd(sse_pe_sum, _mm256_mul_pd(sse_fcon,
                  _mm256_mul_pd(
                    _mm256_sub_pd(sse_vmag, sse_rlen), 
                    _mm256_sub_pd(sse_vmag, sse_rlen)))); 
                __m256d sse_add_part_const = _mm256_div_pd(
                  _mm256_mul_pd(sse_fcon, 
                  _mm256_sub_pd(sse_vmag, sse_rlen)), sse_vmag);

                sse_fx_add_part = _mm256_mul_pd(sse_xdiff, sse_add_part_const);
                __m256d sse_fx_d = _mm256_loadu_pd(&fx[dxdy]);
                sse_fx_d = _mm256_sub_pd(sse_fx_d, sse_fx_add_part);
                _mm256_storeu_pd(&fx[dxdy], sse_fx_d);
                sse_fx_add_part_n = _mm256_add_pd(sse_fx_add_part_n, sse_fx_add_part);
                sse_fy_add_part = _mm256_mul_pd(sse_ydiff, sse_add_part_const);
                __m256d sse_fy_d = _mm256_loadu_pd(&fy[dxdy]);
                sse_fy_d = _mm256_sub_pd(sse_fy_d, sse_fy_add_part);
                _mm256_storeu_pd(&fy[dxdy], sse_fy_d);
                sse_fy_add_part_n = _mm256_add_pd(sse_fy_add_part_n, sse_fy_add_part);
                sse_fz_add_part = _mm256_mul_pd(sse_zdiff, sse_add_part_const);
                __m256d sse_fz_d = _mm256_loadu_pd(&fz[dxdy]);
                sse_fz_d = _mm256_sub_pd(sse_fz_d, sse_fz_add_part);
                _mm256_storeu_pd(&fz[dxdy], sse_fz_d);
                sse_fz_add_part_n = _mm256_add_pd(sse_fz_add_part_n, sse_fz_add_part);
                // refer to the formula given in the readme.md file instead of referring the lecture notes.
              }
              #pragma vector aligned
              #pragma ivdep
              for (; dy < MIN(ny + delta + 1, n); dy+=1){
                __builtin_prefetch(&fx[nxny], 1, 2);
                __builtin_prefetch(&fy[nxny], 1, 2);
                __builtin_prefetch(&fz[nxny], 1, 2);
                int dxdy = dxn + dy;
                rlen =
                sqrt((nx_m_dx_sqare + (ny - dy) * (ny - dy))) *
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

                fx[dxdy] -= xdiff * update_f_cons;
                fy[dxdy] -= ydiff * update_f_cons;
                fz[dxdy] -= zdiff * update_f_cons;
                // refer to the formula given in the readme.md file instead of referring the lecture notes.
              }
              #pragma ivdep
              for (dx = MIN(nx+1, n); dx < MIN(nx + delta + 1, n); dx++) {
                // __m256d sse_dx = _mm256_set1_pd((double) dx);
                int nx_m_dx_sqare = (nx-dx)*(nx-dx);
                int dxn = dx * n;
                __m256d sse_nx_m_dx_sqare = _mm256_set1_pd((double) nx_m_dx_sqare);
                #pragma ivdep
                for (dy = MAX(ny - delta, 0); MIN(ny + delta + 1, n)-4 >= dy; dy+=4){
                  int dxdy = dxn + dy;
                  __builtin_prefetch(&fx[dxdy], 0, 2);
                  __builtin_prefetch(&fy[dxdy], 0, 2);
                  __builtin_prefetch(&fz[dxdy], 0, 2);

                  __builtin_prefetch(&x[dxdy], 0, 2);
                  __builtin_prefetch(&y[dxdy], 0, 2);
                  __builtin_prefetch(&z[dxdy], 0, 2);

                  __builtin_prefetch(&fx[dxdy], 1, 2);
                  __builtin_prefetch(&fy[dxdy], 1, 2);
                  __builtin_prefetch(&fz[dxdy], 1, 2);
                  
                  __m256d sse_fx_add_part = _mm256_set1_pd(0.0);
                  __m256d sse_fy_add_part = _mm256_set1_pd(0.0);
                  __m256d sse_fz_add_part = _mm256_set1_pd(0.0);
                  __m256d sse_dy = _mm256_set_pd((double) dy+3.0, dy+2.0, dy+1.0, dy+0.0);
                  __m256d sse_ny_m_dy_sqare = _mm256_mul_pd(
                                                _mm256_sub_pd(sse_dy, sse_ny), _mm256_sub_pd(sse_dy, sse_ny));
                  // compute reference distance
                  __m256d sse_rlen = _mm256_mul_pd(sse_sep,
                                                  _mm256_sqrt_pd(
                                                  _mm256_add_pd(sse_nx_m_dx_sqare, sse_ny_m_dy_sqare
                                                  )));
                  // compute actual distance
                  __m256d sse_x_d = _mm256_loadu_pd(&x[dxdy]);
                  __m256d sse_xdiff = _mm256_sub_pd(sse_x_d, sse_x_n);
                  __m256d sse_y_d = _mm256_loadu_pd(&y[dxdy]);
                  __m256d sse_ydiff = _mm256_sub_pd(sse_y_d, sse_y_n);
                  __m256d sse_z_d = _mm256_loadu_pd(&z[dxdy]);
                  __m256d sse_zdiff = _mm256_sub_pd(sse_z_d, sse_z_n);
                  __m256d sse_vmag = _mm256_sqrt_pd(
                                        _mm256_add_pd(
                                          _mm256_mul_pd(sse_xdiff, sse_xdiff),
                                            _mm256_add_pd( 
                                              _mm256_mul_pd(sse_ydiff, sse_ydiff),
                                                _mm256_mul_pd(sse_zdiff, sse_zdiff))));
                  // potential energy and force
                  sse_pe_sum = _mm256_add_pd(sse_pe_sum, _mm256_mul_pd(sse_fcon,
                    _mm256_mul_pd(
                      _mm256_sub_pd(sse_vmag, sse_rlen), 
                      _mm256_sub_pd(sse_vmag, sse_rlen)))); 
                  __m256d sse_add_part_const = _mm256_div_pd(
                    _mm256_mul_pd(sse_fcon, 
                    _mm256_sub_pd(sse_vmag, sse_rlen)), sse_vmag);
                  sse_fx_add_part = _mm256_mul_pd(sse_xdiff, sse_add_part_const);
                  __m256d sse_fx_d = _mm256_loadu_pd(&fx[dxdy]);
                  sse_fx_d = _mm256_sub_pd(sse_fx_d, sse_fx_add_part);
                  _mm256_storeu_pd(&fx[dxdy], sse_fx_d);
                  sse_fx_add_part_n = _mm256_add_pd(sse_fx_add_part_n, sse_fx_add_part);
                  sse_fy_add_part = _mm256_mul_pd(sse_ydiff, sse_add_part_const);
                  __m256d sse_fy_d = _mm256_loadu_pd(&fy[dxdy]);
                  sse_fy_d = _mm256_sub_pd(sse_fy_d, sse_fy_add_part);
                  _mm256_storeu_pd(&fy[dxdy], sse_fy_d);
                  sse_fy_add_part_n = _mm256_add_pd(sse_fy_add_part_n, sse_fy_add_part);
                  sse_fz_add_part = _mm256_mul_pd(sse_zdiff, sse_add_part_const);
                  __m256d sse_fz_d = _mm256_loadu_pd(&fz[dx*n+dy]);
                  sse_fz_d = _mm256_sub_pd(sse_fz_d, sse_fz_add_part);
                  _mm256_storeu_pd(&fz[dxdy], sse_fz_d);
                  sse_fz_add_part_n = _mm256_add_pd(sse_fz_add_part_n, sse_fz_add_part);
                  // refer to the formula given in the readme.md file instead of referring the lecture notes.
                }
                #pragma vector aligned
                #pragma ivdep
                for (; dy < MIN(ny + delta + 1, n); dy+=1){
                  int dxdy = dxn + dy;
                  rlen =
                  sqrt((nx_m_dx_sqare + (ny - dy) * (ny - dy))) *
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

                  fx[dxdy] -= xdiff * update_f_cons;
                  fy[dxdy] -= ydiff * update_f_cons;
                  fz[dxdy] -= zdiff * update_f_cons;
                  // refer to the formula given in the readme.md file instead of referring the lecture notes.
                }
              }

              _mm256_storeu_pd(pe_sum, sse_pe_sum);
              pe += pe_sum[0] + pe_sum[1] + pe_sum[2] + pe_sum[3];
              _mm256_storeu_pd(fx_add_part_n, sse_fx_add_part_n);
              _mm256_storeu_pd(fy_add_part_n, sse_fy_add_part_n);
              _mm256_storeu_pd(fz_add_part_n, sse_fz_add_part_n);
              fx[nxny] += fx_add_part_n[0]+fx_add_part_n[1]+fx_add_part_n[2]+fx_add_part_n[3];
              fy[nxny] += fy_add_part_n[0]+fy_add_part_n[1]+fy_add_part_n[2]+fy_add_part_n[3];
              fz[nxny] += fz_add_part_n[0]+fz_add_part_n[1]+fz_add_part_n[2]+fz_add_part_n[3];
            }
          }
        }
      }
    }}

  // Plan B
  int block_x_b;
  int move_block_b = block_size_psedu + 2 * delta;
  #pragma omp parallel for \
  private(pe_sum,fx_add_part_n, fy_add_part_n, fz_add_part_n,\
  block_x_b, i, j, nx, ny, dx, dy, rlen, xdiff, ydiff, zdiff, vmag, update_f_cons)\
  reduction(+:pe) schedule(static,1)
  for (block_x_b = block_size - delta; block_x_b < n; block_x_b += move_block_b)
  {
    // printf("thread %d \n", omp_get_thread_num());
    for (i = 0; i < delta2p1; i++){
      for (j = 0; j < delta2p1; j++){
        #pragma prefetch x:3:5
        #pragma prefetch y:3:5
        #pragma prefetch z:3:5
        #pragma prefetch fx:3:5
        #pragma prefetch fy:3:5
        #pragma prefetch fz:3:5
        #pragma prefetch x:3:3999
        #pragma prefetch y:3:3999
        #pragma prefetch z:3:3999
        #pragma prefetch fx:3:3999
        #pragma prefetch fy:3:3999
        #pragma prefetch fz:3:3999
        for (nx = block_x_b+i; nx < MIN(block_x_b+2*delta, n); nx+=delta2p1){
          int nxn = nx * n;
          for (ny = j; ny <  n; ny+=delta2p1) {
            int nxny = nxn + ny;
            __builtin_prefetch(&fx[nxny], 0, 2);
            __builtin_prefetch(&fy[nxny], 0, 2);
            __builtin_prefetch(&fz[nxny], 0, 2);

            __builtin_prefetch(&x[nxny], 0, 2);
            __builtin_prefetch(&y[nxny], 0, 2);
            __builtin_prefetch(&z[nxny], 0, 2);

            __m256d sse_ny = _mm256_set1_pd((double) ny);
            __m256d sse_x_n = _mm256_set1_pd(x[nxny]);
            __m256d sse_y_n = _mm256_set1_pd(y[nxny]);
            __m256d sse_z_n = _mm256_set1_pd(z[nxny]);
            for (int tmp = 0; tmp < 4; tmp++){
              fx_add_part_n[tmp] = 0.0;
              fy_add_part_n[tmp] = 0.0;
              fz_add_part_n[tmp] = 0.0;
            }
            __m256d sse_pe_sum = _mm256_set1_pd(0.0);
            __m256d sse_fx_add_part_n = _mm256_set1_pd(0.0);
            __m256d sse_fy_add_part_n = _mm256_set1_pd(0.0);
            __m256d sse_fz_add_part_n = _mm256_set1_pd(0.0);

            // loop over displacements
            dx = nx;
            // __m256d sse_dx = _mm256_set1_pd((double) dx);
            int nx_m_dx_sqare = (nx-dx)*(nx-dx);
            __m256d sse_nx_m_dx_sqare = _mm256_set1_pd((double) nx_m_dx_sqare);
            int dxn = dx * n;
            #pragma ivdep
            for (dy = MIN(ny + 1, n); MIN(ny + delta + 1, n) - 4 >= dy; dy += 4)
            {
              int dxdy = dxn+dy;
              __builtin_prefetch(&fx[dxdy], 0, 2);
              __builtin_prefetch(&fy[dxdy], 0, 2);
              __builtin_prefetch(&fz[dxdy], 0, 2);

              __builtin_prefetch(&x[dxdy], 0, 2);
              __builtin_prefetch(&y[dxdy], 0, 2);
              __builtin_prefetch(&z[dxdy], 0, 2);

              __builtin_prefetch(&fx[dxdy], 1, 2);
              __builtin_prefetch(&fy[dxdy], 1, 2);
              __builtin_prefetch(&fz[dxdy], 1, 2);
              __m256d sse_fx_add_part = _mm256_set1_pd(0.0);
              __m256d sse_fy_add_part = _mm256_set1_pd(0.0);
              __m256d sse_fz_add_part = _mm256_set1_pd(0.0);
              __m256d sse_dy = _mm256_set_pd((double) dy+3.0, dy+2.0, dy+1.0, dy+0.0);
              __m256d sse_ny_m_dy_sqare = _mm256_mul_pd(
                                            _mm256_sub_pd(sse_dy, sse_ny), _mm256_sub_pd(sse_dy, sse_ny));
              // compute reference distance
              __m256d sse_rlen = _mm256_mul_pd(sse_sep,
                                                _mm256_sqrt_pd(
                                                _mm256_add_pd(sse_nx_m_dx_sqare, sse_ny_m_dy_sqare
                                                )));

              // compute actual distance
              __m256d sse_x_d = _mm256_loadu_pd(&x[dxdy]);
              __m256d sse_xdiff = _mm256_sub_pd(sse_x_d, sse_x_n);
              __m256d sse_y_d = _mm256_loadu_pd(&y[dxdy]);
              __m256d sse_ydiff = _mm256_sub_pd(sse_y_d, sse_y_n);
              __m256d sse_z_d = _mm256_loadu_pd(&z[dxdy]);
              __m256d sse_zdiff = _mm256_sub_pd(sse_z_d, sse_z_n);
              __m256d sse_vmag = _mm256_sqrt_pd(
                                    _mm256_add_pd(
                                      _mm256_mul_pd(sse_xdiff, sse_xdiff),
                                        _mm256_add_pd( 
                                          _mm256_mul_pd(sse_ydiff, sse_ydiff),
                                            _mm256_mul_pd(sse_zdiff, sse_zdiff))));

              // potential energy and force
              sse_pe_sum = _mm256_add_pd(sse_pe_sum, _mm256_mul_pd(sse_fcon,
                _mm256_mul_pd(
                  _mm256_sub_pd(sse_vmag, sse_rlen), 
                  _mm256_sub_pd(sse_vmag, sse_rlen)))); 
              __m256d sse_add_part_const = _mm256_div_pd(
                _mm256_mul_pd(sse_fcon, 
                _mm256_sub_pd(sse_vmag, sse_rlen)), sse_vmag);

              sse_fx_add_part = _mm256_mul_pd(sse_xdiff, sse_add_part_const);
              __m256d sse_fx_d = _mm256_loadu_pd(&fx[dxdy]);
              sse_fx_d = _mm256_sub_pd(sse_fx_d, sse_fx_add_part);
              _mm256_storeu_pd(&fx[dxdy], sse_fx_d);
              sse_fx_add_part_n = _mm256_add_pd(sse_fx_add_part_n, sse_fx_add_part);
              sse_fy_add_part = _mm256_mul_pd(sse_ydiff, sse_add_part_const);
              __m256d sse_fy_d = _mm256_loadu_pd(&fy[dxdy]);
              sse_fy_d = _mm256_sub_pd(sse_fy_d, sse_fy_add_part);
              _mm256_storeu_pd(&fy[dxdy], sse_fy_d);
              sse_fy_add_part_n = _mm256_add_pd(sse_fy_add_part_n, sse_fy_add_part);
              sse_fz_add_part = _mm256_mul_pd(sse_zdiff, sse_add_part_const);
              __m256d sse_fz_d = _mm256_loadu_pd(&fz[dx*n+dy]);
              sse_fz_d = _mm256_sub_pd(sse_fz_d, sse_fz_add_part);
              _mm256_storeu_pd(&fz[dxdy], sse_fz_d);
              sse_fz_add_part_n = _mm256_add_pd(sse_fz_add_part_n, sse_fz_add_part);
              // refer to the formula given in the readme.md file instead of referring the lecture notes.
            }
            #pragma ivdep
            for (; dy < MIN(ny + delta + 1, n); dy+=1){
              int dxdy = dxn + dy;
              rlen =
              sqrt((nx_m_dx_sqare + (ny - dy) * (ny - dy))) *
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

              fx[dxdy] -= xdiff * update_f_cons;
              fy[dxdy] -= ydiff * update_f_cons;
              fz[dxdy] -= zdiff * update_f_cons;
              // refer to the formula given in the readme.md file instead of referring the lecture notes.
            }
            #pragma ivdep
            for (dx = MIN(nx+1, n); dx < MIN(nx + delta + 1, n); dx++) {
              // __m256d sse_dx = _mm256_set1_pd((double) dx);
              int nx_m_dx_sqare = (nx-dx)*(nx-dx);
              __m256d sse_nx_m_dx_sqare = _mm256_set1_pd((double) nx_m_dx_sqare);
              int dxn = dx*n;
              for (dy = MAX(ny - delta, 0); MIN(ny + delta + 1, n)-4 >= dy; dy+=4){
                int dxdy = dxn+dy;
                __m256d sse_fx_add_part = _mm256_set1_pd(0.0);
                __m256d sse_fy_add_part = _mm256_set1_pd(0.0);
                __m256d sse_fz_add_part = _mm256_set1_pd(0.0);
                __m256d sse_dy = _mm256_set_pd((double) dy+3.0, dy+2.0, dy+1.0, dy+0.0);
                __m256d sse_ny_m_dy_sqare = _mm256_mul_pd(
                                              _mm256_sub_pd(sse_dy, sse_ny), _mm256_sub_pd(sse_dy, sse_ny));
                // compute reference distance
                __m256d sse_rlen = _mm256_mul_pd(sse_sep,
                                                _mm256_sqrt_pd(
                                                _mm256_add_pd(sse_nx_m_dx_sqare, sse_ny_m_dy_sqare
                                                )));
                // compute actual distance
                __m256d sse_x_d = _mm256_loadu_pd(&x[dxdy]);
                __m256d sse_xdiff = _mm256_sub_pd(sse_x_d, sse_x_n);
                __m256d sse_y_d = _mm256_loadu_pd(&y[dxdy]);
                __m256d sse_ydiff = _mm256_sub_pd(sse_y_d, sse_y_n);
                __m256d sse_z_d = _mm256_loadu_pd(&z[dxdy]);
                __m256d sse_zdiff = _mm256_sub_pd(sse_z_d, sse_z_n);
                __m256d sse_vmag = _mm256_sqrt_pd(
                                      _mm256_add_pd(
                                        _mm256_mul_pd(sse_xdiff, sse_xdiff),
                                          _mm256_add_pd( 
                                            _mm256_mul_pd(sse_ydiff, sse_ydiff),
                                              _mm256_mul_pd(sse_zdiff, sse_zdiff))));
                // potential energy and force
                sse_pe_sum = _mm256_add_pd(sse_pe_sum, _mm256_mul_pd(sse_fcon,
                  _mm256_mul_pd(
                    _mm256_sub_pd(sse_vmag, sse_rlen), 
                    _mm256_sub_pd(sse_vmag, sse_rlen)))); 
                __m256d sse_add_part_const = _mm256_div_pd(
                  _mm256_mul_pd(sse_fcon, 
                  _mm256_sub_pd(sse_vmag, sse_rlen)), sse_vmag);
                sse_fx_add_part = _mm256_mul_pd(sse_xdiff, sse_add_part_const);
                __m256d sse_fx_d = _mm256_loadu_pd(&fx[dxdy]);
                sse_fx_d = _mm256_sub_pd(sse_fx_d, sse_fx_add_part);
                _mm256_storeu_pd(&fx[dxdy], sse_fx_d);
                sse_fx_add_part_n = _mm256_add_pd(sse_fx_add_part_n, sse_fx_add_part);
                sse_fy_add_part = _mm256_mul_pd(sse_ydiff, sse_add_part_const);
                __m256d sse_fy_d = _mm256_loadu_pd(&fy[dxdy]);
                sse_fy_d = _mm256_sub_pd(sse_fy_d, sse_fy_add_part);
                _mm256_storeu_pd(&fy[dxdy], sse_fy_d);
                sse_fy_add_part_n = _mm256_add_pd(sse_fy_add_part_n, sse_fy_add_part);
                sse_fz_add_part = _mm256_mul_pd(sse_zdiff, sse_add_part_const);
                __m256d sse_fz_d = _mm256_loadu_pd(&fz[dxdy]);
                sse_fz_d = _mm256_sub_pd(sse_fz_d, sse_fz_add_part);
                _mm256_storeu_pd(&fz[dxdy], sse_fz_d);
                sse_fz_add_part_n = _mm256_add_pd(sse_fz_add_part_n, sse_fz_add_part);
                // refer to the formula given in the readme.md file instead of referring the lecture notes.
              }
              #pragma ivdep
              for (; dy < MIN(ny + delta + 1, n); dy+=1){
                int dxdy = dxn + dy;
                rlen =
                sqrt((nx_m_dx_sqare + (ny - dy) * (ny - dy))) *
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

                fx[dxdy] -= xdiff * update_f_cons;
                fy[dxdy] -= ydiff * update_f_cons;
                fz[dxdy] -= zdiff * update_f_cons;
                // refer to the formula given in the readme.md file instead of referring the lecture notes.
              }
            }

            _mm256_storeu_pd(pe_sum, sse_pe_sum);
            pe += pe_sum[0] + pe_sum[1] + pe_sum[2] + pe_sum[3];
            _mm256_storeu_pd(fx_add_part_n, sse_fx_add_part_n);
            _mm256_storeu_pd(fy_add_part_n, sse_fy_add_part_n);
            _mm256_storeu_pd(fz_add_part_n, sse_fz_add_part_n);
            fx[nx * n + ny] += fx_add_part_n[0]+fx_add_part_n[1]+fx_add_part_n[2]+fx_add_part_n[3];
            fy[nx * n + ny] += fy_add_part_n[0]+fy_add_part_n[1]+fy_add_part_n[2]+fy_add_part_n[3];
            fz[nx * n + ny] += fz_add_part_n[0]+fz_add_part_n[1]+fz_add_part_n[2]+fz_add_part_n[3];
          }
        }
      }
    }
  }

  // Plan C
  // #pragma omp parallel for \
  // private(pe_sum,fx_add_part_n, fy_add_part_n, fz_add_part_n,\
  // block_x_c, block_y_c, i, j, nx, ny, dx, dy, rlen, xdiff, ydiff, zdiff, vmag, update_f_cons)\
  // reduction(+:pe) num_threads(p) schedule(static)
  int block_x_c, block_y_c;
  // omp_set_nested(1);
  omp_set_max_active_levels(2);
  #pragma omp parallel for \
  private(pe_sum,fx_add_part_n, fy_add_part_n, fz_add_part_n,\
  block_x_c, block_y_c, i, j, nx, ny, dx, dy, rlen, xdiff, ydiff, zdiff, vmag, update_f_cons)\
  reduction(+:pe) num_threads(p) schedule(static,1)
  for (block_x_c = 0; block_x_c < n; block_x_c+=(block_size_psedu+2*delta)){
    for (block_y_c = block_size - delta; block_y_c < n; block_y_c+= (block_size_psedu+2*delta)){
      // printf("thread %d \n", omp_get_thread_num());
      #pragma prefetch x:3:5
      #pragma prefetch y:3:5
      #pragma prefetch z:3:5
      #pragma prefetch fx:3:5
      #pragma prefetch fy:3:5
      #pragma prefetch fz:3:5
      #pragma prefetch x:3:3999
      #pragma prefetch y:3:3999
      #pragma prefetch z:3:3999
      #pragma prefetch fx:3:3999
      #pragma prefetch fy:3:3999
      #pragma prefetch fz:3:3999
      for (i=0; i<2*delta+1;i++){
        for (j = 0; j < 2 * delta + 1;j++){
          for (nx = block_x_c+i; nx < MIN(block_x_c + block_size_psedu, n); nx+=2*delta+1){
            int nxn = nx*n;
            // printf("\n nx equal to: %d \n", nx);
            for (ny = block_y_c+j; ny < MIN(block_y_c + 2*delta, n); ny+=2*delta+1){
              int nxny = nxn + ny;
              // printf("ny equal to: %d \n", ny);
              __m256d sse_ny = _mm256_set1_pd((double) ny);
              __m256d sse_x_n = _mm256_set1_pd(x[nx*n+ny]);
              __m256d sse_y_n = _mm256_set1_pd(y[nx*n+ny]);
              __m256d sse_z_n = _mm256_set1_pd(z[nx*n+ny]);
              #pragma ivdep
              for (int tmp = 0; tmp < 4; tmp++){
                fx_add_part_n[tmp] = 0.0;
                fy_add_part_n[tmp] = 0.0;
                fz_add_part_n[tmp] = 0.0;
              }
              __m256d sse_pe_sum = _mm256_set1_pd(0.0);
              __m256d sse_fx_add_part_n = _mm256_set1_pd(0.0);
              __m256d sse_fy_add_part_n = _mm256_set1_pd(0.0);
              __m256d sse_fz_add_part_n = _mm256_set1_pd(0.0);

              // loop over displacements
              dx = nx;
              int dxn = dx * n;
              // __m256d sse_dx = _mm256_set1_pd((double) dx);
              int nx_m_dx_sqare = (nx-dx)*(nx-dx);
              __m256d sse_nx_m_dx_sqare = _mm256_set1_pd((double) nx_m_dx_sqare);
              #pragma ivdep
              for (dy = MIN(ny+1, n); MIN(ny + delta + 1, n)-4 >= dy; dy+=4){
                int dxdy = dxn + dy;
                __m256d sse_fx_add_part = _mm256_set1_pd(0.0);
                __m256d sse_fy_add_part = _mm256_set1_pd(0.0);
                __m256d sse_fz_add_part = _mm256_set1_pd(0.0);
                __m256d sse_dy = _mm256_set_pd((double) dy+3.0, dy+2.0, dy+1.0, dy+0.0);
                __m256d sse_ny_m_dy_sqare = _mm256_mul_pd(
                                              _mm256_sub_pd(sse_dy, sse_ny), _mm256_sub_pd(sse_dy, sse_ny));
                // compute reference distance
                __m256d sse_rlen = _mm256_mul_pd(sse_sep,
                                                  _mm256_sqrt_pd(
                                                  _mm256_add_pd(sse_nx_m_dx_sqare, sse_ny_m_dy_sqare
                                                  )));

                // compute actual distance
                __m256d sse_x_d = _mm256_loadu_pd(&x[dxdy]);
                __m256d sse_xdiff = _mm256_sub_pd(sse_x_d, sse_x_n);
                __m256d sse_y_d = _mm256_loadu_pd(&y[dxdy]);
                __m256d sse_ydiff = _mm256_sub_pd(sse_y_d, sse_y_n);
                __m256d sse_z_d = _mm256_loadu_pd(&z[dxdy]);
                __m256d sse_zdiff = _mm256_sub_pd(sse_z_d, sse_z_n);
                __m256d sse_vmag = _mm256_sqrt_pd(
                                      _mm256_add_pd(
                                        _mm256_mul_pd(sse_xdiff, sse_xdiff),
                                          _mm256_add_pd( 
                                            _mm256_mul_pd(sse_ydiff, sse_ydiff),
                                              _mm256_mul_pd(sse_zdiff, sse_zdiff))));

                // potential energy and force
                sse_pe_sum = _mm256_add_pd(sse_pe_sum, _mm256_mul_pd(sse_fcon,
                  _mm256_mul_pd(
                    _mm256_sub_pd(sse_vmag, sse_rlen), 
                    _mm256_sub_pd(sse_vmag, sse_rlen)))); 
                __m256d sse_add_part_const = _mm256_div_pd(
                  _mm256_mul_pd(sse_fcon, 
                  _mm256_sub_pd(sse_vmag, sse_rlen)), sse_vmag);

                sse_fx_add_part = _mm256_mul_pd(sse_xdiff, sse_add_part_const);
                __m256d sse_fx_d = _mm256_loadu_pd(&fx[dxdy]);
                sse_fx_d = _mm256_sub_pd(sse_fx_d, sse_fx_add_part);
                _mm256_storeu_pd(&fx[dxdy], sse_fx_d);
                sse_fx_add_part_n = _mm256_add_pd(sse_fx_add_part_n, sse_fx_add_part);
                sse_fy_add_part = _mm256_mul_pd(sse_ydiff, sse_add_part_const);
                __m256d sse_fy_d = _mm256_loadu_pd(&fy[dxdy]);
                sse_fy_d = _mm256_sub_pd(sse_fy_d, sse_fy_add_part);
                _mm256_storeu_pd(&fy[dxdy], sse_fy_d);
                sse_fy_add_part_n = _mm256_add_pd(sse_fy_add_part_n, sse_fy_add_part);
                sse_fz_add_part = _mm256_mul_pd(sse_zdiff, sse_add_part_const);
                __m256d sse_fz_d = _mm256_loadu_pd(&fz[dxdy]);
                sse_fz_d = _mm256_sub_pd(sse_fz_d, sse_fz_add_part);
                _mm256_storeu_pd(&fz[dxdy], sse_fz_d);
                sse_fz_add_part_n = _mm256_add_pd(sse_fz_add_part_n, sse_fz_add_part);
                // refer to the formula given in the readme.md file instead of referring the lecture notes.
              }
              #pragma vector aligned
              #pragma ivdep
              for (; dy < MIN(ny + delta + 1, n); dy+=1){
                int dxdy = dxn + dy;
                rlen =
                sqrt((nx_m_dx_sqare + (ny - dy) * (ny - dy))) *
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

                fx[dxdy] -= xdiff * update_f_cons;
                fy[dxdy] -= ydiff * update_f_cons;
                fz[dxdy] -= zdiff * update_f_cons;
                // refer to the formula given in the readme.md file instead of referring the lecture notes.
              }
              #pragma ivdep
              for (dx = MIN(nx+1, n); dx < MIN(nx + delta + 1, n); dx++) {
                // __m256d sse_dx = _mm256_set1_pd((double) dx);
                int nx_m_dx_sqare = (nx-dx)*(nx-dx);
                __m256d sse_nx_m_dx_sqare = _mm256_set1_pd((double) nx_m_dx_sqare);
                dxn = dx * n;

                for (dy = MAX(ny - delta, 0); MIN(ny + delta + 1, n)-4 >= dy; dy+=4){
                  int dxdy = dxn + dy;
                  __m256d sse_fx_add_part = _mm256_set1_pd(0.0);
                  __m256d sse_fy_add_part = _mm256_set1_pd(0.0);
                  __m256d sse_fz_add_part = _mm256_set1_pd(0.0);
                  __m256d sse_dy = _mm256_set_pd((double) dy+3.0, dy+2.0, dy+1.0, dy+0.0);
                  __m256d sse_ny_m_dy_sqare = _mm256_mul_pd(
                                                _mm256_sub_pd(sse_dy, sse_ny), _mm256_sub_pd(sse_dy, sse_ny));
                  // compute reference distance
                  __m256d sse_rlen = _mm256_mul_pd(sse_sep,
                                                  _mm256_sqrt_pd(
                                                  _mm256_add_pd(sse_nx_m_dx_sqare, sse_ny_m_dy_sqare
                                                  )));
                  // compute actual distance
                  __m256d sse_x_d = _mm256_loadu_pd(&x[dxdy]);
                  __m256d sse_xdiff = _mm256_sub_pd(sse_x_d, sse_x_n);
                  __m256d sse_y_d = _mm256_loadu_pd(&y[dxdy]);
                  __m256d sse_ydiff = _mm256_sub_pd(sse_y_d, sse_y_n);
                  __m256d sse_z_d = _mm256_loadu_pd(&z[dxdy]);
                  __m256d sse_zdiff = _mm256_sub_pd(sse_z_d, sse_z_n);
                  __m256d sse_vmag = _mm256_sqrt_pd(
                                        _mm256_add_pd(
                                          _mm256_mul_pd(sse_xdiff, sse_xdiff),
                                            _mm256_add_pd( 
                                              _mm256_mul_pd(sse_ydiff, sse_ydiff),
                                                _mm256_mul_pd(sse_zdiff, sse_zdiff))));
                  // potential energy and force
                  sse_pe_sum = _mm256_add_pd(sse_pe_sum, _mm256_mul_pd(sse_fcon,
                    _mm256_mul_pd(
                      _mm256_sub_pd(sse_vmag, sse_rlen), 
                      _mm256_sub_pd(sse_vmag, sse_rlen)))); 
                  __m256d sse_add_part_const = _mm256_div_pd(
                    _mm256_mul_pd(sse_fcon, 
                    _mm256_sub_pd(sse_vmag, sse_rlen)), sse_vmag);
                  sse_fx_add_part = _mm256_mul_pd(sse_xdiff, sse_add_part_const);
                  __m256d sse_fx_d = _mm256_loadu_pd(&fx[dxdy]);
                  sse_fx_d = _mm256_sub_pd(sse_fx_d, sse_fx_add_part);
                  _mm256_storeu_pd(&fx[dxdy], sse_fx_d);
                  sse_fx_add_part_n = _mm256_add_pd(sse_fx_add_part_n, sse_fx_add_part);
                  sse_fy_add_part = _mm256_mul_pd(sse_ydiff, sse_add_part_const);
                  __m256d sse_fy_d = _mm256_loadu_pd(&fy[dxdy]);
                  sse_fy_d = _mm256_sub_pd(sse_fy_d, sse_fy_add_part);
                  _mm256_storeu_pd(&fy[dxdy], sse_fy_d);
                  sse_fy_add_part_n = _mm256_add_pd(sse_fy_add_part_n, sse_fy_add_part);
                  sse_fz_add_part = _mm256_mul_pd(sse_zdiff, sse_add_part_const);
                  __m256d sse_fz_d = _mm256_loadu_pd(&fz[dx*n+dy]);
                  sse_fz_d = _mm256_sub_pd(sse_fz_d, sse_fz_add_part);
                  _mm256_storeu_pd(&fz[dxdy], sse_fz_d);
                  sse_fz_add_part_n = _mm256_add_pd(sse_fz_add_part_n, sse_fz_add_part);
                  // refer to the formula given in the readme.md file instead of referring the lecture notes.
                }
                #pragma vector aligned
                #pragma ivdep
                for (; dy < MIN(ny + delta + 1, n); dy+=1){
                  int dxdy = dxn + dy;
                  rlen =
                  sqrt((nx_m_dx_sqare + (ny - dy) * (ny - dy))) *
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

                  fx[dxdy] -= xdiff * update_f_cons;
                  fy[dxdy] -= ydiff * update_f_cons;
                  fz[dxdy] -= zdiff * update_f_cons;
                  // refer to the formula given in the readme.md file instead of referring the lecture notes.
                }
              }

              _mm256_storeu_pd(pe_sum, sse_pe_sum);
              pe += pe_sum[0] + pe_sum[1] + pe_sum[2] + pe_sum[3];
              _mm256_storeu_pd(fx_add_part_n, sse_fx_add_part_n);
              _mm256_storeu_pd(fy_add_part_n, sse_fy_add_part_n);
              _mm256_storeu_pd(fz_add_part_n, sse_fz_add_part_n);
              fx[nxny] += fx_add_part_n[0]+fx_add_part_n[1]+fx_add_part_n[2]+fx_add_part_n[3];
              fy[nxny] += fy_add_part_n[0]+fy_add_part_n[1]+fy_add_part_n[2]+fy_add_part_n[3];
              fz[nxny] += fz_add_part_n[0]+fz_add_part_n[1]+fz_add_part_n[2]+fz_add_part_n[3];
            }
          }
        }
      }
    }
    }
            
  // printf("pe value is %f  \n", pe);
  return pe;
}


