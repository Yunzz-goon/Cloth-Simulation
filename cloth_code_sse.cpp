#include "./cloth_code_sse.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "simple_papi.h"
#include <immintrin.h>


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
              double *pe, double *ke, double *te){
  const int n_square = n * n;
  int i, j;
  int k;
  double xdiff, ydiff, zdiff, vmag, damp, velDotDiff;
  double update_const = dt * 0.5 / mass;
  double update_v_const;

    // sse for update position as per MD simulation
    double zpo = 0.1;
    const __m256d sse_dt = _mm256_set1_pd(dt);

    const __m256d sse_update_const = _mm256_set1_pd(update_const);
    const __m256d sse_xball = _mm256_set1_pd(xball);
    const __m256d sse_yball = _mm256_set1_pd(yball);
    const __m256d sse_zball = _mm256_set1_pd(zball);
    const __m256d sse_rball = _mm256_set1_pd(rball);
    const __m256d sse_all_true = _mm256_set1_pd(1);

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
    

    // Add a damping factor to eventually set velocity to zero
  *pe = eval_pef(n, delta, mass, grav, sep, fcon, x, y, z, fx, fy, fz);

  // Add a damping factor to eventually set velocity to zero
  damp = 0.995;
  *ke = 0.0;
  __m256d sse_damp = _mm256_set1_pd(damp);
  __m256d sse_ke_part = _mm256_set1_pd(0.0);
  double ke_part[4] = {0.0, 0.0, 0.0, 0.0};
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
    
    sse_ke_part = _mm256_add_pd(sse_ke_part, 
    _mm256_add_pd(
    _mm256_add_pd(
    _mm256_mul_pd(sse_vx, sse_vx), 
    _mm256_mul_pd(sse_vy, sse_vy)), 
    _mm256_mul_pd(sse_vz, sse_vz)));

    _mm256_storeu_pd(&vx[k], sse_vx);
    _mm256_storeu_pd(&vy[k], sse_vy);
    _mm256_storeu_pd(&vz[k], sse_vz);
  }
  _mm256_storeu_pd(ke_part, sse_ke_part);
  *ke = ke_part[0] + ke_part[1] + ke_part[2] + ke_part[3];

  for (k = (n_square / 4) * 4; k < n_square; k++){
    vx[k] = (vx[k] + (fx[k] + oldfx[k]) * update_const) * damp;
    vy[k] = (vy[k] + (fy[k] + oldfy[k]) * update_const) * damp;
    vz[k] = (vz[k] + (fz[k] + oldfz[k]) * update_const) * damp;
    *ke += vx[k] * vx[k] + vy[k] * vy[k] + vz[k] * vz[k];
  }

  *ke = *ke / 2.0;
  *te = *pe + *ke;
}

double eval_pef(int n, int delta, double mass, double grav, double sep,
                double fcon, double * __restrict__ x, double * __restrict__ y, double * __restrict__ z, double * __restrict__ fx,
                double * __restrict__ fy, double * __restrict__ fz) {
  double pe, rlen, xdiff, ydiff, zdiff, vmag;
  int nx, ny, dx, dy, k;
  double update_f_cons;

  __m256d sse_fxfy = _mm256_set1_pd(0.0);
  __m256d sse_z = _mm256_set1_pd((double)-mass*grav);
  for (k = 0; k < (n * n / 4) * 4; k+=4)
  {
    _mm256_storeu_pd(&fx[k], sse_fxfy);
    _mm256_storeu_pd(&fy[k], sse_fxfy);
    _mm256_storeu_pd(&fz[k], sse_z);
  }
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
  for (nx = 0; nx < n; nx++)
  {
    __m256d sse_nx = _mm256_set1_pd((double) nx);
    for (ny = 0; ny < n; ny++) {
      __m256d sse_ny = _mm256_set1_pd((double) ny);
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
      int nx_m_dx_sqare = (nx-dx)*(nx-dx);
      __m256d sse_nx_m_dx_sqare = _mm256_set1_pd((double) nx_m_dx_sqare);

      for (dy = MIN(ny+1, n); MIN(ny + delta + 1, n)-4 >= dy; dy+=4){
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
        __m256d sse_x_d = _mm256_loadu_pd(&x[dx * n + dy]);
        __m256d sse_xdiff = _mm256_sub_pd(sse_x_d, sse_x_n);
        __m256d sse_y_d = _mm256_loadu_pd(&y[dx*n+dy]);
        __m256d sse_ydiff = _mm256_sub_pd(sse_y_d, sse_y_n);
        __m256d sse_z_d = _mm256_loadu_pd(&z[dx*n+dy]);
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
        __m256d sse_fx_d = _mm256_loadu_pd(&fx[dx*n+dy]);
        sse_fx_d = _mm256_sub_pd(sse_fx_d, sse_fx_add_part);
        _mm256_storeu_pd(&fx[dx * n + dy], sse_fx_d);
        sse_fx_add_part_n = _mm256_add_pd(sse_fx_add_part_n, sse_fx_add_part);
        sse_fy_add_part = _mm256_mul_pd(sse_ydiff, sse_add_part_const);
        __m256d sse_fy_d = _mm256_loadu_pd(&fy[dx*n+dy]);
        sse_fy_d = _mm256_sub_pd(sse_fy_d, sse_fy_add_part);
        _mm256_storeu_pd(&fy[dx * n + dy], sse_fy_d);
        sse_fy_add_part_n = _mm256_add_pd(sse_fy_add_part_n, sse_fy_add_part);
        sse_fz_add_part = _mm256_mul_pd(sse_zdiff, sse_add_part_const);
        __m256d sse_fz_d = _mm256_loadu_pd(&fz[dx*n+dy]);
        sse_fz_d = _mm256_sub_pd(sse_fz_d, sse_fz_add_part);
        _mm256_storeu_pd(&fz[dx * n + dy], sse_fz_d);
        sse_fz_add_part_n = _mm256_add_pd(sse_fz_add_part_n, sse_fz_add_part);
        // refer to the formula given in the readme.md file instead of referring the lecture notes.
      }

      for (; dy < MIN(ny + delta + 1, n); dy+=1){
        rlen =
        sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) *
        sep;
        // compute actual distance
        xdiff = x[dx * n + dy] - x[nx * n + ny];
        ydiff = y[dx * n + dy] - y[nx * n + ny];
        zdiff = z[dx * n + dy] - z[nx * n + ny];
        vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
        // potential energy and force
        pe += fcon * (vmag - rlen) * (vmag - rlen);
        update_f_cons = fcon  * (vmag - rlen) / vmag;
        fx[nx * n + ny] += xdiff * update_f_cons;
        fy[nx * n + ny] += ydiff * update_f_cons;
        fz[nx * n + ny] += zdiff * update_f_cons;

        fx[dx * n + dy] -= xdiff * update_f_cons;
        fy[dx * n + dy] -= ydiff * update_f_cons;
        fz[dx * n + dy] -= zdiff * update_f_cons;
        // refer to the formula given in the readme.md file instead of referring the lecture notes.
      }
      
      for (dx = MIN(nx+1, n); dx < MIN(nx + delta + 1, n); dx++) {
        // __m256d sse_dx = _mm256_set1_pd((double) dx);
        int nx_m_dx_sqare = (nx-dx)*(nx-dx);
        __m256d sse_nx_m_dx_sqare = _mm256_set1_pd((double) nx_m_dx_sqare);

        for (dy = MAX(ny - delta, 0); MIN(ny + delta + 1, n)-4 >= dy; dy+=4){
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
          __m256d sse_x_d = _mm256_loadu_pd(&x[dx * n + dy]);
          __m256d sse_xdiff = _mm256_sub_pd(sse_x_d, sse_x_n);
          __m256d sse_y_d = _mm256_loadu_pd(&y[dx*n+dy]);
          __m256d sse_ydiff = _mm256_sub_pd(sse_y_d, sse_y_n);
          __m256d sse_z_d = _mm256_loadu_pd(&z[dx*n+dy]);
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
          __m256d sse_fx_d = _mm256_loadu_pd(&fx[dx*n+dy]);
          sse_fx_d = _mm256_sub_pd(sse_fx_d, sse_fx_add_part);
          _mm256_storeu_pd(&fx[dx * n + dy], sse_fx_d);
          sse_fx_add_part_n = _mm256_add_pd(sse_fx_add_part_n, sse_fx_add_part);
          sse_fy_add_part = _mm256_mul_pd(sse_ydiff, sse_add_part_const);
          __m256d sse_fy_d = _mm256_loadu_pd(&fy[dx*n+dy]);
          sse_fy_d = _mm256_sub_pd(sse_fy_d, sse_fy_add_part);
          _mm256_storeu_pd(&fy[dx * n + dy], sse_fy_d);
          sse_fy_add_part_n = _mm256_add_pd(sse_fy_add_part_n, sse_fy_add_part);
          sse_fz_add_part = _mm256_mul_pd(sse_zdiff, sse_add_part_const);
          __m256d sse_fz_d = _mm256_loadu_pd(&fz[dx*n+dy]);
          sse_fz_d = _mm256_sub_pd(sse_fz_d, sse_fz_add_part);
          _mm256_storeu_pd(&fz[dx * n + dy], sse_fz_d);
          sse_fz_add_part_n = _mm256_add_pd(sse_fz_add_part_n, sse_fz_add_part);
          // refer to the formula given in the readme.md file instead of referring the lecture notes.
        }

        for (; dy < MIN(ny + delta + 1, n); dy+=1){
          rlen =
          sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) *
          sep;
          // compute actual distance
          xdiff = x[dx * n + dy] - x[nx * n + ny];
          ydiff = y[dx * n + dy] - y[nx * n + ny];
          zdiff = z[dx * n + dy] - z[nx * n + ny];
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          // potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen);
          update_f_cons = fcon  * (vmag - rlen) / vmag;
          fx[nx * n + ny] += xdiff * update_f_cons;
          fy[nx * n + ny] += ydiff * update_f_cons;
          fz[nx * n + ny] += zdiff * update_f_cons;

          fx[dx * n + dy] -= xdiff * update_f_cons;
          fy[dx * n + dy] -= ydiff * update_f_cons;
          fz[dx * n + dy] -= zdiff * update_f_cons;
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
  return pe;
}

