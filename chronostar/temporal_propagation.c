#include <stdio.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <string.h>

//#include "temporal_propagation.h"

///# Bovy 2017
///#~ A0 = 15.3  # km/s/kpc
///#~ B0 = -11.9  # km/s/kpc

///# Unit conversion: convert from km/s/kpc to Myr-1
///#~ A = A0 * 0.0010227121650537077  # Myr-1
///#~ B = B0 * 0.0010227121650537077  # Myr-1

///# Bovy 2017, converted from km/s/kpc to Myr-1
///# TODO: Put this (both Oort's constants and the scaling factors) on the input params list. This is where all the conversions should be done, too.
//~ double A0 = 0.01564749613; // 15.3 km/s/kpc * 0.0010227121650537077 = 0.01564749613 Myr-1
//~ double B0 = -0.01217027476; // -11.9 km/s/kpc * 0.0010227121650537077 = -0.01217027476 Myr-1

//~ double A = A0*sA; // km/s/kpc
//~ double B = B0*sB; // km/s/kpc
//~ double rho_scale_factor = sR; // 1.36
//~ double rho = rho_scale_factor * 0.0889; // M0/pc3
//~ double Grho = rho * 0.004498502151575285; // Myr-2; rho should be given in M0/pc3
//~ double kappa = sqrt(-4.0 * B * (A - B)); // Myr-1
//~ double nu = sqrt(4.0 * M_PI * Grho + (A + B) * (A - B)); // Myr-1

// These constants are computed from the numbers above
// TODO: Move this into run_em.py (values like A, B, RO and VO should be
// in the params file (and default)
#define KAPPA 0.039536939622363355
#define NU 0.07796741048638292
#define A 0.0139262715557
#define B -0.013995815973999999
#define AmB 0.027922087529699997 // A-B

#define RO 8.0
#define VO 220.0
#define R0 8000.0 // RO*1000.0
#define Omega0 0.0275 // VO/R0

void convert_cart2curvilin(double* data, double* curvilin_coord) {
    /*
     * Convert 6D cartesian coordinates XYZUVW 
     * (data = (X, Y, Z, U, V, W)) into curvilinear
     * curvilin_coord = (xi, eta, zeta, xidot, etadot, zetadot).
     * Works only for one 6D point, not for an array of points.
     */
    
    // Place the velocities in a rotating frame
    double U = data[3] - data[1]*Omega0;
    double V = data[4] + data[0]*Omega0;

    double R0mX = R0-data[0];
    double R = sqrt(pow(data[1], 2) + pow(R0mX, 2));
    double phi = atan2(data[1], R0mX);
    double cp = cos(phi);
    double sp = sin(phi);

    curvilin_coord[0] = R0-R; // xi
    curvilin_coord[1] = phi*R0; // eta
    curvilin_coord[2] = data[2]; // zeta
    curvilin_coord[3] = U*cp - V*sp; // xidot
    curvilin_coord[4] = R0/R * (V*cp + U*sp); // etadot
    curvilin_coord[5] = data[5]; // zetadot
}


void convert_curvilin2cart(double* data, double* cart_coord) {
    /*
     * Convert 6D curvilinear coordinates 
     * (data = (xi, eta, zeta, xidot, etadot, zetadot)) into cartesian
     * cart_coord = (X, Y, Z, U, V, W).
     * Works only for one 6D point, not for an array of points.
     */

    double R = R0 - data[0]; // R0 is a constant defined in the header
    double phi = data[1]/R0;
    double cp = cos(phi);
    double sp = sin(phi);
    double RR0 = R/R0;

    cart_coord[0] = data[0]*cp + R0*(1.0-cp); // X
    cart_coord[1] = R*sp; // Y
    cart_coord[2] = data[2]; // Z
    cart_coord[3] = data[3]*cp + RR0*data[4]*sp; // U
    cart_coord[4] = - data[3]*sp + RR0*data[4]*cp; // V
    cart_coord[5] = data[5]; // W

    // Convert to a non-rotating observed frame
    cart_coord[3] += cart_coord[1]*Omega0; // Omega0 is a constant defined in the header
    cart_coord[4] += - cart_coord[0]*Omega0;
}


void epicyclic_approx(double* data, double t, double* new_position) {
    /*
     * Propagate curvilinear coordinates 
     * data = (xi0, eta0, zeta0, xidot0, etadot0, zetadot0) 
     * in time t using epicyclic approximation and store them in
     * new_position = (xi, eta, zeta, xidot, etadot, zetadot).
     */

    double kt=KAPPA*t;
    double nt=NU*t;
    double skt = sin(kt);
    double ckt = cos(kt);
    double snt = sin(nt);
    double cnt = cos(nt);
    double KB = KAPPA*B;
    double twoA = 2.0*A;
    double twoB = 2.0*B;
    double Ockt = (1.0 - ckt);
    
    // Propagate positions
    new_position[0] = data[0] + data[3]/KAPPA*skt + 
        (data[4] - twoA*data[0]) * Ockt / twoB;
        
    new_position[1] = data[1] - data[3] * Ockt / twoB + data[4] *
        (A*kt - (AmB)*skt) / KB - data[0] * twoA*(AmB)*(kt-skt) / KB;
    
    new_position[2] = data[2]*cnt + data[5]/NU*snt;
    
    // Propagate velocities
    new_position[3] = data[3]*ckt + (data[4] - twoA*data[0]) * 
        KAPPA*skt / twoB;
        
    new_position[4] = -data[3]*KAPPA/twoB*skt + 
        data[4]/B*(A-(AmB)*ckt) - twoA*data[0]*(AmB)*Ockt/B;
        
    new_position[5] = -data[2]*NU*snt + data[5]*cnt;
}


void trace_epicyclic_orbit(double* xyzuvw_start, int pos_dim, double t, 
    double* xyzuvw_new, int pos_new_dim) {
    // Units: Velocities are in km/s, convert into pc/Myr
    // TODO: This should be done in the input data!!!
    int i;
    for (i=3; i<6; i++) {
        xyzuvw_start[i] *= 1.0227121650537077; // pc/Myr
    }

    // Transform to curvilinear
    double curvilin[6];
    convert_cart2curvilin(xyzuvw_start, curvilin);

    // Trace orbit with epicyclic approximation
    double new_position[6];
    epicyclic_approx(curvilin, t, new_position);

    // Transform new_position back to cartesian
    convert_curvilin2cart(new_position, xyzuvw_new);

    // Units: Transform velocities from pc/Myr back to km/s
    for (i=3; i<6; i++) {
        xyzuvw_new[i] /= 1.0227121650537077; // km/s
    }
}


void trace_epicyclic_covmatrix(double* cov, int cov_dim1, int cov_dim2,
    double* loc, int loc_dim, double t, float h, 
    double* cov_transformed, int covt_dim) {
    /*
     * Transform covariance matrix `cov` into `cov_transformed`.
     * `loc` is a 6D point in space... time t.
     * `h` is delta...
     * 
    */
    
    int dim = cov_dim1; //TODO
    int i, j;
    double loc_pl[loc_dim];
    double loc_mi[loc_dim];
    double final_pos_pl[loc_dim];
    double final_pos_mi[loc_dim];
    
    gsl_matrix* covgsl = gsl_matrix_alloc(dim, dim);
    gsl_matrix* jac = gsl_matrix_alloc(dim, dim);
    gsl_matrix* jac_transposed = gsl_matrix_alloc(dim, dim);


    for (i=0; i<dim; i++) {
        memcpy(loc_pl, loc, sizeof(double)*loc_dim);
        memcpy(loc_mi, loc, sizeof(double)*loc_dim);
        
        loc_pl[i] = loc_pl[i] + h;
        loc_mi[i] = loc_mi[i] - h;

        trace_epicyclic_orbit(loc_pl, loc_dim, t, final_pos_pl, loc_dim);
        trace_epicyclic_orbit(loc_mi, loc_dim, t, final_pos_mi, loc_dim);

        for (j=0; j<dim; j++) {
            gsl_matrix_set(jac, j, i, 
                (final_pos_pl[j] - final_pos_mi[j]) / (2.0*h));
                
            // Set covgsl (we don't run a separate double loop for that)
            gsl_matrix_set(covgsl, i, j, cov[j*dim+i]);
        }
    }
    
    // Compute a transpose of Jacobian
    gsl_matrix_transpose_memcpy(jac_transposed, jac);
    
    gsl_matrix* C = gsl_matrix_alloc(dim, dim);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                      1.0, covgsl, jac_transposed,
                      0.0, C);

    gsl_matrix* D = gsl_matrix_alloc(dim, dim);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                      1.0, jac, C,
                      0.0, D);
        
    // Copy result into cov_transformed that is the result
    // Cannot return 2D array, so copy to 1D and transform to 2D 
    // later in python.
    for (i=0; i<dim; i++) {
        for (j=0; j<dim; j++) {
            cov_transformed[i*dim+j] = gsl_matrix_get(D, i, j);
        }     
    }
    
    // Deallocate the memory
    gsl_matrix_free(jac);
    gsl_matrix_free(jac_transposed);
}
