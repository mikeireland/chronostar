#ifndef TEMPORAL_PROPAGATION_H_
#define TEMPORAL_PROPAGATION_H_

void convert_cart2curvilin(double* data, double* curvilin_coord);


void convert_curvilin2cart(double* data, double* cart_coord);


void epicyclic_approx(double* data, double t, double* new_position);


void trace_epicyclic_orbit(double* xyzuvw_start, int pos_dim, double t, 
    double* xyzuvw_new, int pos_new_dim);


void trace_epicyclic_covmatrix(double* cov, int cov_dim1, int cov_dim2,
    double* loc, int loc_dim, double t, float h, 
    double* cov_transformed, int covt_dim);

#endif // TEMPORAL_PROPAGATION_H_
