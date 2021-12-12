#ifndef TEMPORAL_PROPAGATION_H_
#define TEMPORAL_PROPAGATION_H_

void convert_cart2curvilin(double* data, double* curvilin_coord);


void convert_curvilin2cart(double* data, double* cart_coord);


void epicyclic_approx(double* data, double t, double* new_position);


void trace_epicyclic_orbit(double* xyzuvw_start, double t, 
    double* xyzuvw_new);


void trace_epicyclic_covmatrix(double* cov, double* loc, double t,
    int dim, float h, double* cov_transformed);

#endif // TEMPORAL_PROPAGATION_H_
