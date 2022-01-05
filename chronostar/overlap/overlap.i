%module overlap

%{
  #define SWIG_FILE_WITH_INIT
  #include "overlap.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

/* not being used */
%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) \
      {(int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D)}

/* rangevec must match the names given in header
 * however names in function definitions do not need to match header or here*/
%apply (double* ARGOUT_ARRAY1, int DIM1) {
    (double* rangevec, int n),
    (double* lnols_output, int n)
}

%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) \
      {(double* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D),
       (double* st_icovs, int st_dim1, int st_dim2, int st_dim3),
       (double* st_covs,  int st_dim1, int st_dim2, int st_dim3)}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) \
      {(double* gr_icov, int gr_dim1, int gr_dim2),
       (double* gr_cov,  int gr_dim1, int gr_dim2),
       (double* st_icov, int st_dim1, int st_dim2),
       (double* st_cov, int st_dim1, int st_dim2),
       (double* st_mns, int st_mn_dim1, int st_mn_dim2)}

%apply (double* IN_ARRAY1, int DIM1) \
      {(double* gr_mn, int gr_mn_dim),
       (double* st_mn, int st_mn_dim),
       (double* st_icov_dets, int st_icov_dets_dim)}

%include "overlap.h"

%clear (double* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D);
