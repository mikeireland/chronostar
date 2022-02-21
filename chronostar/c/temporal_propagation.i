%module temporal_propagation

%{
  #define SWIG_FILE_WITH_INIT
  #include "temporal_propagation.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}


/* OUTPUT */
/* rangevec must match the names given in header
 * however names in function definitions do not need to match header or here*/
%apply (double* ARGOUT_ARRAY1, int DIM1) \
    {(double* xyzuvw_new, int pos_new_dim),
    (double* cov_transformed, int covt_dim)}


/* INPUT */
%apply (double* IN_ARRAY1, int DIM1) \
      {(double* xyzuvw_start, int pos_dim),
      (double* loc, int loc_dim)}
       
%apply (double* IN_ARRAY2, int DIM1, int DIM2) \
       {(double* cov, int cov_dim1, int cov_dim2)}


%include "temporal_propagation.h"

