%module overall_likelihood

%{
  #define SWIG_FILE_WITH_INIT
  #include "overall_likelihood.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}


/* INPUT */
%apply (double* IN_ARRAY1, int DIM1) \
      {(double* bg_lnols, int bg_dim)}
       
%apply (double* IN_ARRAY2, int DIM1, int DIM2) \
       {(double* st_mns, int st_mn_dim1, int st_mn_dim2),
       (double* gr_mns, int gr_mns_dim1, int gr_mns_dim2),
       (double* old_memb_probs, int omemb_dim1, int omemb_dim2),
       (double* memb_probs, int memb_dim1, int memb_dim2)
       }

%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) \
      {(double* st_covs, int st_dim1, int st_dim2, int st_dim3),
       (double* gr_covs, int gr_dim1, int gr_dim2, int gr_dim3)
       }

%include "overall_likelihood.h"
