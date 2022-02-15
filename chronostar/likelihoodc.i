%module likelihoodc

%{
  #define SWIG_FILE_WITH_INIT
  #include "likelihoodc.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}



/* INPUT */
%apply (double* IN_ARRAY1, int DIM1) \
      {(double* pars, int pars_dim),
      (double* args, int args_dim)}
       
%apply (double* IN_ARRAY2, int DIM1, int DIM2) \
       {(double* data, int data_dim1, int data_dim2)}

%include "likelihoodc.h"




