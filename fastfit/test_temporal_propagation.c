// clang -L/usr/local/lib test_temporal_propagation.c  ../chronostar/temporal_propagation.c -o test_temporal_propagation -lgsl -lgslcblas -lm

#include <stdio.h>

#include "../chronostar/temporal_propagation.h" // Sort out these paths

void test_convert_cart2curvilin_curvilin2cart() {
    printf("test_convert_cart2curvilin_curvilin2cart\n");
    // mean0 (time=0)
    double mean[6];
    mean[0]=54.24222826;
    mean[1]=125.66013072;
    mean[2]=-73.70025391;
    mean[3]=-3.7937685;
    mean[4]=-6.78350686;
    mean[5]=-0.22042236;

    //~ double *curvilin_coord;
    double curvilin_coord[6];
    //~ curvilin_coord = 
    convert_cart2curvilin(mean, curvilin_coord);

    // From python
    double curvilin_coord_python[6];
    curvilin_coord_python[0] = 53.24864894;
    curvilin_coord_python[1] = 126.50741166;
    curvilin_coord_python[2] = -73.70025391;
    curvilin_coord_python[3] = -7.16483698;
    curvilin_coord_python[4] = -5.44204004;
    curvilin_coord_python[5] = -0.22042236;


    printf("curvilin_coord - curvilin_coord_python\n");
    for (int i=0; i<6; ++i) {
        printf("%d %f\n", i, curvilin_coord[i]-
            curvilin_coord_python[i]);
    }
    printf("\n");
    
    double cart_coord[6];
    //~ cart_coord = 
    convert_curvilin2cart(curvilin_coord, cart_coord);

    printf("cart_coord - mean\n");
    for (int i=0; i<6; ++i) {
        printf("%d %f\n", i, cart_coord[i]-mean[i]);
    }

    printf("\n");

}

void test_epicyclic_approx() {
    printf("test_epicyclic_approx()\n");
    // From python
    double curvilin_coord_python[6];
    curvilin_coord_python[0] = 53.24864894;
    curvilin_coord_python[1] = 126.50741166;
    curvilin_coord_python[2] = -73.70025391;
    curvilin_coord_python[3] = -7.16483698;
    curvilin_coord_python[4] = -5.44204004;
    curvilin_coord_python[5] = -0.22042236;
    
    double age=18.80975968983912;
        
    double new_position_python[6];
    new_position_python[0] = -4.11843334;
    new_position_python[1] = -20.13449825;
    new_position_python[2] = -10.4809615;
    new_position_python[3] = 1.34887428;
    new_position_python[4] = -8.64565742;
    new_position_python[5] = 5.69208524;
    
    double new_position[6];
    epicyclic_approx(curvilin_coord_python, age, new_position);
    
    printf("new_position - new_position_python\n");
    for (int i=0; i<6; ++i) {
        printf("%d %f\n", i, new_position[i]-new_position_python[i]);
    }
    printf("\n");
}

void test_trace_epicyclic_orbit() {  
    printf("test_trace_epicyclic_orbit()\n"); 
    // mean0 (time=0)
    double mean[6];
    mean[0]=54.24222826;
    mean[1]=125.66013072;
    mean[2]=-73.70025391;
    mean[3]=-3.7937685;
    mean[4]=-6.78350686;
    mean[5]=-0.22042236;

    // mean_now_python
    double mean_now_python[6];
    mean_now_python[0] = -4.04221889;
    mean_now_python[1] = -23.35241922;
    mean_now_python[2] = -10.54482267;
    mean_now_python[3] = 0.80213607;
    mean_now_python[4] = -8.49588952;
    mean_now_python[5] = 5.56516729;
    
    double age=18.80975968983912;
    
    double mean_now[6];
    //~ mean_now = 
    trace_epicyclic_orbit(mean, age, mean_now);
    
    //~ printf("mean_now\n");
    //~ for (int i=0; i<6; ++i) {
        //~ printf("%d %f\n", i, mean_now[i]);
    //~ }
    //~ printf("\n");    
    
    printf("mean_now - mean_now_python\n");
    for (int i=0; i<6; ++i) {
        printf("%d %f\n", i, mean_now[i]-mean_now_python[i]);
    }
    printf("\n");  
    
}

void test_covmatrix() {
    printf("test_convert_cart2curvilin_curvilin2cart_covmatrix\n");
    // mean0 (time=0)
    double mean[6];
    mean[0]=54.24222826;
    mean[1]=125.66013072;
    mean[2]=-73.70025391;
    mean[3]=-3.7937685;
    mean[4]=-6.78350686;
    mean[5]=-0.22042236;

    //~ double *curvilin_coord;
    double curvilin_coord[6];
    //~ curvilin_coord = 
    convert_cart2curvilin(mean, curvilin_coord);
    
    trace_epicyclic_covmatrix(double* cov, double* loc, double t,
    int dim, float h, double* cov_transformed)

    // From python
    double curvilin_coord_python[6];
    curvilin_coord_python[0] = 53.24864894;
    curvilin_coord_python[1] = 126.50741166;
    curvilin_coord_python[2] = -73.70025391;
    curvilin_coord_python[3] = -7.16483698;
    curvilin_coord_python[4] = -5.44204004;
    curvilin_coord_python[5] = -0.22042236;


    printf("curvilin_coord - curvilin_coord_python\n");
    for (int i=0; i<6; ++i) {
        printf("%d %f\n", i, curvilin_coord[i]-
            curvilin_coord_python[i]);
    }
    printf("\n");
    
    double cart_coord[6];
    //~ cart_coord = 
    convert_curvilin2cart(curvilin_coord, cart_coord);

    printf("cart_coord - mean\n");
    for (int i=0; i<6; ++i) {
        printf("%d %f\n", i, cart_coord[i]-mean[i]);
    }

    printf("\n");

}

int main() {
    // Works well
    //~ test_convert_cart2curvilin_curvilin2cart();
    
    // Works well
    test_epicyclic_approx();
    
    test_trace_epicyclic_orbit();
    

    return 0;
}
