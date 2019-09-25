EIGEN='/usr/local/include/eigen3'

all: gslDeterm gslLinEq gslProduct eigDetermBeta eigDeterm

gslDeterm: gslDeterm.c
	gcc gslDeterm.c -lgsl -lgslcblas -o gslDeterm
    
gslLinEq: gslLinEq.c
	gcc gslLinEq.c -lgsl -lgslcblas -o gslLinEq

gslProduct: gslProduct.c
	gcc gslProduct.c -lgsl -lgslcblas -o gslProduct
    
eigDetermBeta: eigDetermBeta.cpp
	g++ eigDetermBeta.cpp -o eigDetermBeta -I /usr/local/bin/ -I $(EIGEN)
	
eigDeterm: eigDeterm.cpp
	g++ -o eigDeterm eigDeterm.cpp -I /usr/local/bin/ -I $(EIGEN)
	
clean:
	rm -f gslLinEq
	rm -f gslDeterm
	rm -f eigDeterm
	rm -f eigDetermBeta
	rm -f gslProduct
