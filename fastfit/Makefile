CC = cc
CFLAGS = 
LDFLAGS =
OBJFILES = ../chronostar/temporal_propagation.o test_c_modules.o
TARGET = test_c_modules

all: $(TARGET)
$(TARGET): $(OBJFILES)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJFILES) $(LDFLAGS)

clean:
	rm -f $(OBJFILES) $(TARGET) *~
