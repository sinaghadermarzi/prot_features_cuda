NVCC      ?= nvcc
NVCCFLAGS ?= -O2

all: prot_features

prot_features: Final_work.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f prot_features

.PHONY: all clean
