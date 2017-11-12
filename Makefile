# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.45 						 11/04/2017 */
#    							  lihui@indiana.edu */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: panda

OPTFLAGS    = -O2
INCFLAGS    = -I. -I/opt/openmpi/include
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=0 
NVCCFLAGS   = $(CFLAGS) -arch=sm_50
LDFLAGS	    = -L/opt/openmpi/lib/ -L./libs
LIBS        = -lmpi -lmpi_cxx -lpthread

#  note:
#  with openmpi add -lmpi_cxx
#  gcc usess -fopenmp
#  icc uses -opnemp
#  support c-means code

OMPFLAGS    = -fopenmp
CC          = mpicxx
MPICC       = mpicxx
NVCC        = nvcc

NVCCFLAGS  += -lcudart
INCFLAGS   += -I/usr/include/ 
INCFLAGS   += -I./include -I./apps/ -I./include/panda -I./ 
LDFLAGS    += -L/usr/local/cuda/lib64/ 

APP_CPP_FILES	:= $(wildcard apps/*.cpp)
APP_CU_FILES	:= $(wildcard apps/*.cu)
OS_CPP_FILES 	:= $(wildcard src/oscpp/*.cpp)
PANDA_CPP_FILES := $(wildcard src/panda/*.cpp)
CUDA_CPP_FILES 	:= $(wildcard src/cudacpp/*.cpp)
CUDA_CU_FILES 	:= $(wildcard src/*.cu)

APP_H_FILES	:= $(wildcard apps/*.h)
OS_H_FILES 	:= $(wildcard include/oscpp/*.h)
PANDA_H_FILES 	:= $(wildcard include/panda/*.h)
CUDA_H_FILES 	:= $(wildcard include/cudacpp/*.h)
H_FILES 	:= $(wildcard include/*.h)

APP_OBJ_FILES	:= $(addprefix obj/,$(notdir $(APP_CPP_FILES:.cpp=.o)))
APP_CU_OBJ_FILES:= $(addprefix cuobj/,$(notdir $(APP_CU_FILES:.cu=.o)))
OS_OBJ_FILES 	:= $(addprefix obj/,$(notdir $(OS_CPP_FILES:.cpp=.o)))
PANDA_OBJ_FILES := $(addprefix obj/,$(notdir $(PANDA_CPP_FILES:.cpp=.o)))
CUDA_OBJ_FILES 	:= $(addprefix obj/,$(notdir $(CUDA_CPP_FILES:.cpp=.o)))
CU_OBJ_FILES 	:= $(addprefix cuobj/,$(notdir $(CUDA_CU_FILES:.cu=.o)))

TARGET_OBJ_FILES:=
WC_OBJ_CU_FILES	:=

panda: panda_cmeans
panda_cmeans: $(APP_OBJ_FILES) $(TARGET_OBJ_FILES) $(OS_OBJ_FILES) $(PANDA_OBJ_FILES) \
		$(CUDA_OBJ_FILES) $(CU_OBJ_FILES) $(APP_CU_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o $@ $^

test: obj/test.o $(OS_OBJ_FILES) $(PANDA_OBJ_FILES)
		$(MPICC) $(LIBS) $(LDFLAGS) -o $@ $^

obj/%.o: apps/%.cpp $(APP_H_FILES) 
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: src/oscpp/%.cpp $(OS_H_FILES)	
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: src/panda/%.cpp $(PANDA_H_FILES) $(H_FILES)
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: src/cudacpp/%.cpp $(CUDA_H_FILES)
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: ./%.cpp $(OS_H_FILES) $(PANDA_H_FILES) $(CUDA_H_FILES) $(H_FILES)
	$(MPICC) $(LIBS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

cuobj/%.o: src/%.cu $(CUDA_H_FILES) $(H_FILES)
	nvcc $(LIBS) $(NVCCFLAGS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

cuobj/%.o: apps/%.cu $(APP_H_FILES)
	nvcc $(LIBS) $(NVCCFLAGS) $((C_FLAGS) $(INCFLAGS) -c -o $@ $<

clean:
	rm -rf obj/*.o cuobj/*.o panda_cmeans test
