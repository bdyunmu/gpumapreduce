# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.61 						 04/29/2018 */
#    							  lihui@indiana.edu */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: terasort teragen

OPTFLAGS    = -O2
INCFLAGS    = -I/opt/openmpi/include
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=0
NVCCFLAGS   = $(CFLAGS)
LDFLAGS	    = -L/opt/openmpi/lib/
LIBS        = -lmpi -lpthread
CC_FLAGS    = -std=c++11
CC          = mpicxx -g
MPICC       = mpicxx -g
NVCC        = nvcc -g

NVCCFLAGS  = -I/opt/openmpi/include -lcudart -arch=sm_20 -I../../include --relocatable-device-code=true
INCFLAGS   += -I/usr/include/  -I/usr/local/cuda/include
INCFLAGS   += -I../../include  -I../../include/panda
LDFLAGS    += -L/usr/local/cuda/lib64/

APP_TERASORT_CPP_FILES:= terasort_main.cpp Unsigned16.cpp Random16.cpp TeraSortPartitioner.cpp TeraInputFormat.cpp
APP_TERASORT_CU_FILES:= terasort_api.cu

OS_CPP_FILES 	:= $(wildcard ../../src/oscpp/*.cpp)
PANDA_CPP_FILES := $(wildcard ../../src/panda/*.cpp)
CUDA_CPP_FILES 	:= $(wildcard ../../src/cudacpp/*.cpp)
CUDA_CU_FILES 	:= $(wildcard ../../src/*.cu)

APP_H_FILES	:= $(wildcard ./*.h)
OS_H_FILES 	:= $(wildcard ../../include/oscpp/*.h)
PANDA_H_FILES 	:= $(wildcard ../../include/panda/*.h)
CUDA_H_FILES 	:= $(wildcard ../../include/cudacpp/*.h)
H_FILES 	:= $(wildcard ../../include/*.h)

APP_TERASORT_OBJ_FILES:= $(addprefix ../../obj/,$(notdir $(APP_TERASORT_CPP_FILES:.cpp=.o)))
APP_TERASORT_CU_OBJ_FILES:= $(addprefix ../../cuobj/,$(notdir $(APP_TERASORT_CU_FILES:.cu=.o)))

OS_OBJ_FILES 	:= $(addprefix ../../obj/,$(notdir $(OS_CPP_FILES:.cpp=.o)))
PANDA_OBJ_FILES := $(addprefix ../../obj/,$(notdir $(PANDA_CPP_FILES:.cpp=.o)))
CUDA_OBJ_FILES 	:= $(addprefix ../../obj/,$(notdir $(CUDA_CPP_FILES:.cpp=.o)))
CU_OBJ_FILES 	:= $(addprefix ../../cuobj/,$(notdir $(CUDA_CU_FILES:.cu=.o)))

TERAGEN_FILES	:= teragen_main.cpp TeraInputFormat.cpp Unsigned16.cpp Random16.cpp
TERAGEN_CU_FILES:= teragen_api.cu

TERAGEN_OBJ_FILES:= $(addprefix ../../obj/,$(notdir $(TERAGEN_FILES:.cpp=.o)))
TERAGEN_CU_OBJ_FILES:= $(addprefix ../../cuobj/,$(notdir $(TERAGEN_CU_FILES:.cu=.o)))

teragen: $(TERAGEN_OBJ_FILES) $(OS_OBJ_FILES) $(PANDA_OBJ_FILES) \
		$(CUDA_OBJ_FILES) $(CU_OBJ_FILES) $(TERAGEN_CU_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o ../../bin/$@ $^

terasort: $(APP_TERASORT_OBJ_FILES) $(OS_OBJ_FILES) $(PANDA_OBJ_FILES) \
		$(CUDA_OBJ_FILES) $(CU_OBJ_FILES) $(APP_TERASORT_CU_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o ../../bin/$@ $^

../../obj/%.o: ./%.cpp $(APP_TERASORT_H_FILES)
	$(MPICC) $(LIBS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/oscpp/%.cpp $(OS_H_FILES)	
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/panda/%.cpp $(PANDA_H_FILES) $(H_FILES)
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/cudacpp/%.cpp $(CUDA_H_FILES)
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../%.cpp $(OS_H_FILES) $(PANDA_H_FILES) $(CUDA_H_FILES) $(H_FILES)
	$(MPICC) $(LIBS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../cuobj/%.o: ./%.cu $(APP_WC_H_FILES)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

../../cuobj/%.o: ../../src/%.cu $(CUDA_H_FILES) $(H_FILES)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf ../../obj/*.o ../../cuobj/*.o ../../bin/terasort  ../../bin/teragen
