# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.44 						  4/18/2016 */
#    							  lihui@indiana.edu */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: panda

OPTFLAGS    = -O
INCFLAGS    = -I. -I/home/p2plive/Desktop/openmpi-1.10.0/include
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=0 
NVCCFLAGS   = $(CFLAGS) --ptxas-options=-v -arch=sm_20
LDFLAGS	    = -L/home/p2plive/Desktop/openmpi-1.10.0/lib/ -L./libs
LIBS        = -lmpi -lmpi_cxx -lpthread -letransfer -lcommon

#  note:
#  with openmpi add -lmpi_cxx
#  gcc usess -fopenmp
#  icc uses -opnemp
#  support wc,dgemm,gemv,c-means,gmm, change the include USERAPI.cu within PandaLib.cu for compiling code
#  with libetransfer.a & libcommon.a, there is dependency between etransfer & common

OMPFLAGS    = -fopenmp
CC          = g++
MPICC       = mpicxx
NVCC        = nvcc

NVCCFLAGS  += -Xcompiler -fopenmp
INCFLAGS   += -I/usr/local/cuda-5.5/include/ -I/usr/local/cuda-5.5/NVIDIA_GPU_Computing_SDK/C/common/inc/
INCFLAGS   += -I./include -I./apps/ -I./include/panda -I./
LDFLAGS    += -L/usr/local/cuda-5.5/lib64/ -L/usr/local/cuda-5.5/NVIDIA_GPU_Computing_SDK/C/lib/

APP_CPP_FILES	:= $(wildcard apps/*.cpp)
OS_CPP_FILES 	:= $(wildcard src/oscpp/*.cpp)
PANDA_CPP_FILES := $(wildcard src/panda/*.cpp)
CUDA_CPP_FILES 	:= $(wildcard src/cudacpp/*.cpp)
CUDA_CU_FILES 	:= $(wildcard src/*.cu_no)

APP_H_FILES	:= $(wildcard apps/*.h)
OS_H_FILES 	:= $(wildcard include/oscpp/*.h)
PANDA_H_FILES 	:= $(wildcard include/panda/*.h)
CUDA_H_FILES 	:= $(wildcard include/cudacpp/*.h)
H_FILES 	:= $(wildcard include/*.h)

APP_OBJ_FILES	:= $(addprefix obj/,$(notdir $(APP_CPP_FILES:.cpp=.o)))
OS_OBJ_FILES 	:= $(addprefix obj/,$(notdir $(OS_CPP_FILES:.cpp=.o)))
PANDA_OBJ_FILES := $(addprefix obj/,$(notdir $(PANDA_CPP_FILES:.cpp=.o)))
CUDA_OBJ_FILES 	:= $(addprefix obj/,$(notdir $(CUDA_CPP_FILES:.cpp=.o)))
CU_OBJ_FILES 	:= $(addprefix cuobj/,$(notdir $(CUDA_CU_FILES:.cu_no=.o)))

TARGET_OBJ_FILES:=
WC_OBJ_CU_FILES	:=

panda: panda_cmeans
panda_cmeans: $(APP_OBJ_FILES) $(TARGET_OBJ_FILES) $(OS_OBJ_FILES) $(PANDA_OBJ_FILES) \
		$(CUDA_OBJ_FILES) $(CU_OBJ_FILES)
		g++ $(LIBS) $(LDFLAGS) -o $@ $^

test: obj/test.o $(OS_OBJ_FILES) $(PANDA_OBJ_FILES)
		g++ $(LIBS) $(LDFLAGS) -o $@ $^

obj/%.o: apps/%.cpp $(APP_H_FILES) 
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: src/oscpp/%.cpp $(OS_H_FILES)	
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: src/panda/%.cpp $(PANDA_H_FILES) $(H_FILES)
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: src/cudacpp/%.cpp $(CUDA_H_FILES)
	g++ $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

obj/%.o: ./%.cpp $(OS_H_FILES) $(PANDA_H_FILES) $(CUDA_H_FILES) $(H_FILES)
	g++ $(LIBS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

cuobj/%.o: src_no/%.cu $(CUDA_H_FILES) $(H_FILES)
	nvcc $(LIBS) $(NVCCFLAGS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

clean:
	rm -rf obj/*.o cuobj/*.o panda_cmeans test
