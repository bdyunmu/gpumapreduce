# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.61 						 04/29/2018 */
#    							  lihui@indiana.edu */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: word_count

OPTFLAGS    = -O2
INCFLAGS    = -I/opt/openmpi/include
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=0
NVCCFLAGS   = $(CFLAGS)
LDFLAGS	    = -L/opt/openmpi/lib/
LIBS        = -lmpi -lpthread

CC          = mpicxx
MPICC       = mpicxx
NVCC        = nvcc

NVCCFLAGS  = -I/opt/openmpi/include -lcudart -arch=sm_30 -I../../include --relocatable-device-code=true
INCFLAGS   += -I/usr/include/  -I/usr/local/cuda/include
INCFLAGS   += -I../../include -I../../include/panda
LDFLAGS    += -L/usr/local/cuda/lib64/ 

APP_WC_CPP_FILES:= $(wildcard ./*.cpp)
APP_WC_CU_FILES	:= $(wildcard ./*.cu)

OS_CPP_FILES 	:= $(wildcard ../../src/oscpp/*.cpp)
PANDA_CPP_FILES := $(wildcard ../../src/panda/*.cpp)
CUDA_CPP_FILES 	:= $(wildcard ../../src/cudacpp/*.cpp)
CUDA_CU_FILES 	:= $(wildcard ../../src/*.cu)

APP_WC_H_FILES	:= $(wildcard ./word_count*.h)
OS_H_FILES 	:= $(wildcard ../../include/oscpp/*.h)
PANDA_H_FILES 	:= $(wildcard ../../include/panda/*.h)
CUDA_H_FILES 	:= $(wildcard ../../include/cudacpp/*.h)
H_FILES 	:= $(wildcard ../../include/*.h)

APP_WC_OBJ_FILES:= $(addprefix ../../obj/,$(notdir $(APP_WC_CPP_FILES:.cpp=.o)))
APP_WC_CU_OBJ_FILES:= $(addprefix ../../cuobj/,$(notdir $(APP_WC_CU_FILES:.cu=.o)))

OS_OBJ_FILES 	:= $(addprefix ../../obj/,$(notdir $(OS_CPP_FILES:.cpp=.o)))
PANDA_OBJ_FILES := $(addprefix ../../obj/,$(notdir $(PANDA_CPP_FILES:.cpp=.o)))
CUDA_OBJ_FILES 	:= $(addprefix ../../obj/,$(notdir $(CUDA_CPP_FILES:.cpp=.o)))
CU_OBJ_FILES 	:= $(addprefix ../../cuobj/,$(notdir $(CUDA_CU_FILES:.cu=.o)))

TARGET_OBJ_FILES:=
WC_OBJ_CU_FILES	:=

word_count: $(APP_WC_OBJ_FILES) $(OS_OBJ_FILES) $(PANDA_OBJ_FILES) \
		$(CUDA_OBJ_FILES) $(CU_OBJ_FILES) $(APP_WC_CU_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o ../../bin/$@ $^

../../obj/%.o: ./%.cpp $(APP_WC_H_FILES) 
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/oscpp/%.cpp $(OS_H_FILES)	
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/panda/%.cpp $(PANDA_H_FILES) $(H_FILES)
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/cudacpp/%.cpp $(CUDA_H_FILES)
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../%.cpp $(OS_H_FILES) $(PANDA_H_FILES) $(CUDA_H_FILES) $(H_FILES)
	$(MPICC) $(LIBS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../cuobj/%.o: ./%.cu $(APP_WC_H_FILES)
	nvcc $(NVCCFLAGS) -c -o $@ $<

../../cuobj/%.o: ../../src/%.cu $(CUDA_H_FILES) $(H_FILES)
	nvcc $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf obj/*.o cuobj/*.o panda_word_count 
