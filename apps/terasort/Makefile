# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.41 						 04/29/2018 */
#    							huili@ruijie.com.cn */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: terasort teragen

OPTFLAGS    = -O2
INCFLAGS    = -I/opt/openmpi/include
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=0
NVCCFLAGS   = $(CFLAGS)
LDFLAGS	    = -L/opt/openmpi/lib/
LIBS        = -lmpi -lpthread
CC_FLAGS    = -std=c++11

MPICC       = mpicxx 
NVCC        = nvcc

NVCCFLAGS  = -I/opt/openmpi/include -lcudart -arch=sm_20 -I../../include --relocatable-device-code=true
INCFLAGS   += -I/usr/include/  -I/usr/local/cuda/include
INCFLAGS   += -I../../include  -I../../include/panda
LDFLAGS    += -L/usr/local/cuda/lib64/

CPP_FILES_5	:= terasort_main.cpp Unsigned16.cpp Random16.cpp TeraSortPartitioner.cpp TeraInputFormat.cpp
CPP_FILES_0 	:= $(wildcard ../../src/oscpp/*.cpp)
CPP_FILES_1 	:= $(wildcard ../../src/pandajob/*.cpp)
CPP_FILES_2	:= $(wildcard ../../src/cudacpp/*.cpp)
CPP_FILES_3   	:= $(wildcard ../../src/message/*.cpp)
CPP_FILES_4 	:= $(wildcard ../../src/inputformat/*.cpp)

CU_FILES_1	:= $(wildcard ../../src/runtime/*.cu)
CU_FILES_0	:= terasort_api.cu

TERASORT_CPP_FILES:= $(CPP_FILES_5) $(CPP_FILES_0) $(CPP_FILES_1) $(CPP_FILES_2) $(CPP_FILES_3) $(CPP_FILES_4)
TERASORT_CU_FILES:= $(CU_FILES_0) $(CU_FILES_1) 

H_FILES_0	:= $(wildcard ./*.h)
H_FILES_1 	:= $(wildcard ../../include/oscpp/*.h)
H_FILES_2 	:= $(wildcard ../../include/panda/*.h)
H_FILES_3 	:= $(wildcard ../../include/cudacpp/*.h)
H_FILES_4 	:= $(wildcard ../../include/*.h)
TARGET_H_FILES  := $(H_FILES_0) $(H_FILES_1) $(H_FILES_2) $(H_FILES_3) $(H_FILES_4)

TERASORT_OBJ_FILES:= $(addprefix ../../obj/,$(notdir $(TERASORT_CPP_FILES:.cpp=.o)))
TERASORT_CU_OBJ_FILES:= $(addprefix ../../cuobj/,$(notdir $(TERASORT_CU_FILES:.cu=.o)))

terasort: $(TERASORT_OBJ_FILES) $(TERASORT_CU_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o ../../bin/$@ $^

CPP_FILES_6	:= teragen_main.cpp TeraInputFormat.cpp Unsigned16.cpp Random16.cpp
TERAGEN_CPP_FILES:= $(CPP_FILES_6) $(CPP_FILES_0) $(CPP_FILES_1) $(CPP_FILES_2) $(CPP_FILES_3) $(CPP_FILES_4)

CU_FILES_2	:= teragen_api.cu
TERAGEN_CU_FILES:= $(CU_FILES_2) $(CU_FILES_1) 

TERAGEN_OBJ_FILES:= $(addprefix ../../obj/,$(notdir $(TERAGEN_CPP_FILES:.cpp=.o)))
TERAGEN_CU_OBJ_FILES:= $(addprefix ../../cuobj/,$(notdir $(TERAGEN_CU_FILES:.cu=.o)))

teragen: $(TERAGEN_OBJ_FILES) $(TERAGEN_CU_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o ../../bin/$@ $^

../../obj/%.o: ./%.cpp $(TARGET_H_FILES)
	$(MPICC) $(LIBS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/oscpp/%.cpp $(TARGET_H_FILES)	
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/inputformat/%.cpp $(TARGET_H_FILES)	
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/message/%.cpp $(TARGET_H_FILES)  
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/pandajob/%.cpp $(TARGET_H_FILES)
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../src/cudacpp/%.cpp $(TARGET_H_FILES)
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../obj/%.o: ../../%.cpp $(TARGET_H_FILES)
	$(MPICC) $(LIBS) $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

../../cuobj/%.o: ./%.cu $(TARGET_H_FILES)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

../../cuobj/%.o: ../../src/runtime/%.cu $(TARGET_H_FILES)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf ../../obj/*.o ../../cuobj/*.o ../../bin/terasort  ../../bin/teragen
