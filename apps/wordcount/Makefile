# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.42 						 04/29/2018 */
#    							huili@ruijie.com.cn */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: wordcount

OPTFLAGS    = -O2
INCFLAGS    = -I/opt/openmpi/include
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=0
NVCCFLAGS   = $(CFLAGS)
LDFLAGS	    = -L/opt/openmpi/lib/
LIBS        = -lmpi -lpthread

CC          = mpicc 
MPICC       = mpicxx 
NVCC        = nvcc

NVCCFLAGS  = -I/opt/openmpi/include -lcudart -arch=sm_20 -I../../include --relocatable-device-code=true
INCFLAGS   += -I/usr/include/  -I/usr/local/cuda/include
INCFLAGS   += -I../../include -I../../include/panda
LDFLAGS    += -L/usr/local/cuda/lib64/ 

CPP_FILES_0     := $(wildcard ./*.cpp)
CPP_FILES_1 	:= $(wildcard ../../src/oscpp/*.cpp)
CPP_FILES_2     := $(wildcard ../../src/pandajob/*.cpp)
CPP_FILES_3	:= $(wildcard ../../src/cudacpp/*.cpp)
CPP_FILES_4     := $(wildcard ../../src/message/*.cpp)
CPP_FILES_5     := $(wildcard ../../src/inputformat/*.cpp)

CU_FILES_0 	:= $(wildcard ../../src/runtime/*.cu)
CU_FILES_1	:= $(wildcard ./*.cu)

H_FILES_0	:= $(wildcard ./*.h)
H_FILES_1 	:= $(wildcard ../../include/oscpp/*.h)
H_FILES_2 	:= $(wildcard ../../include/panda/*.h)
H_FILES_3 	:= $(wildcard ../../include/cudacpp/*.h)
H_FILES_4 	:= $(wildcard ../../include/*.h)

TARGET_H_FILES  := $(H_FILES_0) $(H_FILES_1) $(H_FILES_2) \
			$(H_FILES_3) $(H_FILES_4)
TARGET_CPP_FILES:= $(CPP_FILES_0) $(CPP_FILES_1) $(CPP_FILES_2) \
		     $(CPP_FILES_3) $(CPP_FILES_4) $(CPP_FILES_5)
TARGET_CU_FILES := $(CU_FILES_0) $(CU_FILES_1)
CPP_OBJ_FILES   := $(addprefix ../../obj/,$(notdir $(TARGET_CPP_FILES:.cpp=.o)))
CU_OBJ_FILES    := $(addprefix ../../cuobj/,$(notdir $(TARGET_CU_FILES:.cu=.o)))
TARGET_OBJ_FILES:= $(CPP_OBJ_FILES) $(CU_OBJ_FILES)

wordcount: $(TARGET_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o ../../bin/wordcount $^

FILES_0		:= ./%.cpp
FILES_1 	:= ../../src/oscpp/%.cpp
FILES_2     	:= ../../src/pandajob/%.cpp
FILES_3		:= ../../src/cudacpp/%.cpp
FILES_4     	:= ../../src/message/%.cpp
FILES_5     	:= ../../src/inputformat/%.cpp

../../obj/%.o: $(FILES_0) $(TARGET_H_FILES) 
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<
../../obj/%.o: $(FILES_1) $(TARGET_H_FILES) 
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<
../../obj/%.o: $(FILES_2) $(TARGET_H_FILES) 
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<
../../obj/%.o: $(FILES_3) $(TARGET_H_FILES) 
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<
../../obj/%.o: $(FILES_4) $(TARGET_H_FILES) 
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<
../../obj/%.o: $(FILES_5) $(TARGET_H_FILES) 
	$(MPICC) $(LIBS)  $(CC_FLAGS) $(INCFLAGS) -c -o $@ $<

FILES_6 	:= ../../src/runtime/%.cu
FILES_7		:= ./%.cu

../../cuobj/%.o: $(FILES_6) $(TARGET_H_FILES)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<
../../cuobj/%.o: $(FILES_7) $(TARGET_H_FILES)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf ../../obj/*.o ../../cuobj/*.o ../../bin/wordcount 
