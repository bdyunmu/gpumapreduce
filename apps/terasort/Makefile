# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.41 						 04/29/2018 */
#    							huili@ruijie.com.cn */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: terasort teragen

include ../../makefile.in

CPP_FILES_7	:= terasort_main.cpp Unsigned16.cpp Random16.cpp TeraSortPartitioner.cpp TeraInputFormat.cpp tsoutputformat.cpp

CU_FILES_2	:= terasort_api.cu

TERASORT_CPP_FILES:= $(CPP_FILES_1) $(CPP_FILES_2) $(CPP_FILES_3) $(CPP_FILES_4) $(CPP_FILES_5) $(CPP_FILES_5) $(CPP_FILES_6) $(CPP_FILES_7) 
TERASORT_CU_FILES:= $(CU_FILES_1) $(CU_FILES_2) 

TERASORT_OBJ_FILES:= $(addprefix ../../obj/,$(notdir $(TERASORT_CPP_FILES:.cpp=.o)))
TERASORT_CU_OBJ_FILES:= $(addprefix ../../cuobj/,$(notdir $(TERASORT_CU_FILES:.cu=.o)))

terasort: $(TERASORT_OBJ_FILES) $(TERASORT_CU_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o ../../bin/$@ $^

CPP_FILES_8	:= teragen_main.cpp TeraInputFormat.cpp Unsigned16.cpp Random16.cpp

TERAGEN_CPP_FILES:= $(CPP_FILES_1) $(CPP_FILES_2) $(CPP_FILES_3) $(CPP_FILES_4) $(CPP_FILES_5) $(CPP_FILES_5) $(CPP_FILES_6) $(CPP_FILES_8) 

CU_FILES_3	:= teragen_api.cu
TERAGEN_CU_FILES:= $(CU_FILES_1) $(CU_FILES_3) 

TERAGEN_OBJ_FILES:= $(addprefix ../../obj/,$(notdir $(TERAGEN_CPP_FILES:.cpp=.o)))
TERAGEN_CU_OBJ_FILES:= $(addprefix ../../cuobj/,$(notdir $(TERAGEN_CU_FILES:.cu=.o)))

teragen: $(TERAGEN_OBJ_FILES) $(TERAGEN_CU_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o ../../bin/$@ $^

clean:
	rm -rf ../../obj/*.o ../../cuobj/*.o ../../bin/terasort  ../../bin/teragen
