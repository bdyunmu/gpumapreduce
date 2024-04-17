# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.42 						 04/18/2024 */
#    							huili@ruijie.com.cn */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: heapsort

include ../../makefile.in

CPP_FILES_7	:= hs_main.cpp  hsoutputformat.cpp
CU_FILES_2	:= hs_api.cu

HS_CPP_FILES:= $(CPP_FILES_1) $(CPP_FILES_2) $(CPP_FILES_3) $(CPP_FILES_4) $(CPP_FILES_5) $(CPP_FILES_5) $(CPP_FILES_6) $(CPP_FILES_7) 
HS_CU_FILES:= $(CU_FILES_1) $(CU_FILES_2) 

HS_OBJ_FILES:= $(addprefix ../../obj/,$(notdir $(HS_CPP_FILES:.cpp=.o)))
HS_CU_OBJ_FILES:= $(addprefix ../../cuobj/,$(notdir $(HS_CU_FILES:.cu=.o)))

heapsort: $(HS_OBJ_FILES) $(HS_CU_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o ../../bin/$@ $^

clean:
	rm -rf ../../obj/*.o ../../cuobj/*.o ../../bin/heapsort
