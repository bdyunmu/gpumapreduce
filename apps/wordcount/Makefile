# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.41 						 04/29/2018 */
#    							huili@ruijie.com.cn */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: wordcount

include ../../makefile.in

CPP_FILES_7	:= wc_main.cpp  wcoutputformat.cpp
CU_FILES_2	:= wc_api.cu

WC_CPP_FILES:= $(CPP_FILES_1) $(CPP_FILES_2) $(CPP_FILES_3) $(CPP_FILES_4) $(CPP_FILES_5) $(CPP_FILES_5) $(CPP_FILES_6) $(CPP_FILES_7) 
WC_CU_FILES:= $(CU_FILES_1) $(CU_FILES_2) 

WC_OBJ_FILES:= $(addprefix ../../obj/,$(notdir $(WC_CPP_FILES:.cpp=.o)))
WC_CU_OBJ_FILES:= $(addprefix ../../cuobj/,$(notdir $(WC_CU_FILES:.cu=.o)))

wordcount: $(WC_OBJ_FILES) $(WC_CU_OBJ_FILES)
		$(NVCC) $(LIBS) $(NVCCFLAGS) $(LDFLAGS) -o ../../bin/$@ $^

clean:
	rm -rf ../../obj/*.o ../../cuobj/*.o ../../bin/wordcount
