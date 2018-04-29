# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.61 						 04/29/2018 */
#    							  lihui@indiana.edu */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: panda
panda: bin_dir obj_dir cuobj_dir word_count cmeans

BIN_DIR:=bin
OBJ_DIR:=obj
CUOBJ_DIR:=cuobj

bin_dir:
	@if test ! -d $(BIN_DIR);\
	then \
	  mkdir $(BIN_DIR);\
	fi
obj_dir:
	@if test ! -d $(OBJ_DIR);\
	then \
	  mkdir $(OBJ_DIR);\
	fi	
cuobj_dir:
	@if test ! -d $(CUOBJ_DIR);\
	then \
	  mkdir $(CUOBJ_DIR);\
	fi

word_count:
	make -C apps/word_count/

cmeans:
	make -C apps/cmeans/

clean:
	rm -rf obj/*.o cuobj/*.o bin/word_count bin/cmeans
