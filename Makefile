# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    Panda Code V0.42 						 04/29/2018 */
#    							 huili@ruijie.com.cn*/
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

all: panda

panda: bindir objdir cuobjdir wordcount terasort

BIN_DIR:=bin
OBJ_DIR:=obj
CUOBJ_DIR:=cuobj

bindir:
	@if test ! -d $(BIN_DIR);\
	then \
	  mkdir $(BIN_DIR);\
	fi
objdir:
	@if test ! -d $(OBJ_DIR);\
	then \
	  mkdir $(OBJ_DIR);\
	fi	
cuobjdir:
	@if test ! -d $(CUOBJ_DIR);\
	then \
	  mkdir $(CUOBJ_DIR);\
	fi

wordcount:
	make -C apps/wordcount/

terasort:
	make -C apps/terasort/

clean:
	make clean -C apps/wordcount && make clean -C apps/terasort

clean.old:
	rm -rf obj/*.o cuobj/*.o bin/wordcount bin/terasort
