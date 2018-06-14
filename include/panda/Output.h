#ifndef _OUTPUT_H_
#define _OUTPUT_H_
namespace panda{

class Output{
    public:
	Output();
	~Output();
        virtual void write(char *buf, void *key, void *val);
};

}

#endif
