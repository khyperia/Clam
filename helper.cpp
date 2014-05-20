#include "helper.h"
#include <stdexcept>
#include <sstream>

void HandleErrImpl(int errcode, const char* filename, int line)
{
    if (errcode != 0)
	{
		std::ostringstream output;
		output << filename;
		output << "(";
		output << line;
		output << "): errcode ";
		output << errcode;
		throw std::runtime_error(output.str());
    }
}
