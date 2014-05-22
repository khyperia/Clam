#include "helper.h"
#include <stdexcept>
#include <sstream>

void HandleErrImpl(int errcode, const char* message, const char* filename, int line)
{
    if (errcode != 0)
	{
		std::ostringstream output;
		output << filename;
		output << "(";
		output << line;
		output << "): errcode ";
		output << errcode;
                output << " -- ";
                output << message;
		throw std::runtime_error(output.str());
    }
}
