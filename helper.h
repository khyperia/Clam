template<typename T>
struct is_pointer { static const bool value = false; };

template<typename T>
struct is_pointer<T*> { static const bool value = true; };

void HandleErrImpl(int errcode, const char* filename, int line);

#define HandleErr(errcode) HandleErrImpl((errcode), __FILE__, __LINE__)

