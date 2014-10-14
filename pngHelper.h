int savePng(int uid1, int uid2, float* data, long width, long height);
float* loadPng(const char* filename, long* width, long* height); // must call free()
