#include <stdio.h>

int main(int argc, char **argv)
{
    FILE *input;
    FILE *output;
    const char *varname;
    int c;
    if (argc != 4)
    {
        puts("Argcount must be 4");
        return 1;
    }
    if ((input = fopen(argv[1], "rb")) == 0)
    {
        puts("Cannot open input file");
        return 1;
    }
    if ((output = fopen(argv[2], "w")) == 0)
    {
        puts("Cannot open output file");
        return 1;
    }
    varname = argv[3];
    fprintf(output, "const unsigned char %s[] = {\n", varname);
    int length = 0;
    int column = 0;
    while ((c = fgetc(input)) != EOF)
    {
        fprintf(output, "0x%02x,", (unsigned char)c);
        length++;
        column++;
        if (column >= 12)
        {
            fprintf(output, "\n");
            column = 0;
        }
    }
    fprintf(output, "\n};\n\n");
    fprintf(output, "const int %s_len = %d;\n\n", varname, length);
    fclose(input);
    fclose(output);
    return 0;
}
