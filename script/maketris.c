#include <lua.h>
#include <lauxlib.h>
#include <assimp/cimport.h>
#include <assimp/scene.h>

void writetri(struct aiVector3D vert0, struct aiVector3D vert1, struct aiVector3D vert2,
        FILE* file)
{
    float data[] = {
        vert0.x, vert0.y, vert0.z,
        vert1.x, vert1.y, vert1.z,
        vert2.x, vert2.y, vert2.z
    };
    fwrite(data, sizeof(float), sizeof(data) / sizeof(float), file);
}

int run_loadtris(lua_State* state)
{
    const char* objfile = luaL_checkstring(state, 1);
    const char* outputFilename = "tris.dat";

    const struct aiScene* scene = aiImportFile(objfile, 0);
    if (!scene)
        luaL_error(state, "Failed to load model: %s", aiGetErrorString());

    FILE* output = fopen(outputFilename, "wb");

    for (unsigned int meshI = 0; meshI < scene->mNumMeshes; meshI++)
    {
        struct aiMesh* mesh = scene->mMeshes[meshI];
        for (unsigned int faceI = 0; faceI < mesh->mNumFaces; faceI++)
        {
            struct aiFace face = mesh->mFaces[faceI];
            switch (face.mNumIndices)
            {
                case 3:
                    writetri(
                            mesh->mVertices[face.mIndices[0]],
                            mesh->mVertices[face.mIndices[1]],
                            mesh->mVertices[face.mIndices[2]],
                            output);
                    break;
                case 4:
                    writetri(
                            mesh->mVertices[face.mIndices[0]],
                            mesh->mVertices[face.mIndices[1]],
                            mesh->mVertices[face.mIndices[2]],
                            output);
                    writetri(
                            mesh->mVertices[face.mIndices[0]],
                            mesh->mVertices[face.mIndices[2]],
                            mesh->mVertices[face.mIndices[3]],
                            output);
                    break;
                default:
                    puts("Face didn't have 3 or 4 indices, skipping polygon");
                    break;
            }
        }
    }

    fclose(output);

    aiReleaseImport(scene);

    lua_pushstring(state, outputFilename);
    return 1;
}

int maketris(lua_State* state)
{
    lua_register(state, "maketris", run_loadtris);
    return 0;
}
