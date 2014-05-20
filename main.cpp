#include "interop.h"
#include <GL/glut.h>
#include <thread>
#include <set>

int oldWidth = -1, oldHeight = -1;
ClamKernel kernel;
ClamInterop interop;
float offsetX = 0.f, offsetY = 0.f, zoom = 1.f;

std::set<unsigned char> pressedKeys;
std::set<int> pressedSpecialKeys;

auto lastUpdate = std::chrono::steady_clock::now();

void displayFunc()
{
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    if (width != oldWidth || height != oldHeight)
    {
        oldWidth = width;
        oldHeight = height;
        interop.Resize(width, height);
    }
    interop.Render(kernel, offsetX, offsetY, zoom);

    glutSwapBuffers();
}

void updateState()
{
    auto thisUpdate = std::chrono::steady_clock::now();
    auto frameSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(thisUpdate - lastUpdate).count() / 1000.f;
    lastUpdate = thisUpdate;

    if (pressedSpecialKeys.find(GLUT_KEY_UP) != pressedSpecialKeys.end())
        offsetY -= zoom * frameSeconds;
    if (pressedSpecialKeys.find(GLUT_KEY_DOWN) != pressedSpecialKeys.end())
        offsetY += zoom * frameSeconds;
    if (pressedSpecialKeys.find(GLUT_KEY_LEFT) != pressedSpecialKeys.end())
        offsetX -= zoom * frameSeconds;
    if (pressedSpecialKeys.find(GLUT_KEY_RIGHT) != pressedSpecialKeys.end())
        offsetX += zoom * frameSeconds;
    if (pressedKeys.find('w') != pressedKeys.end())
        zoom /= 1 + frameSeconds;
    if (pressedKeys.find('s') != pressedKeys.end())
        zoom *= 1 + frameSeconds;
}

void keyboardDownFunc(unsigned char key, int x, int y)
{
    if (pressedKeys.find(key) != pressedKeys.end())
        return;
    pressedKeys.insert(key);
}

void keyboardUpFunc(unsigned char key, int x, int y)
{
    if (pressedKeys.find(key) == pressedKeys.end())
        return;
    pressedKeys.erase(pressedKeys.find(key));
}

void specialDownFunc(int key, int x, int y)
{
    if (pressedSpecialKeys.find(key) != pressedSpecialKeys.end())
        return;
    pressedSpecialKeys.insert(key);
}

void specialUpFunc(int key, int x, int y)
{
    if (pressedSpecialKeys.find(key) == pressedSpecialKeys.end())
        return;
    pressedSpecialKeys.erase(pressedSpecialKeys.find(key));
}

void idleFunc()
{
    updateState();
    glutPostRedisplay();
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutCreateWindow("Clam2");
    glewInit();
    ClamContext context;
    kernel = ClamKernel(context.GetContext(), context.GetDevice(), "kernel.cl");
    interop = ClamInterop(context.GetContext());

    glutDisplayFunc(displayFunc);
    glutKeyboardUpFunc(keyboardUpFunc);
    glutKeyboardFunc(keyboardDownFunc);
    glutIdleFunc(idleFunc);
    glutSpecialUpFunc(specialUpFunc);
    glutSpecialFunc(specialDownFunc);
    glutMainLoop();
    return 0;
}
