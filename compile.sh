nvcc -std=c++17 -c SceneNode.cc -o SceneNode.o     
nvcc -std=c++17 -c Renderer.cc -o Renderer.o
nvcc -std=c++17 -rdc=true SceneNode.o Renderer.o main.cu -o main.out 
