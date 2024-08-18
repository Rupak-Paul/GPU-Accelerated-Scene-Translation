/*
	CS 6023 Assignment 3. 
	Author: Rupak Paul (CS23M056)
*/

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include <iostream>
#include <vector>



__global__
void applyTranslation(int numTranslations, int *d_TranslationMesh, int *d_TranslationCommand, int *d_TranslationAmount, int *d_TotalMovementInXCoord, int *d_TotalMovementInYCoord) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < numTranslations) {
		int mesh = d_TranslationMesh[id];
		int command = d_TranslationCommand[id];
		int amount = d_TranslationAmount[id];

		int *iteTotalMovementArr[2] = {&d_TotalMovementInYCoord[mesh], &d_TotalMovementInXCoord[mesh]};
	  	atomicAdd(iteTotalMovementArr[command < 2], (command%2 ? amount : -amount));
	}
}

__global__
void applyTransitiveTranslation(int *d_Offset, int *d_Csr, int *d_WorkListCurr, int *d_WorkListNew, int *d_totalVisitedNode, int *d_TotalMovementInXCoord, int *d_TotalMovementInYCoord, int V) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < V && d_WorkListCurr[id] != -1) {
		int node = id;
		int indexAdjNodes = d_Offset[node];
		int noOfAdjNodes = d_Offset[node+1] - indexAdjNodes;
		atomicAdd(d_totalVisitedNode, 1);

		for(int i = 0; i < noOfAdjNodes; i++) {
			int adj = d_Csr[indexAdjNodes+i];
			d_WorkListNew[adj] = 0;
			d_TotalMovementInXCoord[adj] += d_TotalMovementInXCoord[node];
			d_TotalMovementInYCoord[adj] += d_TotalMovementInYCoord[node]; 
		}
	}
}

__global__
void finalPositionOfMeshes(int *d_TotalMovementInXCoord, int *d_TotalMovementInYCoord, int *d_GlobalCoordinatesX, int *d_GlobalCoordinatesY, int V) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < V) {
		d_TotalMovementInXCoord[id] += d_GlobalCoordinatesX[id];
		d_TotalMovementInYCoord[id] += d_GlobalCoordinatesY[id];
	}
}

__global__
void computeSceneOpacity(int *d_sceneOpacity, int *d_TotalMovementInXCoord, int *d_TotalMovementInYCoord, int *d_FrameSizeX, int *d_FrameSizeY, int *d_Opacity, int sceneSizeX, int sceneSizeY) {
	int meshId = blockIdx.x;
	int meshXCoord = blockIdx.y;
	int meshYCoord = threadIdx.x;

	if(meshXCoord < d_FrameSizeX[meshId] && meshYCoord < d_FrameSizeY[meshId]) {
		int xPosInScene = meshXCoord + d_TotalMovementInXCoord[meshId];
		int yPosInScene = meshYCoord + d_TotalMovementInYCoord[meshId];

		if(xPosInScene >= 0 && xPosInScene < sceneSizeX && yPosInScene >= 0 && yPosInScene < sceneSizeY) {
			atomicMax(&d_sceneOpacity[xPosInScene*sceneSizeY + yPosInScene], d_Opacity[meshId]);
		}
	}
}

__global__
void computeFinalPNG(int *d_finalPNG, int *d_sceneOpacity, int *d_TotalMovementInXCoord, int *d_TotalMovementInYCoord, int *d_FrameSizeX, int *d_FrameSizeY, int **d_Mesh, int *d_Opacity, int sceneSizeX, int sceneSizeY) {
	int meshId = blockIdx.x;
	int meshXCoord = blockIdx.y;
	int meshYCoord = threadIdx.x;

	if(meshXCoord < d_FrameSizeX[meshId] && meshYCoord < d_FrameSizeY[meshId]) {
		int xPosInScene = meshXCoord + d_TotalMovementInXCoord[meshId];
		int yPosInScene = meshYCoord + d_TotalMovementInYCoord[meshId];

		if(xPosInScene >= 0 && xPosInScene < sceneSizeX && yPosInScene >= 0 && yPosInScene < sceneSizeY) {
			if(d_Opacity[meshId] == d_sceneOpacity[xPosInScene*sceneSizeY + yPosInScene]) {
				int *mesh = d_Mesh[meshId];
				d_finalPNG[xPosInScene*sceneSizeY + yPosInScene] = mesh[meshXCoord*d_FrameSizeY[meshId] + meshYCoord];
			}
		}
	}
}



void copyTranslationFromHostToDevice(int *device_desination, std::vector<std::vector<int>> &translations, int meshCommandAmountFlag) {
	int numOfTranslation = translations.size();
    int *arr = new int[numOfTranslation];
	for(int i = 0; i < numOfTranslation; i++) arr[i] = translations[i][meshCommandAmountFlag];

	cudaMemcpy(device_desination, arr, numOfTranslation*sizeof(int), cudaMemcpyHostToDevice);
    delete arr;
}

void copyMesheshFromHostToDevice(int **hMesh, int **d_Mesh, int *hFrameSizeX, int *hFrameSizeY, int V) {
	int **meshPtrs = new int*[V];
    
	for(int i = 0; i < V; i++) {
		cudaMalloc(&meshPtrs[i], hFrameSizeX[i]*hFrameSizeY[i]*sizeof(int));
		cudaMemcpy(meshPtrs[i], hMesh[i], hFrameSizeX[i]*hFrameSizeY[i]*sizeof(int), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_Mesh, meshPtrs, V*sizeof(int*), cudaMemcpyHostToDevice);
    delete meshPtrs;
}



void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}

void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}



int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;  
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;

	// Code begins here.
		
	int *d_TotalMovementInXCoord;
	int *d_TotalMovementInYCoord;

	cudaMalloc(&d_TotalMovementInXCoord, V*sizeof(int));
	cudaMalloc(&d_TotalMovementInYCoord, V*sizeof(int));
	cudaMemset(d_TotalMovementInXCoord, 0, V*sizeof(int));
	cudaMemset(d_TotalMovementInYCoord, 0, V*sizeof(int));



	int *d_TranslationMesh;
	int *d_TranslationCommand;
	int *d_TranslationAmount;
	
	cudaMalloc(&d_TranslationMesh, numTranslations*sizeof(int));
	cudaMalloc(&d_TranslationCommand, numTranslations*sizeof(int));
	cudaMalloc(&d_TranslationAmount, numTranslations*sizeof(int));
	copyTranslationFromHostToDevice(d_TranslationMesh, translations, 0);
	copyTranslationFromHostToDevice(d_TranslationCommand, translations, 1);
	copyTranslationFromHostToDevice(d_TranslationAmount, translations, 2);

	applyTranslation<<<int(ceil(numTranslations / 1024.0)), 1024>>>(numTranslations, d_TranslationMesh, d_TranslationCommand, d_TranslationAmount, d_TotalMovementInXCoord, d_TotalMovementInYCoord);
	cudaDeviceSynchronize();

	cudaFree(d_TranslationMesh);
	cudaFree(d_TranslationCommand);
	cudaFree(d_TranslationAmount);



	int *d_WorkListCurr;
	int *d_WorkListNew;
	int *d_totalVisitedNode;
	int *d_Offset;
	int *d_Csr;
	
	cudaMalloc(&d_WorkListCurr, V*sizeof(int));
	cudaMalloc(&d_WorkListNew, V*sizeof(int));
	cudaMalloc(&d_totalVisitedNode, sizeof(int));
	cudaMalloc(&d_Offset, (V+1)*sizeof(int));
	cudaMalloc(&d_Csr, E*sizeof(int));
	cudaMemset(d_WorkListCurr, -1, V*sizeof(int));
	cudaMemset(d_WorkListNew, -1, V*sizeof(int));
	cudaMemset(d_WorkListCurr, 0, sizeof(int));
	cudaMemset(d_totalVisitedNode, 0, sizeof(int));
	cudaMemcpy(d_Offset, hOffset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Csr, hCsr, E*sizeof(int), cudaMemcpyHostToDevice);
	
	while(true) {
		applyTransitiveTranslation<<<int(ceil(V / 1024.0)), 1024>>>(d_Offset, d_Csr, d_WorkListCurr, d_WorkListNew, d_totalVisitedNode, d_TotalMovementInXCoord, d_TotalMovementInYCoord, V);
		cudaDeviceSynchronize();
		
		std::swap(d_WorkListCurr, d_WorkListNew);
		
		int totalVisitedNode;
		cudaMemcpy(&totalVisitedNode, d_totalVisitedNode, sizeof(int), cudaMemcpyDeviceToHost);
		if(totalVisitedNode == V) break;
		else cudaMemset(d_WorkListNew, -1, V*sizeof(int));
	}

	cudaFree(d_WorkListCurr);
	cudaFree(d_WorkListNew);
	cudaFree(d_totalVisitedNode);
	cudaFree(d_Offset);
	cudaFree(d_Csr);



	int *d_GlobalCoordinatesX;
	int *d_GlobalCoordinatesY;

	cudaMalloc(&d_GlobalCoordinatesX, V*sizeof(int));
	cudaMalloc(&d_GlobalCoordinatesY, V*sizeof(int));
	cudaMemcpy(d_GlobalCoordinatesX, hGlobalCoordinatesX, V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_GlobalCoordinatesY, hGlobalCoordinatesY, V*sizeof(int), cudaMemcpyHostToDevice);

	finalPositionOfMeshes<<<int(ceil(V / 1024.0)), 1024>>>(d_TotalMovementInXCoord, d_TotalMovementInYCoord, d_GlobalCoordinatesX, d_GlobalCoordinatesY, V);
	cudaDeviceSynchronize();

	cudaFree(d_GlobalCoordinatesX);
	cudaFree(d_GlobalCoordinatesY);



	int *d_finalPNG;
	int *d_sceneOpacity;
	int *d_FrameSizeX;
	int *d_FrameSizeY;
	int *d_Opacity;
	int **d_Mesh;

	cudaMalloc(&d_finalPNG, frameSizeX*frameSizeY*sizeof(int));
	cudaMalloc(&d_sceneOpacity, frameSizeX*frameSizeY*sizeof(int));
	cudaMalloc(&d_FrameSizeX, V*sizeof(int));
	cudaMalloc(&d_FrameSizeY, V*sizeof(int));
	cudaMalloc(&d_Opacity, V*sizeof(int));
	cudaMalloc(&d_Mesh, V*sizeof(int*));
	cudaMemset(d_finalPNG, 0, frameSizeX*frameSizeY*sizeof(int));
	cudaMemset(d_sceneOpacity, -1, frameSizeX*frameSizeY*sizeof(int));
	cudaMemcpy(d_FrameSizeX, hFrameSizeX, V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FrameSizeY, hFrameSizeY, V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Opacity, hOpacity, V*sizeof(int), cudaMemcpyHostToDevice);
	copyMesheshFromHostToDevice(hMesh, d_Mesh, hFrameSizeX, hFrameSizeY, V);

	computeSceneOpacity<<<dim3(V, 100, 1), 100>>>(d_sceneOpacity, d_TotalMovementInXCoord, d_TotalMovementInYCoord, d_FrameSizeX, d_FrameSizeY, d_Opacity, frameSizeX, frameSizeY);

	computeFinalPNG<<<dim3(V, 100, 1), 100>>>(d_finalPNG, d_sceneOpacity, d_TotalMovementInXCoord, d_TotalMovementInYCoord, d_FrameSizeX, d_FrameSizeY, d_Mesh, d_Opacity, frameSizeX, frameSizeY);
	
	cudaDeviceSynchronize();
	cudaMemcpy(hFinalPng, d_finalPNG, frameSizeX*frameSizeY*sizeof(int), cudaMemcpyDeviceToHost);

	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
