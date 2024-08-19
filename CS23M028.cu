/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include <vector>
#include <stack>

#include<iostream>
class Node {
public:
    int val;
    Node* next;

    Node(int v) : val(v), next(nullptr) {}
};




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

int call(int ind,int* hOffset,int* hCsr,std::vector<int>&subtree_nodes_count)
{
  int res = 1;
  for(int i = 0 ; i < *(hOffset+ind+1) - *(hOffset+ind) ;i++)
  {
    res+=call(*(hCsr+*(hOffset+ind)+i),hOffset,hCsr,subtree_nodes_count);
  }
  subtree_nodes_count[ind] = res;
  return res;	 
}



std::vector<int> flatten(const std::vector<std::vector<int>>& vec) {
    std::vector<int> flattened;
    for (const auto& inner_vec : vec) {
        flattened.insert(flattened.end(), inner_vec.begin(), inner_vec.end());
    }
    return flattened;
}

__global__ void scenceGen(int V,int frameSizeX,int frameSizeY,int* gpuScene,int* gpuhOpacity,int* gpuhFrameSizeX,int* gpuhFrameSizeY,int* gpuX,int* gpuY,int** mesh_matrix)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(id < frameSizeX*frameSizeY)
	{
		int row = id/frameSizeY;
		int col = id % frameSizeY;
    int data = 0;
		int localOpacity = 0;
		for(int iter = 0; iter < V;iter++)
		{
			if(*(gpuX+iter) <= row && *(gpuY+iter) <= col && (*(gpuX+iter) + *(gpuhFrameSizeX+iter)) > row && (*(gpuY+iter) + *(gpuhFrameSizeY+iter)) > col )
			{
        if(*(gpuhOpacity+iter) > localOpacity )
        {
          localOpacity = *(gpuhOpacity+iter);
          data = *(*(mesh_matrix+iter) + (row - *(gpuX+iter))* (*(gpuhFrameSizeY+iter)) + (col-*(gpuY+iter)));
        }
			
			}
		}
		*(gpuScene+id) = data;
		
	}
}

__global__ void translateMeshes(int V,int T,int* gpuX,int*gpuY,int* gpu_translations,int*gpu_nodeCount,int*gpuMap,int* gpuPreOrder)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < T)
	{
		int root,dir,step,node;
		root = *(gpu_translations + id*3);
		dir = *(gpu_translations + id*3 +1);
		step = *(gpu_translations + id*3 + 2);
		int ind = *(gpuMap+root);
		for(int iter = 0; iter < *(gpu_nodeCount+root); iter++)
		{
			
			node = *(gpuPreOrder+ind+iter);
			if(dir == 0)
			{
				atomicAdd(gpuX + node, -1*step);
			}
			else if(dir == 1)
			{
				atomicAdd(gpuX + node, step);
			}
			else if(dir == 2)
			{
				atomicAdd(gpuY + node, -1*step);
			}
			else
			{
				atomicAdd(gpuY + node, step);
				
			}
		}
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

	auto start = std::chrono::high_resolution_clock::now ();

	// Code begins here.
	// Do not change anything above this comment.
	
    int* gpuScene;
  	cudaMalloc(&gpuScene,  frameSizeX * frameSizeY * sizeof(int));

	int blocks1 = ceil( frameSizeX * frameSizeY / 1024.0);


	int* gpuhOpacity;
    cudaMalloc(&gpuhOpacity,  V* sizeof(int));

	int *gpuhFrameSizeX;
	int *gpuhFrameSizeY;
	cudaMalloc(&gpuhFrameSizeX,  V* sizeof(int));
	cudaMalloc(&gpuhFrameSizeY,  V* sizeof(int));

	cudaMemcpy(gpuhOpacity, hOpacity,V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuhFrameSizeX, hFrameSizeX,V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuhFrameSizeY, hFrameSizeY,V * sizeof(int), cudaMemcpyHostToDevice);

	int** mesh_matrix;
	cudaMalloc(&mesh_matrix, V * sizeof(int*)); 
	for (int i = 0; i < V; ++i) {
		int* temp_mesh_mat;
		cudaMalloc(&temp_mesh_mat, *(hFrameSizeX+i) * *(hFrameSizeY+i) * sizeof(int)); 
		cudaMemcpy(temp_mesh_mat, hMesh[i], *(hFrameSizeX+i) * *(hFrameSizeY+i)  * sizeof(int), cudaMemcpyHostToDevice); 
		cudaMemcpy(mesh_matrix + i, &temp_mesh_mat,sizeof(int*), cudaMemcpyHostToDevice); 
	}
	
    int* gpuX;
    int* gpuY;

    cudaMalloc(&gpuX, V * sizeof(int));
    cudaMalloc(&gpuY, V * sizeof(int));

    cudaMemcpy(gpuX, hGlobalCoordinatesX, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuY, hGlobalCoordinatesY, V * sizeof(int), cudaMemcpyHostToDevice);


	std::vector<int> flattened = flatten(translations);
    int* gpu_translations;
    cudaMalloc(&gpu_translations, flattened.size() * sizeof(int));

    cudaMemcpy(gpu_translations, flattened.data(), flattened.size() * sizeof(int), cudaMemcpyHostToDevice);

	int blocks = ceil(numTranslations / 1024.0);
	
  
	int* preOrder = (int*)malloc(sizeof(int)*V);
	int* map = (int*)malloc(sizeof(int)*V);

	int* computationOffset = (int*)malloc(sizeof(int)*V);

	int counter = 0;
	std::stack<int> s;
	int top;
	
	s.push(0);
	while(!s.empty())
	{
		top = s.top();
		s.pop();
		*(preOrder+counter) = top;
		counter++;
		for(int i = 0 ; i < *(hOffset+top+1) - *(hOffset+top) ;i++)
		{
			s.push(*(hCsr+*(hOffset+top)+i));
		}
	}

	for(int iter = 0; iter < V; iter++)
	{
		*(map+*(preOrder+iter)) = iter;
	}

  
	int n = V;  // Number of vertices
   	std::vector<int> subtree_nodes_count(n,0);
   	call(0,hOffset,hCsr,subtree_nodes_count);
    

	int* gpuMap;
	cudaMalloc(&gpuMap,V*sizeof(int));
	cudaMemcpy(gpuMap,map,V*sizeof(int),cudaMemcpyHostToDevice);

	int* gpuPreOrder;
	cudaMalloc(&gpuPreOrder,V*sizeof(int));
	cudaMemcpy(gpuPreOrder,preOrder,V*sizeof(int),cudaMemcpyHostToDevice);


    int* gpu_nodeCount;
    cudaMalloc(&gpu_nodeCount, V * sizeof(int));

    cudaMemcpy(gpu_nodeCount, subtree_nodes_count.data(), V * sizeof(int), cudaMemcpyHostToDevice);

	translateMeshes<<<blocks,1024>>>(V,numTranslations,gpuX,gpuY,gpu_translations,gpu_nodeCount,gpuMap,gpuPreOrder);
	cudaFree(gpu_translations);
	cudaFree(gpuPreOrder);
	cudaFree(gpuMap);
	free(computationOffset);
	computationOffset = NULL;
	free(map);
	map = NULL;
  	scenceGen<<<blocks1,1024>>>(V,frameSizeX,frameSizeY,gpuScene,gpuhOpacity,gpuhFrameSizeX,gpuhFrameSizeY,gpuX,gpuY,mesh_matrix);

	cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
	cudaDeviceSynchronize();
	cudaMemcpy(hFinalPng,gpuScene,frameSizeX*frameSizeY*sizeof(int),cudaMemcpyDeviceToHost);
	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
