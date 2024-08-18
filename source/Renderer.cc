#include "Renderer.h"

Renderer::Renderer (int &num_nodes) {
	this->num_nodes = num_nodes ;
	this->num_edges = 0 ;
	graph.resize (num_nodes) ;
}

Renderer::Renderer (std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges) {
	this->num_nodes = scenes.size () ;
	this->num_edges = edges.size () ;
	graph.resize (this->num_nodes) ;
	for (auto &edge:edges) {
		addEdgeById (edge) ;
	}
	populateMap (scenes) ;
} 

int Renderer::getNumNodes () {
	return num_nodes ;
}

void Renderer::addEdgeById (std::vector<int> &edge) {
	int u = edge[0] ;
	int v = edge[1] ;
	graph[u].push_back (v) ;
}

void Renderer::populateMap (std::vector<SceneNode*> &scenes) {
	int id = 0 ;
	for (auto &scene:scenes) {
		this->sceneMap[id++] = scene ;
	}
}

void Renderer::add_edge (SceneNode *node_u, SceneNode *node_v) {
	int uId = node_u->getNodeId() ;
	int vId = node_v->getNodeId() ;
	if (this->sceneMap.find (uId) == sceneMap.end () ) {
		sceneMap[uId]=(node_u) ;
	}
	if (this->sceneMap.find (vId)== sceneMap.end () ) {
		sceneMap[vId] = node_v ;	
	}
	graph[uId].push_back (vId) ;
	this->num_edges++ ;
}

int* Renderer::get_h_offset () {
	return this->hOffset;
}

int* Renderer::get_h_csr () {
	return this->hCsr ;
}

int* Renderer::get_opacity () {
	return this->hOpacity ;
}

int** Renderer::get_mesh_csr () {
	return this->hMesh ;
}

int * Renderer::getGlobalCoordinatesX () {
	return hGlobalCoordinatesX ;
} ;
int * Renderer::getGlobalCoordinatesY () {
	return hGlobalCoordinatesY ;
} ;
int * Renderer::getFrameSizeX () {
	return hFrameSizeX ;
} ;
int * Renderer::getFrameSizeY () {
	return hFrameSizeY ;
} ;

SceneNode* Renderer::extractScene (int &nodeId) {
	return this->sceneMap[nodeId] ;
}


void Renderer::make_csr () {

//	printf ("Inside Make CSR\n") ;

	// Print the graph once 
	/*for (int i=0; i<this->num_nodes; i++) {
		printf ("%d : ", i) ;
		for (int j=0; j<this->graph[i].size (); j++) {
			printf (" %d ", graph[i][j]) ;	
		}
		printf ("\n") ;
	}*/
	
	// Node properties.
	this->hOffset = (int*) malloc (sizeof (int) * (this->num_nodes+1)) ;
	this->hOpacity = (int*) malloc (sizeof (int) * this->num_nodes) ;
	this->hMesh = (int**) malloc (sizeof (int**) * this->num_nodes) ;
	this->hGlobalCoordinatesX = (int*) malloc (sizeof(int)* this->num_nodes) ;
	this->hGlobalCoordinatesY = (int*) malloc (sizeof(int)* this->num_nodes) ;
	this->hFrameSizeX = (int*) malloc (sizeof(int)* this->num_nodes) ;
	this->hFrameSizeY = (int*) malloc (sizeof(int)* this->num_nodes) ;

	// Fill up Node Properties.
	for (int nodeId = 0 ; nodeId < this->num_nodes; nodeId++) {

		SceneNode* currentScene = this->sceneMap[nodeId] ;
	 	this->hMesh[nodeId] = currentScene->getMesh () ;	
		this->hGlobalCoordinatesX[nodeId] = currentScene->getGlobalPositionX () ;
		this->hGlobalCoordinatesY[nodeId] = currentScene->getGlobalPositionY () ;
		this->hFrameSizeX[nodeId] = currentScene->getFrameSizeX () ;
		this->hFrameSizeY[nodeId] = currentScene->getFrameSizeY () ;
		this->hOpacity[nodeId] = currentScene->getOpacity () ;
	}

	// Edge Properties.
	hCsr = (int*) malloc (sizeof (int) * this->num_edges) ;

	// Making of the CSR 
	int offset_helper = 0 ;
	for (int i=0; i<this->num_nodes; i++) {
		this->hOffset[i] = offset_helper ;
		for (int j=0; j<(int)graph[i].size (); j++) {
			this->hCsr[offset_helper]=graph[i][j] ;	
			offset_helper++ ;
		}
	}
	hOffset[num_nodes]=offset_helper ;
}
