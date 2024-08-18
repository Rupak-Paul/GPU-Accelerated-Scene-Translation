#ifndef _RENDERER
#define _RENDERER

#include "SceneNode.h"
#include <vector>
#include <map>
#include <cstdlib>

class Renderer {
	
	private:
	int num_nodes ;
	int num_edges ;
	int *hOffset ;
	int *hCsr ;
	int *hOpacity ;
	int **hMesh ;
	int *hGlobalCoordinatesX ;
	int *hGlobalCoordinatesY ;
	int *hFrameSizeX ;
	int *hFrameSizeY ;
	std::vector<std::vector<int> > graph ;
	std::map<int, SceneNode*> sceneMap ;

	public:
	Renderer (int &num_nodes) ;
	Renderer (std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges) ;
	void add_edge (SceneNode *node_u, SceneNode *node_v) ;
	void make_csr () ;
	int getNumNodes () ;
	int *get_h_offset () ;
	int *get_h_csr () ;
	int *get_opacity () ;
	int **get_mesh_csr () ;
	int *getGlobalCoordinatesX () ;
	int *getGlobalCoordinatesY () ;
	int *getFrameSizeX () ;
	int *getFrameSizeY () ;
	void populateMap (std::vector<SceneNode*> &scenes) ;
	void addEdgeById (std::vector<int> &edge) ;
	SceneNode* extractScene (int &nodeId) ;
} ;
#endif
