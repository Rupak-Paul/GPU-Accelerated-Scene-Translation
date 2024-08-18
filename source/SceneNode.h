#ifndef _SCENE_NODE
#define _SCENE_NODE

#include <vector>

class SceneNode {

	private :
	
	int id ;
	int globalPositionX, globalPositionY ;
	int localPositionX, localPositionY ;
	int* mesh ;
	int meshX ;
	int meshY ;
	int opacity ;
	std::vector<SceneNode*> children ;

	public :
	
	SceneNode (int* mesh, int meshX, int meshY) ;
	SceneNode (const int &id, int* mesh, const int &meshX, const int &meshY, const int &globalPosition, const int &localPosition, const int &opacity) ;

	int getNodeId () ;
	int getGlobalPositionX () ;
	int getGlobalPositionY () ;
	int getFrameSizeX () ;
	int getFrameSizeY () ;
	void setGlobalPosition (const int &globalPositionX, const int &globalPositionY) ;
	void rotateMesh (const int &amount) ;
	void setLocalPosition (int localPosition) ;
	int* getMesh () ;
	int getOpacity () ;
	void addChildren (SceneNode* child) ;
} ;
#endif
