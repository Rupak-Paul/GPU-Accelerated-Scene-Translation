#include "SceneNode.h"

SceneNode::SceneNode (const int &id, int *mesh, const int &meshX, const int &meshY, const int &globalPositionX, const int &globalPositionY, const int &opacity) {
	
	this->id = id ;
	this->mesh = mesh ;
	this->meshX = meshX ;
	this->meshY = meshY ;
	this->opacity = opacity ;

	setGlobalPosition (globalPositionX, globalPositionY) ;
}

int SceneNode::getNodeId () {
	return id ;
}

void SceneNode::setGlobalPosition (const int &globalPositionX, const int &globalPositionY) {
	this->globalPositionX = globalPositionX ;
	this->globalPositionY = globalPositionY ;
}

int SceneNode::getFrameSizeX () {
	return this->meshX ;
}

int SceneNode::getFrameSizeY () {
	return this->meshY ;
}

void SceneNode::rotateMesh (const int &amount) {
	//pass 
}


int SceneNode::getGlobalPositionX () {
	return this->globalPositionX ;
}

int SceneNode::getGlobalPositionY () {
	return this->globalPositionY ;
}


int* SceneNode::getMesh () {
	return this->mesh ;
}

int SceneNode::getOpacity () {
	return this->opacity ;
}

void SceneNode::addChildren (SceneNode* child) {
	this->children.push_back (child) ;	
}
