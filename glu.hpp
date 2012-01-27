////////////////////////////////////////////////
// \author 	Jonathan DUPUY
// \file 	glu.hpp
// \brief 	Cross platform OpenGL utility functions
////////////////////////////////////////////////

#include "GL/glew.h"
#include <string>

namespace glu
{
	// Create a Shader object, returns 0 if error
	// Content can be added in the shader source code using the additionnalContent variable
	// (content will be added right after the version declaration, or at the beginning of the source
	// if no version declaration was found)
	GLuint createShader(GLenum shaderType, const std::string& filepath, const std::string& additionnalContent = "");


	// Validate a program, returns GL_FALSE and prints Log content if error, GL_TRUE otherwise
	GLuint validateProgram(GLuint program);


	// Specify a buffer offset in bytes
	// (use with glVertexAttribPointer)
	const GLvoid* bufferOffset(GLuint offset);


	// Convert GL Error to string
	std::string errorString(GLenum error);


	// Get platform information
	// (ie GL_VENDOR and GL_RENDERER)
	// Returns a string with the form :
	// GL_VENDOR : xxx \n
	// GL_RENDERER : xxx \n
	std::string platformInfoString();


	// Get Context information
	// (ie GL_VERSION and GL_SHADING_LANGUAGE_VERSION)
	// Returns a string with the form :
	// GL_VERSION : xxx \n
	// GL_SHADING_LANGUAGE_VERSION : xxx \n
	std::string contextInfoString();


	// Get Information on Uniform Block of an active program
	std::string uniformBlockInfoString(GLuint program, const std::string& uniformBlockName);
}
