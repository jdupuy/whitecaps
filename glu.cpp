#include "glu.hpp"
#include <iostream>
#include <sstream>
#include <fstream>

namespace glu
{

/////////////////////////////////////////////
// Helper Functions
namespace priv
{
	/////////////////////////////////////////////
	// Print Program Log
	static void printProgramLog(GLuint program)
	{
		GLint logLength = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
		if(logLength > 0)
		{
			GLchar* logContent = new GLchar[logLength];
			glGetProgramInfoLog(program, logLength, NULL, logContent);
			std::cerr << logContent << std::endl;
			delete[] logContent;
		}
		else
			std::cerr << "Empty" << std::endl;
	}


	/////////////////////////////////////////////
	// GLSL Types to string
	static std::string glslTypeNameTokenString(GLenum type)
	{
		switch(type)
		{
			case GL_FLOAT:
				return "float";
			case GL_FLOAT_VEC2:
				return "vec2";
			case GL_FLOAT_VEC3:
				return "vec3";
			case GL_FLOAT_VEC4:
				return "vec4";

			case GL_DOUBLE:
				return "double";
			case GL_DOUBLE_VEC2:
				return "dvec2";
			case GL_DOUBLE_VEC3:
				return "dvec3";
			case GL_DOUBLE_VEC4:
				return "dvec4";

			case GL_INT:
				return "int";
			case GL_INT_VEC2:
				return "ivec2";
			case GL_INT_VEC3:
				return "ivec3";
			case GL_INT_VEC4:
				return "ivec4";

			case GL_UNSIGNED_INT:
				return "unsigned int";
			case GL_UNSIGNED_INT_VEC2:
				return "uvec2";
			case GL_UNSIGNED_INT_VEC3:
				return "uvec3";
			case GL_UNSIGNED_INT_VEC4:
				return "uvec4";

			case GL_BOOL:
				return "bool";
			case GL_BOOL_VEC2:
				return "bvec2";
			case GL_BOOL_VEC3:
				return "bvec3";
			case GL_BOOL_VEC4:
				return "bvec4";

			case GL_FLOAT_MAT2:
				return "mat2";
			case GL_FLOAT_MAT3:
				return "mat3";
			case GL_FLOAT_MAT4:
				return "mat4";
			case GL_FLOAT_MAT2x3:
				return "mat2x3";
			case GL_FLOAT_MAT2x4:
				return "mat2x4";
			case GL_FLOAT_MAT3x2:
				return "mat3x2";
			case GL_FLOAT_MAT3x4:
				return "mat3x4";
			case GL_FLOAT_MAT4x2:
				return "mat4x2";
			case GL_FLOAT_MAT4x3:
				return "mat4x3";

			case GL_DOUBLE_MAT2:
				return "dmat2";
			case GL_DOUBLE_MAT3:
				return "dmat3";
			case GL_DOUBLE_MAT4:
				return "dmat4";
			case GL_DOUBLE_MAT2x3:
				return "dmat2x3";
			case GL_DOUBLE_MAT2x4:
				return "dmat2x4";
			case GL_DOUBLE_MAT3x2:
				return "dmat3x2";
			case GL_DOUBLE_MAT3x4:
				return "dmat3x4";
			case GL_DOUBLE_MAT4x2:
				return "dmat4x2";
			case GL_DOUBLE_MAT4x3:
				return "dmat4x3";

			case GL_SAMPLER_1D:
				return "sampler1D";
			case GL_SAMPLER_2D:
				return "sampler2D";
			case GL_SAMPLER_3D:
				return "sampler3D";
			case GL_SAMPLER_CUBE:
				return "samplerCube";
			case GL_SAMPLER_1D_SHADOW:
				return "sampler1DShadow";
			case GL_SAMPLER_2D_SHADOW:
				return "sampler2DShadow";
			case GL_SAMPLER_1D_ARRAY:
				return "sampler1DArray";
			case GL_SAMPLER_2D_ARRAY:
				return "sampler2DArray";
			case GL_SAMPLER_CUBE_MAP_ARRAY:
				return "samplerCubeMapArray";
			case GL_SAMPLER_1D_ARRAY_SHADOW:
				return "sampler1DArrayShadow";
			case GL_SAMPLER_2D_ARRAY_SHADOW:
				return "sampler2DArrayShadow";
			case GL_SAMPLER_2D_MULTISAMPLE:
				return "sampler2DMS";
			case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
				return "sampler2DMSArray";
			case GL_SAMPLER_CUBE_SHADOW:
				return "samplerCubeShadow";
			case GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW:
				return "samplerCubeMapArrayShadow";
			case GL_SAMPLER_BUFFER:
				return "samplerBuffer";
			case GL_SAMPLER_2D_RECT:
				return "sampler2DRect";
			case GL_SAMPLER_2D_RECT_SHADOW:
				return "sampler2DRectShadow";

			case GL_INT_SAMPLER_1D:
				return "isampler1D";
			case GL_INT_SAMPLER_2D:
				return "isampler2D";
			case GL_INT_SAMPLER_3D:
				return "isampler3D";
			case GL_INT_SAMPLER_CUBE:
				return "isamplerCube";
			case GL_INT_SAMPLER_1D_ARRAY:
				return "isampler1DArray";
			case GL_INT_SAMPLER_2D_ARRAY:
				return "isampler2DArray";
			case GL_INT_SAMPLER_CUBE_MAP_ARRAY:
				return "isamplerCubeMapArray";
			case GL_INT_SAMPLER_2D_MULTISAMPLE:
				return "isampler2DMS";
			case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
				return "isampler2DMSArray";
			case GL_INT_SAMPLER_BUFFER:
				return "isamplerBuffer";
			case GL_INT_SAMPLER_2D_RECT:
				return "isampler2DRect";

			case GL_UNSIGNED_INT_SAMPLER_1D:
				return "usampler1D";
			case GL_UNSIGNED_INT_SAMPLER_2D:
				return "usampler2D";
			case GL_UNSIGNED_INT_SAMPLER_3D:
				return "usampler3D";
			case GL_UNSIGNED_INT_SAMPLER_CUBE:
				return "usamplerCube";
			case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
				return "usampler1DArray";
			case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
				return "usampler2DArray";
			case GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY:
				return "usamplerCubeMapArray";
			case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
				return "usampler2DMS";
			case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
				return "usampler2DMSArray";
			case GL_UNSIGNED_INT_SAMPLER_BUFFER:
				return "usamplerBuffer";
			case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
				return "usampler2DRect";

			default:
				return "unknown?";
		}
	}

}	// namespace priv


/////////////////////////////////////////////
// Create Shader
GLuint createShader(GLenum shaderType, const std::string& filepath, const std::string& additionnalContent)
{
	// Create Shader Object
	GLuint shader = glCreateShader(shaderType);
	if(shader == 0)
	{
		std::cerr << "GL could not create Shader Object for " << filepath << std::endl;
		return 0;
	}

	// Load file and check
	std::ifstream file;
	file.open(filepath.c_str());
	if(file.fail())
	{
		std::cerr << "Could not open source file "<< filepath << std::endl;
		glDeleteShader(shader);
		return 0;
	}

	// Extract file content and add optionnal content
	// (after version specification)
	std::string shaderSource;
	std::string line;
	bool isVersionSpecified = false;
	while(getline(file, line))
	{
		shaderSource += line + "\n";
		if(line.find("#version") != std::string::npos)	// assumes code does not have a commented version declaration expression
		{
			isVersionSpecified = true;
			shaderSource += "\n" + additionnalContent + "\n";
		}
	}
	if(!isVersionSpecified)
	{
		shaderSource = additionnalContent + "\n" + shaderSource;
		std::cerr << "WARNING Undeclared language version in " << filepath << std::endl;
	}

	// Close file
	file.close();

	// Set Shader source and compile
	const GLchar* data = shaderSource.c_str();
	glShaderSource(shader, 1, &data, NULL);
	glCompileShader(shader);

	// Check compilation status
	GLint isCompiled = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
	if(isCompiled == GL_TRUE)
		return shader;

	// Print Compile Log and Source Code and return 0;
	GLint logLength = 0;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
	std::cerr 	<< "Shader compilation failed!\n"
				<< "Compiler output (in "
				<< filepath
				<< ") :\n";
	if(logLength > 0)
	{
		GLchar* logContent = new GLchar[logLength];
		glGetShaderInfoLog(shader, logLength, NULL, logContent);
		std::cerr << logContent << std::endl;
		delete[] logContent;
	}
	else
		std::cerr << "Empty" << std::endl;

	// Delete shader and return
	glDeleteShader(shader);

	return 0;
}

/////////////////////////////////////////////
// Validate Program
GLuint validateProgram(GLuint program)
{
	// Check if program is linked
	GLint isLinked;
	glGetProgramiv(program, GL_LINK_STATUS, &isLinked);

	// Check if program is valid
	glValidateProgram(program);
	GLint isValid;
	glGetProgramiv(program, GL_VALIDATE_STATUS, &isValid);

	// Check results
	if(isLinked==GL_FALSE)
	{
		std::cerr 	<< "Program is not linked !\n"
					<< "Log output :\n";
		priv::printProgramLog(program);
		return GL_FALSE;
	}

	if(isValid==GL_FALSE)
	{
		std::cerr 	<< "Program is not validated !\n"
					<< "Log output :\n";
		priv::printProgramLog(program);
		return GL_FALSE;
	}

	// Okay
	return GL_TRUE;
}


/////////////////////////////////////////////
// Set Buffer Offset
#define GLU_BUFFER_OFFSET(i) 	((char *)NULL + (i))
const GLvoid* bufferOffset(GLuint offset)
{
	return ((char *)NULL + (offset));
}


/////////////////////////////////////////////
// Get GL_ERROR as a string
std::string errorString(GLenum error)
{
	switch(error)
	{
		case GL_NO_ERROR:
			return "GL_NO_ERROR";
		case GL_INVALID_ENUM:
			return "GL_INVALID_ENUM";
		case GL_INVALID_VALUE:
			return "GL_INVALID_VALUE";
		case GL_INVALID_OPERATION:
			return "GL_INVALID_OPERATION";
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			return "GL_INVALID_FRAMEBUFFER_OPERATION";
		case GL_OUT_OF_MEMORY:
			return "GL_OUT_OF_MEMORY";
		default:
			return "Unknown";
	};
}


/////////////////////////////////////////////
// Get Platform Information
std::string platformInfoString()
{
	std::ostringstream data;
	const GLubyte* vendorInfo 	= glGetString(GL_VENDOR);
	const GLubyte* rendererInfo = glGetString(GL_RENDERER);
	data << "GL_VENDOR\t: ";
	if(vendorInfo == 0)
		data << "Unknown\n";
	else
		data << vendorInfo <<"\n";

	data << "GL_RENDERER\t: ";
	if(rendererInfo == 0)
		data << "Unknown\n";
	else
		data << rendererInfo <<"\n";

	return data.str();
}


/////////////////////////////////////////////
// Get Context Information
std::string contextInfoString()
{
	std::ostringstream data;
	const GLubyte* versionInfo 	= glGetString(GL_VERSION);
	const GLubyte* slInfo 		= glGetString(GL_SHADING_LANGUAGE_VERSION);
	data << "GL_VERSION\t: ";
	if(versionInfo == 0)
		data << "Unknown\n";
	else
		data << versionInfo <<"\n";

	data << "GL_SHADING_LANGUAGE_VERSION\t: ";
	if(slInfo == 0)
		data << "Unknown\n";
	else
		data << slInfo <<"\n";

	return data.str();
}


/////////////////////////////////////////////
// Uniform Block information
std::string uniformBlockInfoString(GLuint program, const std::string& uniformBlockName)
{
	// Fetch name
	GLuint blockIndex = glGetUniformBlockIndex(program, uniformBlockName.c_str());
	if(blockIndex == GL_INVALID_INDEX)
		return "No such active block\n";

	// Fetch active uniforms inside the block
	GLint activeUniforms = 0;
	glGetActiveUniformBlockiv(program, blockIndex, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &activeUniforms);

	// Get size of block in bytes
	GLint dataSize = 0;
	glGetActiveUniformBlockiv(program, blockIndex, GL_UNIFORM_BLOCK_DATA_SIZE, &dataSize);
}

}	// namespace glu
