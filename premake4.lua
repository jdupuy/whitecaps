solution "OpenGL"
	configurations {
	"debug",
	"release"
	}
	platforms { "x64", "x32" }

-- ---------------------------------------------------------
-- Project 
	project "demo"
		basedir "./"
		language "C++"
		location "./"
		kind "ConsoleApp" -- Shouldn't this be in configuration section ?
		files { "*.hpp", "*.cpp" }
		files { "core/*.cpp" }
		includedirs {
		"include",
		"core"
		}
		objdir "obj"

-- Debug configurations
		configuration {"debug"}
			defines {"DEBUG"}
			flags {"Symbols", "ExtraWarnings"}

-- Release configurations
		configuration {"release"}
			defines {"NDEBUG"}
			flags {"Optimize"}

-- Linux x86 platform gmake
		configuration {"linux", "gmake", "x32"}
			linkoptions {
			"-Wl,-rpath,./lib/linux/lin32 -L./lib/linux/lin32 -lGLEW -lglut -lAntTweakBar"
			}
			libdirs {
			"lib/linux/lin32"
			}

-- Linux x64 platform gmake
		configuration {"linux", "gmake", "x64"}
			linkoptions {
			"-Wl,-rpath,./lib/linux/lin64 -L./lib/linux/lin64 -lGLEW -lglut -lAntTweakBar"
			}
			libdirs {
			"lib/linux/lin64"
			}

-- Visual x86
		configuration {"vs2010", "x32"}
			libdirs {
			"lib/windows/win32"
			}
			links {
			"glew32s",
			"freeglut",
			"AntTweakBar"
			}

---- Visual x64
--		configuration {"vs2010", "x64"}
--			links {
--			"glew32s",
--			"freeglut",
--			}
--			libdirs {
--			"lib/windows/win64"
--			}


