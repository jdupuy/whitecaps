/*BMPLoader - loads Microsoft .bmp format
    Copyright (C) 2006  Chris Backhouse

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


  cjbackhouse@hotmail.com 		www.backhouse.tk
  
  I would appreciate it if anyone using this in something cool would tell me
  so I can see where it ends up.

  Takes a filename, returns an array of RGB pixel data
  Loads:
  24bit bitmaps
  256 colour bitmaps
  16 colour bitmaps
  2 colour bitmaps  (Thanks to Charles Rabier)

  This code is designed for use in openGL programs, so bitmaps not correctly padded will not
  load properly, I believe this only applies to: 
  256cols if width is not a multiple of 4
  16cols if width is not a multiple of 8
  2cols if width is not a multiple of 32

  Sample code:

	BMPClass bmp;
	BMPLoad(fname,bmp);
	glTexImage2D(GL_TEXTURE_2D,0,3,bmp.width,bmp.height,0,GL_RGB,GL_UNSIGNED_BYTE,bmp.bytes);
*/

#ifndef BMPLOADER_H
#define BMPLOADER_H

#include <string>
typedef unsigned char BYTE;

class BMPClass
{
public:
	BMPClass();
	~BMPClass();
	BYTE& pixel(int x,int y,int c);
	void allocateMem();
	int width,height;
	BYTE* bytes;			//OpenGL formatted pixels
};

#define BMPError char
#define BMPNOTABITMAP 'b'	//Possible error flags
#define BMPNOOPEN 'o'
#define BMPFILEERROR 'f'
#define BMPBADINT 'i'
#define BMPNOERROR '\0'
#define BMPUNKNOWNFORMAT 'u'

//Loads the bmp in fname, and puts the data in bmp
BMPError BMPLoad(std::string fname,BMPClass& bmp);

//Translates my error codes into English	
std::string TranslateBMPError(BMPError err);	

//Load and select in OpenGL
BMPError BMPLoadGL(std::string fname);

#endif
