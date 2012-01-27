#extension GL_EXT_gpu_shader4 : enable

/**
 * Real-time Realistic Ocean Lighting using Seamless Transitions from Geometry to BRDF
 * Copyright (c) 2009 INRIA
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holders nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Author: Eric Bruneton
 */

varying vec2 uv;	// texcoords

#ifdef _VERTEX_
void main() {
    uv = gl_Vertex.zw;
    gl_Position = vec4(gl_Vertex.xy, 0.0, 1.0);
}

#endif

#ifdef _FRAGMENT_


#define LAYER_JACOBIAN_XX 	5.0
#define LAYER_JACOBIAN_YY	6.0
#define LAYER_JACOBIAN_XY	7.0

uniform sampler2DArray fftWavesSampler;
uniform vec4 choppy;
//uniform sampler2DArray prevfftWavesSampler;
//uniform float delta;
//uniform vec4 GRID_SIZES;

void main() {

    // fftWavesSampler has 4 height values
    // heights.r : Lx, Lz = x0
    // heights.b : Lx, Lz = x1
    // heights.b : Lx, Lz = x2
    // heights.a : Lx, Lz = x3
    // with x0 < x1 < x2 < x3
    vec4 heights = texture2DArray(fftWavesSampler, vec3(uv, 0.0));

    gl_FragData[0] = vec4(heights.x, heights.x * heights.x, heights.y, heights.y * heights.y);
    gl_FragData[1] = vec4(heights.z, heights.z * heights.z, heights.w, heights.w * heights.w);

	// store Jacobian coeff value and variance
	vec4 Jxx = choppy*texture2DArray(fftWavesSampler, vec3(uv, LAYER_JACOBIAN_XX));
	vec4 Jyy = choppy*texture2DArray(fftWavesSampler, vec3(uv, LAYER_JACOBIAN_YY));
	vec4 Jxy = choppy*texture2DArray(fftWavesSampler, vec3(uv, LAYER_JACOBIAN_XY));

	// Store Jxx values and variance
//	gl_FragData[2] = vec4(Jxx.x, Jxx.x*Jxx.x, Jxx.y, Jxx.y*Jxx.y);
//	gl_FragData[3] = vec4(Jxx.z, Jxx.z*Jxx.z, Jxx.w, Jxx.w*Jxx.w);
//
//	// Store Jyy values and variance
//	gl_FragData[4] = vec4(Jyy.x, Jyy.x*Jyy.x, Jyy.y, Jyy.y*Jyy.y);
//	gl_FragData[5] = vec4(Jyy.z, Jyy.z*Jyy.z, Jyy.w, Jyy.w*Jyy.w);
//
//	// Store Jxy values and variance
//	gl_FragData[6] = vec4(Jxy.x, Jxy.x*Jxy.x, Jxy.y, Jxy.y*Jxy.y);
//	gl_FragData[7] = vec4(Jxy.z, Jxy.z*Jxy.z, Jxy.w, Jxy.w*Jxy.w);

    // Store partial jacobians
    vec4 res = Jxx + Jyy + Jxx*Jyy - Jxy*Jxy;
    vec4 res2 = res*res;
    gl_FragData[2] = vec4(res.x, res2.x, res.y, res2.y);
    gl_FragData[3] = vec4(res.z, res2.z, res.w, res2.w);

//	// displacements at current time
//	vec4 displace1t0 = texture2DArray(fftWavesSampler, vec3(uv, 3.0));
//	vec4 displace2t0 = texture2DArray(fftWavesSampler, vec3(uv, 4.0));
//	// displacements at previous time
//	vec4 displace1t1 = texture2DArray(prevfftWavesSampler, vec3(uv, 3.0));
//	vec4 displace2t1 = texture2DArray(prevfftWavesSampler, vec3(uv, 4.0));

//	// x vel
//	gl_FragData[2] = vec4(displace1t0.xz, displace2t0.xz)*(choppy);
//    // y vel
//    gl_FragData[3] = vec4(displace1t0.yw, displace2t0.yw)*(choppy);
////    gl_FragData[3] = vec4(1.0);

//	// jacobian compute
//	float Jxx, Jxy, Jyy;
//	float Jeval;	// Minimum Eigen Value
//	vec2 u = gl_FragCoord.xy / 256.0;
//
//	// Jxx1..4 : partial Jxx
//	float Jxx1 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_XX)).r;
//	float Jxx2 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_XX)).g;
//	float Jxx3 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_XX)).b;
//	float Jxx4 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_XX)).a;
//	Jxx = 1.0 + dot((choppy), vec4(Jxx1,Jxx2,Jxx3,Jxx4));
//
//	// Jyy1..4 : partial Jyy
//	float Jyy1 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_YY)).r;
//	float Jyy2 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_YY)).g;
//	float Jyy3 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_YY)).b;
//	float Jyy4 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_YY)).a;
//	Jyy = 1.0 + dot((choppy), vec4(Jyy1,Jyy2,Jyy3,Jyy4));
//
//	// Jxy1..4 : partial Jxy
//	float Jxy1 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_XY)).r;
//	float Jxy2 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_XY)).g;
//	float Jxy3 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_XY)).b;
//	float Jxy4 = 1.0*texture2DArray(fftWavesSampler, vec3(u * GRID_SIZES.x, LAYER_JACOBIAN_XY)).a;
//	Jxy = dot((choppy),vec4(Jxy1, Jxy2, Jxy3, Jxy4));
//
//	Jeval = Jxx*Jyy - Jxy*Jxy;
//
//	float jxx, jyy, jxy;
//	jxx = Jxx1*choppy.x + 1.0;
//	jyy = Jyy1*choppy.x + 1.0;
//	jxy = Jxy1*choppy.x;
//	gl_FragData[2].r = step(jxx*jyy - jxy*jxy,0.81);
//	jxx = Jxx2*choppy.y + 1.0;
//	jyy = Jyy2*choppy.y + 1.0;
//	jxy = Jxy2*choppy.y;
//	gl_FragData[2].g = step(jxx*jyy - jxy*jxy,-0.45);
//	jxx = Jxx3*choppy.z + 1.0;
//	jyy = Jyy3*choppy.z + 1.0;
//	jxy = Jxy3*choppy.z;
//	gl_FragData[2].b = step(jxx*jyy - jxy*jxy,-0.35);
//	jxx = Jxx4*choppy.w + 1.0;
//	jyy = Jyy4*choppy.w + 1.0;
//	jxy = Jxy4*choppy.w;
//	gl_FragData[2].a = step(jxx*jyy - jxy*jxy,-0.10);



//	gl_FragData[2].r = step(Jeval, 0.5);

//	gl_FragData[2] = vec4(	smoothstep(6.064, 7.1, heights.x),
//							smoothstep(0.825, 1.0, heights.y),
//							smoothstep(0.102, 0.12, heights.z),
//							smoothstep(0.015, 0.024, heights.w)	);
}

#endif
