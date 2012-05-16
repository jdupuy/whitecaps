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
	vec4 Jxy = choppy*choppy*texture2DArray(fftWavesSampler, vec3(uv, LAYER_JACOBIAN_XY));

	// Store partial jacobians
	vec4 res = Jxx + Jyy + choppy*Jxx*Jyy - Jxy*Jxy;
	vec4 res2 = res*res;
	gl_FragData[2] = vec4(res.x, res2.x, res.y, res2.y);
	gl_FragData[3] = vec4(res.z, res2.z, res.w, res2.w);

}

#endif
