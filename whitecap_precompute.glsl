#extension GL_EXT_gpu_shader4 : enable


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
	vec4 Jxy = choppy*texture2DArray(fftWavesSampler, vec3(uv, LAYER_JACOBIAN_XY));

	// Store partial jacobians
	vec4 res = 0.25 + Jxx + Jyy + Jxx*Jyy - Jxy*Jxy;
	vec4 res2 = res*res;
	gl_FragData[2] = vec4(res.x, res2.x, res.y, res2.y);
	gl_FragData[3] = vec4(res.z, res2.z, res.w, res2.w);

}

#endif
