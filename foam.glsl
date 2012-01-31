#extension GL_EXT_gpu_shader4 : enable

#define LAYER_HEIGHT		0.0
#define LAYER_JACOBIAN_XX 	5.0
#define LAYER_JACOBIAN_YY	6.0
#define LAYER_JACOBIAN_XY	7.0

uniform sampler2DArray fftWavesSampler;
uniform sampler2DArray gausszSampler;

uniform mat4 invProjection; // screen space to camera space
uniform mat4 invView; 		// camera space to world space
uniform mat4 mvp; 			// world space to screen space
uniform vec3 camWorldPos; 	// camera position in world space


uniform vec4 GRID_SIZES;
uniform vec2 gridSize;

uniform vec4 choppy_factor;
uniform float jacobian_scale;

varying vec2 u;
varying vec2 umin; 
varying vec2 umax; 

vec2 oceanPos(vec4 vertex) {
	vec3 cameraDir = normalize((invProjection * vertex).xyz);
	vec3 worldDir = (invView * vec4(cameraDir, 0.0)).xyz;
	float t = -camWorldPos.z / worldDir.z;
	return camWorldPos.xy + t * worldDir.xy;
}

#ifdef _VERTEX_

void main() {
	u = oceanPos(gl_Vertex);

	vec2 ux = oceanPos(gl_Vertex + vec4(gridSize.x, 0.0, 0.0, 0.0));
	vec2 uy = oceanPos(gl_Vertex + vec4(0.0, gridSize.y, 0.0, 0.0));
	vec2 dux = ux - u;
	vec2 duy = uy - u;

	umin = min(min(u, ux),uy);
	umax = max(max(u, ux),uy);

	// sum altitudes (use grad to get correct mipmap level)
	vec3 dP = vec3(0.0);
	dP.z += texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.x, LAYER_HEIGHT), dux / GRID_SIZES.x, duy / GRID_SIZES.x).x;
	dP.z += texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.y, LAYER_HEIGHT), dux / GRID_SIZES.y, duy / GRID_SIZES.y).y;
	dP.z += texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.z, LAYER_HEIGHT), dux / GRID_SIZES.z, duy / GRID_SIZES.z).z;
	dP.z += texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.w, LAYER_HEIGHT), dux / GRID_SIZES.w, duy / GRID_SIZES.w).w;

	dP.xy += vec2(choppy_factor.x)*texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.x, 3.0), dux / GRID_SIZES.x, duy / GRID_SIZES.x).xy;
	dP.xy += vec2(choppy_factor.y)*texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.y, 3.0), dux / GRID_SIZES.y, duy / GRID_SIZES.y).zw;
	dP.xy += vec2(choppy_factor.z)*texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.z, 4.0), dux / GRID_SIZES.z, duy / GRID_SIZES.z).xy;
	dP.xy += vec2(choppy_factor.w)*texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.w, 4.0), dux / GRID_SIZES.w, duy / GRID_SIZES.w).zw;

	gl_Position = mvp * vec4(u + dP.xy, dP.z, 1.0);
}

#endif

#ifdef _FRAGMENT_

// from http://en.wikipedia.org/wiki/Error_function
float error_function(float x) {
	float a  = 0.140012;
	float pi = 3.14159265;
	float x2 = x*x;
	float ax2 = a*x2;

	return sign(x) * sqrt( 1.0 - exp(-x2*(4.0/pi + ax2)/(1.0 + ax2)) );
}


// given a value, provide mean and average
float breakingValue(float scale, float mu, float sigma2) {
	return 0.5*error_function(0.5*sqrt(2.0)*(scale - 1.0 - mu)*inversesqrt(sigma2)) + 0.5;
}

float foam_primitive(float x) {
	return 0.5*(x*error_function(x) + exp(-x*x)*inversesqrt(3.14159265)+x);
}


// Compute foam on screen space
// Get Sampler in Screen Space, using gl_FragCoord
void main() {
	// get jacobian data
	vec2 jm1 = texture2DArray(gausszSampler, vec3(u / GRID_SIZES.x, 2.0)).rg;
	vec2 jm2 = texture2DArray(gausszSampler, vec3(u / GRID_SIZES.y, 2.0)).ba;
	vec2 jm3 = texture2DArray(gausszSampler, vec3(u / GRID_SIZES.z, 3.0)).rg;
	vec2 jm4 = texture2DArray(gausszSampler, vec3(u / GRID_SIZES.w, 3.0)).ba;
	vec2 jm  = jm1+jm2+jm3+jm4;
	float jSigma2 = max(jm.y - (jm1.x*jm1.x + jm2.x*jm2.x + jm3.x*jm3.x + jm4.x*jm4.x), 0.0);

	float foam  = breakingValue(jacobian_scale, jm.x, jSigma2);

	gl_FragData[0].r = foam;
#if 0
	// use erf as main function
	float jxx0 = texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.x, LAYER_JACOBIAN_XX),0.0).r*choppy_factor.x
	           + texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.y, LAYER_JACOBIAN_XX),0.0).g*choppy_factor.y
	           + texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.z, LAYER_JACOBIAN_XX),0.0).b*choppy_factor.z
	           + texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.w, LAYER_JACOBIAN_XX),0.0).a*choppy_factor.w
	           + 1.0;

	float jyy0 = texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.x, LAYER_JACOBIAN_YY),0.0).r*choppy_factor.x
	           + texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.y, LAYER_JACOBIAN_YY),0.0).g*choppy_factor.y
	           + texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.z, LAYER_JACOBIAN_YY),0.0).b*choppy_factor.z
	           + texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.w, LAYER_JACOBIAN_YY),0.0).a*choppy_factor.w
	           + 1.0;

	float jxy0 = texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.x, LAYER_JACOBIAN_XY),0.0).r*choppy_factor.x
	           + texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.y, LAYER_JACOBIAN_XY),0.0).g*choppy_factor.y
	           + texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.z, LAYER_JACOBIAN_XY),0.0).b*choppy_factor.z
	           + texture2DArrayLod(fftWavesSampler, vec3(umin/GRID_SIZES.w, LAYER_JACOBIAN_XY),0.0).a*choppy_factor.w;

	float jxx1 = texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.x, LAYER_JACOBIAN_XX),0.0).r*choppy_factor.x
	           + texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.y, LAYER_JACOBIAN_XX),0.0).g*choppy_factor.y
	           + texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.z, LAYER_JACOBIAN_XX),0.0).b*choppy_factor.z
	           + texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.w, LAYER_JACOBIAN_XX),0.0).a*choppy_factor.w
	           + 1.0;
	float jyy1 = texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.x, LAYER_JACOBIAN_YY),0.0).r*choppy_factor.x
	           + texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.y, LAYER_JACOBIAN_YY),0.0).g*choppy_factor.y
	           + texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.z, LAYER_JACOBIAN_YY),0.0).b*choppy_factor.z
	           + texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.w, LAYER_JACOBIAN_YY),0.0).a*choppy_factor.w
	           + 1.0;

	float jxy1 = texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.x, LAYER_JACOBIAN_XY),0.0).r*choppy_factor.x
	           + texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.y, LAYER_JACOBIAN_XY),0.0).g*choppy_factor.y
	           + texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.z, LAYER_JACOBIAN_XY),0.0).b*choppy_factor.z
	           + texture2DArrayLod(fftWavesSampler, vec3(umax/GRID_SIZES.w, LAYER_JACOBIAN_XY),0.0).a*choppy_factor.w;

	float detJ0 = jxx0*jyy0 - jxy0*jxy0;
	float detJ1 = jxx1*jyy1 - jxy1*jxy1;

	if(gl_FragCoord.x > 800.0)
//		gl_FragData[0].r = error_function(4.0*(jacobian_scale-jxx0*jyy0+jxy0*jxy0))
//		                 * 0.5 + 0.5;
		gl_FragData[0].r = foam_primitive(jacobian_scale-detJ1)
		                 - foam_primitive(jacobian_scale-detJ0);

	if(isnan(gl_FragData[0].r))
		gl_FragData[0].r = 1.0;
#endif
}

#endif
