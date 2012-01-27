#extension GL_EXT_gpu_shader4 : enable

#define LAYER_HEIGHT		0.0
#define LAYER_JACOBIAN_XX 	5.0
#define LAYER_JACOBIAN_YY	6.0
#define LAYER_JACOBIAN_XY	7.0

uniform mat4 invProjection; // screen space to camera space
uniform mat4 invView; 		// camera space to world space
uniform mat4 view; 		// camera space to world space
uniform mat4 mvp; 			// world space to screen space
uniform vec3 camWorldPos; 	// camera position in world space

uniform mat4 prevMVP; 		// world space to screen space transformation @ t-1

uniform sampler2DArray fftWavesSampler;
uniform sampler2DArray gausszSampler;

uniform vec4 GRID_SIZES;
uniform vec2 gridSize;

uniform sampler2D prevFoamSampler;

//uniform vec4 wzmin;
//uniform vec4 wzmax;

uniform vec4 foamParam;
uniform vec4 foamParam2;

uniform vec4 choppy_factor;

//uniform float choppy;
uniform float jacobian_scale;
uniform float delta;		// delta ticks, in seconds

varying vec2 u; 			// horizontal coordinates in world space used to compute P(u)
//varying vec2 uview; 		// horizontal coordinates in view space
varying vec3 P;						// Current position


vec2 oceanPos(vec4 vertex) {
    vec3 cameraDir = normalize((invProjection * vertex).xyz);
    vec3 worldDir = (invView * vec4(cameraDir, 0.0)).xyz;
    float t = -camWorldPos.z / worldDir.z;
    return camWorldPos.xy + t * worldDir.xy;
}

#ifdef _VERTEX_

void main() {
    u = oceanPos(gl_Vertex);
//	gl_Position = mvp * vec4(u, 0.0, 1.0);

    vec2 ux = oceanPos(gl_Vertex + vec4(gridSize.x, 0.0, 0.0, 0.0));
    vec2 uy = oceanPos(gl_Vertex + vec4(0.0, gridSize.y, 0.0, 0.0));
    vec2 dux = ux - u;
    vec2 duy = uy - u;

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

	P = vec3(u + dP.xy, dP.z);
	gl_Position = mvp * vec4(P, 1.0);
//	gl_Position = mvp * vec4(u, 0.0, 1.0);
//	gl_Position = mvp * vec4(P.xy, 0.0, 1.0);
}

#endif

#ifdef _FRAGMENT_

float myerfc(float x) {
	return 2.0 * exp(-x * x) / (2.319 * x + sqrt(4.0 + 1.52 * x * x));
}

float myerf(float x) {
    float y = 1.0 - myerfc(abs(x));
    return x >= 0.0 ? y : -y;
}

// from http://en.wikipedia.org/wiki/Error_function
float myerf2(float x)
{
   float a = 0.140012;
   float pi = 3.14159265;
   float x2 = x*x;
   float ax2 = a*x2;

   return sign(x) * sqrt( 1.0 - exp(-x2*(4.0/pi + ax2)/(1.0 + ax2)) );
}


// given a value, provide mean and average
float breakingValue(float scale, float mu, float sigma2)
{
    float SIGMA = sqrt(sigma2);
    float SQRT_2 = sqrt(2.0);

    return 0.5*myerf2(0.5*SQRT_2*(scale - 1.0 - mu)/SIGMA) + 0.5;
}

// get the filtered lifetime of the whitecap
float lifeValue(float scale, float mu)
{
    return scale - 1.0 - mu;
}

// Compute foam on screen space
// Get Sampler in Screen Space, using gl_FragCoord
void main() {
	// extract variables from foamparam
	float jmin = foamParam.x;
	float jmax = foamParam.y;
	float zmin = foamParam.z;
	float zmax = foamParam.w;
	float lifetime = foamParam2.x;
	float gen = foamParam2.y;
	float amp = foamParam2.z;

    // get jacobian data
    vec2 jm1 = texture2DArray(gausszSampler, vec3(u / GRID_SIZES.x, 2.0)).rg;
    vec2 jm2 = texture2DArray(gausszSampler, vec3(u / GRID_SIZES.y, 2.0)).ba;
    vec2 jm3 = texture2DArray(gausszSampler, vec3(u / GRID_SIZES.z, 3.0)).rg;
    vec2 jm4 = texture2DArray(gausszSampler, vec3(u / GRID_SIZES.w, 3.0)).ba;
    vec2 jm = jm1+jm2+jm3+jm4;
    float jSigma2 = max(jm.y - (jm1.x*jm1.x + jm2.x*jm2.x + jm3.x*jm3.x + jm4.x*jm4.x), 0.0);

    float foam  = breakingValue(jacobian_scale, jm.x, jSigma2);
    float life  = max(lifeValue(jacobian_scale, jm.x)*lifetime, 0.001); // we need to filter this !

	// get whitecap history
//	vec4 prevClip 	= prevMVP * vec4(P,1.0);
	vec4 prevClip 	= prevMVP * vec4(u, 0.0,1.0);
	vec3 prevNDC 	= prevClip.xyz/prevClip.w;
	if(any(greaterThan(abs(prevNDC),vec3(1.0))))
	{
		// no whitecap history
		gl_FragData[0].r 	= foam;
		gl_FragData[0].g 	= life;
//		gl_FragData[0].b 	= foam;
		return;
	}
	vec4 prevData 	= texture2D(prevFoamSampler, prevNDC.xy * 0.5 + 0.5);
    float lastfoam 	= prevData.r;
    float lastlife  = max(prevData.g,0.0001);

    // update whitecap value
//	foam 	    = min(max(lastfoam - delta/lastlife,foam),1.0);
    life        = max(life, lastlife);

	gl_FragData[0].r 	= foam;
	gl_FragData[0].g 	= life;
//	gl_FragData[0].b 	= foam > (lastfoam - delta/lastlife) ? foam : prevData.g;
//	gl_FragData[0].a 	= foam > 0.0 ? 1.0 : max(prevData.b - delta/lastlife,0.0);

//    vec2 vel1 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.x, 5.0)).xy;
//    vec2 vel2 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.y, 5.0)).zw;
//    vec2 vel3 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.z, 6.0)).xy;
//    vec2 vel4 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.w, 6.0)).zw;
//
//	gl_FragData[1].rg 	= choppy_factor.x*vel1
//                        + choppy_factor.y*vel2
//                        + choppy_factor.z*vel3
//                        + choppy_factor.w*vel4;
//    gl_FragData[1].r = gl_FragData[0].b;

}

#endif
