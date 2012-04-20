#ifdef HARDWARE_ANISTROPIC_FILTERING
#extension GL_EXT_gpu_shader4 : enable
#endif

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

#define LAYER_HEIGHT		0.0

#define LAYER_JACOBIAN_XX 	5.0
#define LAYER_JACOBIAN_YY	6.0
#define LAYER_JACOBIAN_XY	7.0


uniform mat4 screenToCamera; // screen space to camera space
uniform mat4 cameraToWorld; // camera space to world space
uniform mat4 worldToScreen; // world space to screen space
uniform mat4 worldDirToScreen; // world space to screen space
uniform mat4 modelView;       // modelViewMatrix
uniform vec3 worldCamera; // camera position in world space
uniform vec3 worldSunDir; // sun direction in world space

uniform vec2 gridSize;
uniform float normals;
uniform float choppy;
uniform vec4 choppy_factor;
//uniform float displacer;

uniform sampler2DArray fftWavesSampler;	// ocean surface

uniform vec4 GRID_SIZES;

uniform sampler3D slopeVarianceSampler;

uniform sampler2D foamSampler;	// Whitecap coverage information
uniform sampler2D foamNormalSampler;	// bump map for foam
uniform sampler2D noiseSampler2;

uniform vec3 seaColor; // sea bottom color

varying vec2 u; 	// horizontal coordinates in world space used to compute P(u)
varying vec3 P; 	// wave point P(u) in world space

#ifdef _VERTEX_

vec2 oceanPos(vec4 vertex) {
    vec3 cameraDir = normalize((screenToCamera * vertex).xyz);
    vec3 worldDir = (cameraToWorld * vec4(cameraDir, 0.0)).xyz;
    float t = -worldCamera.z / worldDir.z;
    return worldCamera.xy + t * worldDir.xy;
}

void main() {
//    gl_Position = gl_Vertex;
//	foam = 0.0;
    u = oceanPos(gl_Vertex);
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

    // choppy
    if (choppy > 0.0) {

        dP.xy += vec2(choppy_factor.x)*texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.x, 3.0), dux / GRID_SIZES.x, duy / GRID_SIZES.x).xy;
        dP.xy += vec2(choppy_factor.y)*texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.y, 3.0), dux / GRID_SIZES.y, duy / GRID_SIZES.y).zw;
        dP.xy += vec2(choppy_factor.z)*texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.z, 4.0), dux / GRID_SIZES.z, duy / GRID_SIZES.z).xy;
        dP.xy += vec2(choppy_factor.w)*texture2DArrayGrad(fftWavesSampler, vec3(u / GRID_SIZES.w, 4.0), dux / GRID_SIZES.w, duy / GRID_SIZES.w).zw;
    }
    P = vec3(u + dP.xy, dP.z);
//    P = vec3(u, 0);

	// Final position
	gl_Position = worldToScreen * vec4(P, 1.0);
}

#endif

#ifdef _FRAGMENT_

// ---------------------------------------------------------------------------
// REFLECTED SUN RADIANCE
// ---------------------------------------------------------------------------

// assumes x>0
float erfc(float x) {
	return 2.0 * exp(-x * x) / (2.319 * x + sqrt(4.0 + 1.52 * x * x));
}

float Lambda(float cosTheta, float sigmaSq) {
	float v = cosTheta / sqrt((1.0 - cosTheta * cosTheta) * (2.0 * sigmaSq));
    return max(0.0, (exp(-v * v) - v * sqrt(M_PI) * erfc(v)) / (2.0 * v * sqrt(M_PI)));
	//return (exp(-v * v)) / (2.0 * v * sqrt(M_PI)); // approximate, faster formula
}

// L, V, N, Tx, Ty in world space
float reflectedSunRadiance(vec3 L, vec3 V, vec3 N, vec3 Tx, vec3 Ty, vec2 sigmaSq, out float p) {
    vec3 H = normalize(L + V);
    float zetax = dot(H, Tx) / dot(H, N);
    float zetay = dot(H, Ty) / dot(H, N);

    float zL = dot(L, N); // cos of source zenith angle
    float zV = dot(V, N); // cos of receiver zenith angle
    float zH = dot(H, N); // cos of facet normal zenith angle
    float zH2 = zH * zH;

    /*float*/ p = exp(-0.5 * (zetax * zetax / sigmaSq.x + zetay * zetay / sigmaSq.y))
                / (2.0 * M_PI * sqrt(sigmaSq.x * sigmaSq.y));

    float tanV = atan(dot(V, Ty), dot(V, Tx));
    float cosV2 = 1.0 / (1.0 + tanV * tanV);
    float sigmaV2 = sigmaSq.x * cosV2 + sigmaSq.y * (1.0 - cosV2);

    float tanL = atan(dot(L, Ty), dot(L, Tx));
    float cosL2 = 1.0 / (1.0 + tanL * tanL);
    float sigmaL2 = sigmaSq.x * cosL2 + sigmaSq.y * (1.0 - cosL2);

    float fresnel = 0.02 + 0.98 * pow(1.0 - dot(V, H), 5.0);

    zL = max(zL, 0.01);
    zV = max(zV, 0.01);

    return fresnel * p / ((1.0 + Lambda(zL, sigmaL2) + Lambda(zV, sigmaV2)) * zV * zH2 * zH2 * 4.0);
}

// ---------------------------------------------------------------------------
// REFLECTED SKY RADIANCE
// ---------------------------------------------------------------------------

// manual anisotropic filter
vec4 myTexture2DGrad(sampler2D tex, vec2 u, vec2 s, vec2 t)
{
    const float TEX_SIZE = 512.0; // 'tex' size in pixels
    const int N = 1; // use (2*N+1)^2 samples
    vec4 r = vec4(0.0);
    float l = max(0.0, log2(max(length(s), length(t)) * TEX_SIZE) - 0.0);
    for (int i = -N; i <= N; ++i) {
        for (int j = -N; j <= N; ++j) {
            r += texture2DLod(tex, u + (s * float(i) + t * float(j)) / float(N), l);
        }
    }
    return r / pow(2.0 * float(N) + 1.0, 2.0);
}

// V, N, Tx, Ty in world space
vec2 U(vec2 zeta, vec3 V, vec3 N, vec3 Tx, vec3 Ty) {
    vec3 f = normalize(vec3(-zeta, 1.0)); // tangent space
    vec3 F = f.x * Tx + f.y * Ty + f.z * N; // world space
    vec3 R = 2.0 * dot(F, V) * F - V;
    return R.xy / (1.0 + R.z);
}

float meanFresnel(float cosThetaV, float sigmaV) {
	return pow(1.0 - cosThetaV, 5.0 * exp(-2.69 * sigmaV)) / (1.0 + 22.7 * pow(sigmaV, 1.5));
}

// V, N in world space
float meanFresnel(vec3 V, vec3 N, vec2 sigmaSq) {
    vec2 v = V.xy; // view direction in wind space
    vec2 t = v * v / (1.0 - V.z * V.z); // cos^2 and sin^2 of view direction
    float sigmaV2 = dot(t, sigmaSq); // slope variance in view direction
    return meanFresnel(dot(V, N), sqrt(sigmaV2));
}

// V, N, Tx, Ty in world space;
vec3 meanSkyRadiance(vec3 V, vec3 N, vec3 Tx, vec3 Ty, vec2 sigmaSq) {
    vec4 result = vec4(0.0);

    const float eps = 0.001;
    vec2 u0 = U(vec2(0.0), V, N, Tx, Ty);
    vec2 dux = 2.0 * (U(vec2(eps, 0.0), V, N, Tx, Ty) - u0) / eps * sqrt(sigmaSq.x);
    vec2 duy = 2.0 * (U(vec2(0.0, eps), V, N, Tx, Ty) - u0) / eps * sqrt(sigmaSq.y);

#ifdef HARDWARE_ANISTROPIC_FILTERING
    result = texture2DGrad(skySampler, u0 * (0.5 / 1.1) + 0.5, dux * (0.5 / 1.1), duy * (0.5 / 1.1));
#else
    result = myTexture2DGrad(skySampler, u0 * (0.5 / 1.1) + 0.5, dux * (0.5 / 1.1), duy * (0.5 / 1.1));
#endif
    //if texture2DLod and texture2DGrad are not defined, you can use this (no filtering):
	//result = texture2D(skySampler, u0 * (0.5 / 1.1) + 0.5);

    return result.rgb;
}


/*
	Whitecap reflectance
	R = R_0*exp{-sqrt(a*b)}
	R_0~1.0
	a : absorption coefficient
		a = C_p * a_p + C_y * a_y + a_w
		*_p : pigment concentration params
		*_y : yellow substance params
	b = s*Q\B2
		s = B\B2*d*sqrt(c)
			d : bubble diameter size
			c : liquid fraction
			B = 4/sqrt(3D) : D ~ 1
		Q = q(o)*q(o_0) / R_0
		q(o) = 3(1+2cos(o))/7

	// alpha : absorption (uniform variable)
	vec3 foamReflectance(in vec3 alpha)
	{
		// wavelength(lambda) of each color (red, green, blue) = (650, 510, 475) nm
		vec3 lambda = vec3(650, 510, 475);

		lambda_0 = 440;	// nm

		// Compute s
		s = 2.3 * 2.3 * d * sqrt(c);
		// Compute Q
		Q = q(a)*q(b)/R0;

		vec3 b = s * Q * Q;
		// b = 2, 17, or 85 in Kokhanovsky2004 use fix for now
		// HAVE TO COMPUTE DEPENDING ON WHITECAP AGE
		return R_0 * exp( -sqrt(alpha * b) );
	}


 */

// ----------------------------------------------------------------------------

void main() {

	vec3 V = normalize(worldCamera - P);

	vec2 slopes = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.x, 1.0)).xy;
	slopes += texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.y, 1.0)).zw;
	slopes += texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.z, 2.0)).xy;
	slopes += texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.w, 2.0)).zw;

	if(choppy > 0.0)
	{
		float Jxx, Jxy, Jyy;
		vec4 lambda = choppy_factor;
		// Jxx1..4 : partial Jxx
		float Jxx1 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.x, LAYER_JACOBIAN_XX)).r;
		float Jxx2 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.y, LAYER_JACOBIAN_XX)).g;
		float Jxx3 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.z, LAYER_JACOBIAN_XX)).b;
		float Jxx4 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.w, LAYER_JACOBIAN_XX)).a;
		Jxx = dot((lambda), vec4(Jxx1,Jxx2,Jxx3,Jxx4));

		// Jyy1..4 : partial Jyy
		float Jyy1 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.x, LAYER_JACOBIAN_YY)).r;
		float Jyy2 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.y, LAYER_JACOBIAN_YY)).g;
		float Jyy3 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.z, LAYER_JACOBIAN_YY)).b;
		float Jyy4 = texture2DArray(fftWavesSampler, vec3(u / GRID_SIZES.w, LAYER_JACOBIAN_YY)).a;
		Jyy = dot((lambda), vec4(Jyy1,Jyy2,Jyy3,Jyy4));

		slopes /= (1.0 + vec2(Jxx, Jyy));
	}

	vec3 N = normalize(vec3(-slopes.x, -slopes.y, 1.0));
	if (dot(V, N) < 0.0) {
		N = reflect(N, V); // reflects backfacing normals
	}

	float Jxx = dFdx(u.x);
	float Jxy = dFdy(u.x);
	float Jyx = dFdx(u.y);
	float Jyy = dFdy(u.y);
	float A = Jxx * Jxx + Jyx * Jyx;
	float B = Jxx * Jxy + Jyx * Jyy;
	float C = Jxy * Jxy + Jyy * Jyy;
	const float SCALE = 10.0;
	float ua = pow(A / SCALE, 0.25);
	float ub = 0.5 + 0.5 * B / sqrt(A * C);
	float uc = pow(C / SCALE, 0.25);
	vec2 sigmaSq = texture3D(slopeVarianceSampler, vec3(ua, ub, uc)).xw;

	sigmaSq = max(sigmaSq, 2e-5);

	vec3 Ty = normalize(vec3(0.0, N.z, -N.y));
	vec3 Tx = cross(Ty, N);

	vec3 Rf = vec3(0.0);
	vec3 Rs = vec3(0.0);
	vec3 Ru = vec3(0.0);
	float p = 1.0;

#if defined(SEA_CONTRIB) || defined(SKY_CONTRIB)
	float fresnel = 0.02 + 0.98 * meanFresnel(V, N, sigmaSq);
#endif

	vec3 Lsun;
	vec3 Esky;
	vec3 extinction;
	sunRadianceAndSkyIrradiance(worldCamera + earthPos, worldSunDir, Lsun, Esky);

	gl_FragColor = vec4(0.0);

#ifdef SUN_CONTRIB
	reflectedSunRadiance(worldSunDir, V, N, Tx, Ty, sigmaSq, p);
	Rs += reflectedSunRadiance(worldSunDir, V, N, Tx, Ty, sigmaSq, p) * Lsun;
	gl_FragColor.rgb = Rs;
#endif

#ifdef SKY_CONTRIB
	Rs += fresnel * meanSkyRadiance(V, N, Tx, Ty, sigmaSq);
	gl_FragColor.rgb = Rs;
#endif

#ifdef SEA_CONTRIB
	vec3 Lsea = seaColor * Esky / M_PI;
	Ru += (1.0 - fresnel) * Lsea;
	gl_FragColor.rgb += Ru;
#endif

#if !defined(SEA_CONTRIB) && !defined(SKY_CONTRIB) && !defined(SUN_CONTRIB)
	Rs = 0.0001 * seaColor * (Lsun * max(dot(N, worldSunDir), 0.0) + Esky) / M_PI;
	gl_FragColor.rgb = Rs;
#endif

#ifdef FOAM_CONTRIB

	// compute projected grid vertex
//	vec4 uclip 	= worldToScreen2 * vec4(u, 0.0, 1.0);
	vec4 uclip 	= worldToScreen * vec4(P, 1.0);
	vec3 undc 	= uclip.xyz/uclip.w;
	vec2 ufg 	= undc.xy * 0.5 + 0.5;

	// get foam data
	vec4 wData = texture2DLod(foamSampler, ufg, 0.0);
//	float W_a = wData.y;
	float W_b = wData.r;//>0.0?wData.r:wData.b;
//	float Rt_a = 2.0 / (18.0*(1.0-W_a) + 2.0); // f(t) in [0.55,0.05]

	// base foam texture
//	float x0 = texture2DLod(noiseSampler2, u * 0.25, 8.0).r;
//	float x0 = texture2D(noiseSampler2, u.xy * 0.25).r;//*0.5;
//	x0 += texture2D(noiseSampler2, u.xy * 0.25).r*0.5;

	// stage b : upper

//	float t_b = max(wData.b - 0.15,0.0);
//	float x_b = t_b < 0.015 ? 0.95 : step(pow(t_b-0.015,0.125), x0);
//	float x_b = t_b < 0.015 ? x0 : x0*step(pow(t_b-0.015,0.0125), x0);
	float x_b = 1.0;
//    if(wData.r > 0.0)
//	{
    float x0 = texture2D(noiseSampler2, u.xy * 0.25).r;
//	    float t_b = 1.0 - min(wData.r + 1.15,1.0);
    float t_b = 1.0 - min(wData.r*2.0,1.0);
    x_b = x0*step(pow(t_b,0.125), x0);
    x_b = 1.0;
//	}
//	float t_b = (Rt_b - 0.05)*2.0;
//	float x_b = step(pow(t_b,0.2), x0);

//	// foam surface aspect
//	// play with thresholds to get different aspects
//	//           { 0 if x < s0(t)
//	// f(x,t) := { 1 if x > s1(t)	s0(t) <= s1(t), t in [0,1] is the lifetime of the whitecap
//	//           { x otherwise
//	float t = 1.0-W;
//	float s0 = t > 0.5 ? t*2.0 : 0.0;
//	float s1 = t <= 0.5 ? t*2.0 : 1.0;
//	float x = x0*step(s0,x0);
//	x = max(x, step(s1,x));
//	float tmp = max(dot(N, vec3(0.0,0.0,1.0)),0.0);
//	vec3 N_f = (vec4(normalize(texture2D(foamNormalSampler, u * 0.25).rgb*2.0-1.0),0.0)).xyz; // fetch surface normal

	float l = length(Lsun * max(dot(N, worldSunDir), 0.0) + Esky) / M_PI;
	float l2 = length(Lsun + Esky) / M_PI;

	vec3 R_ftot = vec3(x_b * wData.r * l * 0.8);//* exp(-(1.0-x_b * wData.r)*2.0*vec3(0.34, 0.032, 0.011));
	gl_FragColor.rgb += R_ftot ;// + (1.0-wData.r)*Rs + (1.0-R_ftot)*Ru;//* x0;// * x_b;


/// Raytrace the surface and add foam (if any)
	// http://www.thepolygoners.com/tutorials/lineplane/lineplane.html
	// todo : refract the angle (which differs ith wavelength ..)
//	float n1 = 1.00;	// refraction index of air
//	float n2 = (W_b + (1.0-W_b)*1.33);	// refraction index of water
//	vec3 rayOrigin 	= P;
//	vec3 rayDir 	= normalize(P - worldCamera);
//
//	rayDir = refract(rayDir, N, n2);
//
//	float r_length = 15.0;
//
//	vec3 u2;
//	vec2 dsp = u - P.xy;	// displacement (hoping it's roughly uniform which it AIN'T)
//	float max_iter = 17.0;//*max(100.0/dot(worldCamera-P,worldCamera-P), 0.0);
//	for(float j = 1.0; j < max_iter; j += 1.0)
//	{
//		// get new World position and project
//		u2 = P + (rayDir*r_length) * (j/max_iter);
//
//		vec4 u2clip = worldToScreen2 * vec4(u2.xy+dsp, 0.0, 1.0);
////		vec4 u2clip = worldToScreen2 * vec4(u2, 1.0);
//		vec2 u2ndc 	= u2clip.xy/u2clip.w;
//		// get W coverage
//		vec4 w2_data 	= texture2DLod(foamSampler, u2ndc * vec2(0.5) + vec2(0.5),1.0);
//		float x02 		= 1.0;//texture2DLod(noiseSampler2, (u2.xy+dsp) * 0.25, j).r;
////		x02 = x02*step(pow(min(w2_data.a+0.1,1.0),0.25), x02);
//		float W_b2 = w2_data.r;
//		float z2 = w2_data.b;	// height of wave
////		float zw = w2_data.b;// depth of whitecap
//		float k = distance(P,u2);
//		gl_FragColor.rgb += (1.0-wData.r)*W_b2/(4.0*3.14)*r_length/max_iter*(1.0 - fresnel)*l2*exp(-k*vec3(0.34, 0.032, 0.011));
//	}

#endif

	gl_FragColor.rgb = hdr(gl_FragColor.rgb);

	// render normals
	if (normals > 0.0) {
		gl_FragColor.rgb = abs(N);
	}
}

#endif
