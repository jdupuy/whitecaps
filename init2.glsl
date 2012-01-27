
uniform sampler2D spectrum_1_2_Sampler;
uniform sampler2D spectrum_3_4_Sampler;

uniform float FFT_SIZE;

uniform vec4 INVERSE_GRID_SIZES;

uniform float t;

varying vec2 uv;

#ifdef _VERTEX_

void main() {
    uv = gl_Vertex.zw;
    gl_Position = vec4(gl_Vertex.xy, 0.0, 1.0);
}

#else


vec2 i(vec2 z) {
    return vec2(-z.y, z.x); // returns i times z (complex number)
}


// dh(k,t)/dt, complex number
vec2 getSpectrumDt(float k, vec2 s0, vec2 s0c) {
    float w = sqrt(9.81 * k * (1.0 + (k * k) / (370.0 * 370.0)));
    float c = cos(w * t);
    float s = sin(w * t);

    s0 = i(s0);
    s0c = -i(s0c);

    return w * vec2((s0.x + s0c.x) * c - (s0.y + s0c.y) * s, (s0.x - s0c.x) * s + (s0.y - s0c.y) * c);
}


void main() {
    vec2 st = floor(uv * FFT_SIZE) / FFT_SIZE; // in [-N/2,N/2[
    float x = uv.x > 0.5 ? st.x - 1.0 : st.x;
    float y = uv.y > 0.5 ? st.y - 1.0 : st.y;

	// h0(k)
    vec4 s12 = texture2DLod(spectrum_1_2_Sampler, uv, 0.0);
    vec4 s34 = texture2DLod(spectrum_3_4_Sampler, uv, 0.0);
    // conjugate (h0(k))
    vec4 s12c = texture2DLod(spectrum_1_2_Sampler, vec2(1.0 + 0.5 / FFT_SIZE) - st, 0.0);
    vec4 s34c = texture2DLod(spectrum_3_4_Sampler, vec2(1.0 + 0.5 / FFT_SIZE) - st, 0.0);

	// k
    vec2 k1 = vec2(x, y) * INVERSE_GRID_SIZES.x;
    vec2 k2 = vec2(x, y) * INVERSE_GRID_SIZES.y;
    vec2 k3 = vec2(x, y) * INVERSE_GRID_SIZES.z;
    vec2 k4 = vec2(x, y) * INVERSE_GRID_SIZES.w;

	// k magnitude
    float K1 = length(k1);
    float K2 = length(k2);
    float K3 = length(k3);
    float K4 = length(k4);

	// 1/kmag
    float IK1 = K1 == 0.0 ? 0.0 : 1.0 / K1;
    float IK2 = K2 == 0.0 ? 0.0 : 1.0 / K2;
    float IK3 = K3 == 0.0 ? 0.0 : 1.0 / K3;
    float IK4 = K4 == 0.0 ? 0.0 : 1.0 / K4;

    // dh(k,t)/dt
    vec2 dh1dt = getSpectrumDt(K1, s12.xy, s12c.xy);
    vec2 dh2dt = getSpectrumDt(K2, s12.zw, s12c.zw);
    vec2 dh3dt = getSpectrumDt(K3, s34.xy, s34c.xy);
    vec2 dh4dt = getSpectrumDt(K4, s34.zw, s34c.zw);

    // get horizontal velocities
    gl_FragData[0] = vec4(k1 * dh1dt.x, k2 * dh2dt.x) * vec4(IK1, IK1, IK2, IK2);
    gl_FragData[1] = vec4(k3 * dh3dt.x, k4 * dh4dt.x) * vec4(IK3, IK3, IK4, IK4);
}

#endif
