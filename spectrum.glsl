uniform sampler2D spectrum_1_2_Sampler;
uniform sampler2D spectrum_3_4_Sampler;

uniform vec4 INVERSE_GRID_SIZES;

uniform float FFT_SIZE;

uniform float zoom;

uniform float linear;

varying vec2 uv;

#ifdef _VERTEX_

void main() {
    uv = gl_Vertex.zw;
    gl_Position = vec4(gl_Vertex.xy, 0.0, 1.0);
}

#else

void main() {
    vec2 st = uv - vec2(0.5);
    float r = length(st);
    float k = pow(10.0, -3.0 + 12.0 * r);

    vec2 kxy = st * pow(10.0, zoom * 3.0);

    float S = 0.0;
    if (abs(kxy.x) < INVERSE_GRID_SIZES.x && abs(kxy.y) < INVERSE_GRID_SIZES.x) {
        st = 0.5 * kxy / INVERSE_GRID_SIZES.x + 0.5 / FFT_SIZE;
        S += length(texture2D(spectrum_1_2_Sampler, st).xy) * FFT_SIZE / INVERSE_GRID_SIZES.x;
    }
    if (abs(kxy.x) < INVERSE_GRID_SIZES.y && abs(kxy.y) < INVERSE_GRID_SIZES.y) {
        st = 0.5 * kxy / INVERSE_GRID_SIZES.y + 0.5 / FFT_SIZE;
        S += length(texture2D(spectrum_1_2_Sampler, st).zw) * FFT_SIZE / INVERSE_GRID_SIZES.y;
    }
    if (abs(kxy.x) < INVERSE_GRID_SIZES.z && abs(kxy.y) < INVERSE_GRID_SIZES.z) {
        st = 0.5 * kxy / INVERSE_GRID_SIZES.z + 0.5 / FFT_SIZE;
        S += length(texture2D(spectrum_3_4_Sampler, st).xy) * FFT_SIZE / INVERSE_GRID_SIZES.z;
    }
    if (abs(kxy.x) < INVERSE_GRID_SIZES.w && abs(kxy.y) < INVERSE_GRID_SIZES.w) {
        st = 0.5 * kxy / INVERSE_GRID_SIZES.w + 0.5 / FFT_SIZE;
        S += length(texture2D(spectrum_3_4_Sampler, st).zw) * FFT_SIZE / INVERSE_GRID_SIZES.w;
    }
    S = S * S * 0.5;

    float s;
    if (linear > 0.0) {
        s = S * 100.0; // linear scale in intensity
    } else {
        s = (log(S) / log(10.0) + 15.0) / 18.0; // logarithmic scale in intensity
    }

    gl_FragColor = vec4(s);
}

#endif
