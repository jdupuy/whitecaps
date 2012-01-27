#version 330 compatibility

#ifdef _VERTEX_

layout(location=0)
	in vec4 vsin_Pos;

out vec2 u;

// Compute ocean pos
// if a particle gets culled, reset to ndc.
// what about sky ?
// => always present, but no displacement
// how to detect this ?
//
vec2 oceanPos(vec4 vertex) {
    vec3 cameraDir = normalize((invProjection * vertex).xyz);
    vec3 worldDir = (invView * vec4(cameraDir, 0.0)).xyz;
    float t = -camWorldPos.z / worldDir.z;
    return camWorldPos.xy + t * worldDir.xy;
}

void main() {
    u = oceanPos(vsin_Pos);
	// check if backfiring, if yes, terminate

	// get curent and previous horizontal displacement

	// get displacement and affect to vertex

	// if the vertex is culled,

	// also move particles with camera motion. But what if the camera zooms out ? FUCK !
	// idea is to have a regular sampling
	//

}

#endif // _VERTEX_

#ifdef _GEOMETRY_

void main()
{
	// Get
}

#endif // _GEOMETRY_

#ifdef _FRAGMENT_

void main()
{

}

#endif // _FRAGMENT_

