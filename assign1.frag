// FILE: assign1.frag

//============================================================================
// STUDENT NAME: Kuang Hao
// COMMENTS TO GRADER: For task 2, I scale the height of bump, computed perturbed 
// normal vector in eye space, and output color with ambient & diffuse factors.
// For task 3, I followed the lecture note. For bump area, I construct 
// eye-space vectors, then tangent-space perturbation vector for a hemispherical
// bump, and compute the reflect color. For other area, I simply compute the color
// using wooden texture.
//============================================================================

// FRAGMENT SHADER

#version 430 core

//============================================================================
// Received from rasterizer.
//============================================================================
in vec3 ecPosition;   // Fragment's 3D position in eye space.
in vec3 ecNormal;     // Fragment's normal vector in eye space.
in vec3 v2fTexCoord;  // Fragment's texture coordinates. It is 3D when it is 
                      //   used as texture coordinates to a cubemap.


//============================================================================
// Indicates which object is being rendered.
// 0 -- draw skybox, 1 -- draw brick cube, 2 -- draw wooden cube.
//============================================================================
uniform int WhichObj;  


//============================================================================
// View and projection matrices, etc.
//============================================================================
uniform mat4 ViewMatrix;          // View transformation matrix.
uniform mat4 ModelViewMatrix;     // ModelView matrix.
uniform mat4 ModelViewProjMatrix; // ModelView matrix * Projection matrix.
uniform mat3 NormalMatrix;        // For transforming object-space direction 
                                  //   vector to eye space.

//============================================================================
// Light info.
//============================================================================
uniform vec4 LightPosition; // Given in eye space. Can be directional.
uniform vec4 LightAmbient; 
uniform vec4 LightDiffuse;
uniform vec4 LightSpecular;

// Material shininess for specular reflection.
const float MatlShininess = 128.0;


//============================================================================
// Environment cubemap used for skybox and reflection mapping.
//============================================================================
layout (binding = 0) uniform samplerCube EnvMap;

//============================================================================
// The brick texture map whose color is used as the ambient and diffuse 
// material in the lighting computation.
//============================================================================
layout (binding = 1) uniform sampler2D BrickDiffuseMap;

//============================================================================
// The brick normal map whose color is used as perturbed normal vector
// in the tangent space.
//============================================================================
layout (binding = 2) uniform sampler2D BrickNormalMap;

//============================================================================
// The wood texture map whose color is used as the ambient and diffuse 
// material in the lighting computation.
//============================================================================
layout (binding = 3) uniform sampler2D WoodDiffuseMap;


//============================================================================
// MirrorTileDensity defines the number of hemispherical mirrors across each 
// dimension when the corresponding texture coordinate ranges from 0.0 to 1.0.
//============================================================================
const float MirrorTileDensity = 2.0;  // (0.0, inf)


//============================================================================
// MirrorRadius is the radius of the hemispherical mirror in each tile. 
// The radius is relative to the tile size, which is considered to be 1.0 x 1.0.
//============================================================================
const float MirrorRadius = 0.4;  // (0.0, 0.5]


//============================================================================
// DeltaNormal_Z_Scale is used to exaggerate the height of bump when doing
// normal mapping. The z component of the decoded perturbed normal vector 
// read from the normal map is multiplied by DeltaNormal_Z_Adj.
//============================================================================
const float DeltaNormal_Z_Scale = 1.0 / 5.0;


//============================================================================
// Output to color buffer.
//============================================================================
layout (location = 0) out vec4 FragColor;



/////////////////////////////////////////////////////////////////////////////
// Compute fragment color on skybox.
/////////////////////////////////////////////////////////////////////////////
void drawSkybox() 
{
    FragColor = texture(EnvMap, v2fTexCoord);
}



/////////////////////////////////////////////////////////////////////////////
// Compute the Tangent vector T and the Binormal vector B, given the
// Normal vector N, a 3D position p, and 2D texture coordinates uv.
// Note that T, B, N and p are all in the same coordinate space.
/////////////////////////////////////////////////////////////////////////////
void compute_tangent_vectors( vec3 N, vec3 p, vec2 uv, out vec3 T, out vec3 B )
{
    // Please refer to "Followup: Normal Mapping Without Precomputed Tangents" at
    // http://www.thetenthplanet.de/archives/1180

    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );
 
    // solve the linear system
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    T = normalize( dp2perp * duv1.x + dp1perp * duv2.x );  // Tangent vector
    B = normalize( dp2perp * duv1.y + dp1perp * duv2.y );  // Binormal vector
}



/////////////////////////////////////////////////////////////////////////////
// Compute fragment color on brick cube.
/////////////////////////////////////////////////////////////////////////////
void drawBrickCube()
{
    if (gl_FrontFacing) {
        vec3 viewVec = -normalize(ecPosition);
        vec3 necNormal = normalize(ecNormal);

        vec3 lightVec;
        if (LightPosition.w == 0.0 )
            lightVec = normalize(LightPosition.xyz);
        else
            lightVec = normalize(LightPosition.xyz - ecPosition);

        /////////////////////////////////////////////////////////////////////////////
        // TASK 2:
        // * Construct eye-space Tangent and Binormal vectors.
        // * Read and decode tangent-space perturbation vector from normal map.
        // * Transform perturbation vector to eye space.
        // * Use eye-space perturbation vector as normal vector in lighting
        //   computation using Phong Reflection Model.
        // * Write computed fragment color to FragColor.
        /////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////
        // TASK 2: WRITE YOUR CODE HERE. //
        ///////////////////////////////////

        // Set up variables for building tangent frame.
        vec2 uv = v2fTexCoord.st;
        vec3 B,T;

        // Use the above given function to build tangent frame.
        compute_tangent_vectors(necNormal, ecPosition, uv, T, B);

        // Get normal vector in tangent space from normal map range [0,1]
        vec3 tanNormal = texture(BrickNormalMap, uv).rgb;

        // Get diffuse color.
        vec3 diffse = texture(BrickDiffuseMap, uv).rgb;

        // Compute ambient color from experience.
        vec3 ambientCol = 0.1 * diffse * LightAmbient.rgb;

        // Convert normal vector from [0,1] to range [-1,1].
        tanNormal = tanNormal * 2.0 - 1.0;

        // Exaggerate the height of bump using z_scale
        tanNormal.b *= DeltaNormal_Z_Scale;

        // Normalize the tangent normal vetcor AFTER scaling the bump.
        tanNormal = normalize(tanNormal);

        // Compute the perturbed normal vector in eye space
        vec3 ecPerturbedNormal = tanNormal.x * T + tanNormal.y * B + tanNormal.z * necNormal;

        // Compute diffuse color with light influence.
        float diff = max(dot(lightVec, ecPerturbedNormal), 0.0);
        vec3 diffuseCol = diffse * LightDiffuse.rgb * diff;

        // Brick don`t have specular color, so LightSpecular isn`t applied here.
        FragColor = vec4(diffuseCol + ambientCol, 1.0); 
    }
    else discard;
}



/////////////////////////////////////////////////////////////////////////////
// Compute fragment color on wooden cube.
/////////////////////////////////////////////////////////////////////////////
void drawWoodenCube()
{
    if (gl_FrontFacing) {
        vec3 viewVec = -normalize(ecPosition);
        vec3 necNormal = normalize(ecNormal);

        vec3 lightVec;
        if (LightPosition.w == 0.0 )
            lightVec = normalize(LightPosition.xyz);
        else
            lightVec = normalize(LightPosition.xyz - ecPosition);

        /////////////////////////////////////////////////////////////////////////////
        // TASK 3:
        // * Determine whether fragment is in wood region or mirror region.
        // * If fragment is in wood region,
        //    -- Read from wood texture map. 
        //    -- Perform Phong lighting computation using the wood texture 
        //       color as the ambient and diffuse material.
        //    -- Write computed fragment color to FragColor.
        // * If fragment is in mirror region,
        //    -- Construct eye-space Tangent and Binormal vectors.
        //    -- Construct tangent-space perturbation vector for a
        //       hemispherical bump.
        //    -- Transform perturbation vector to eye space.
        //    -- Reflect the view vector about the eye-space perturbation vector.
        //    -- Transform reflection vector to World Space.
        //    -- Use world-space reflection vector to access environment cubemap.
        //    -- Write computed fragment color to FragColor.
        /////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////
        // TASK 3: WRITE YOUR CODE HERE. //
        ///////////////////////////////////
        // Set up variables for building tangent frame.
        vec2 uv = v2fTexCoord.st;
        vec3 N = normalize(ecNormal);
        vec3 B;
        vec3 T;
        float PI = 3.1415926535;

        // Compute perturbed normal vector in tangent space of fragment.
        vec2 c = MirrorTileDensity * uv;
        vec2 p = fract(c) - vec2(0.5);
        float sqrDist = p.x * p.x + p.y * p.y;
       
        // If it is within bump area
        if (sqrDist <= MirrorRadius * MirrorRadius) {

            // Use the above given function to build tangent frame.
            compute_tangent_vectors(N, ecPosition, p, T, B);

            // Consider that the bump is a hemisphere, p.z isn`t always 1.
            float d = sqrt(sqrDist) / MirrorRadius;
            float bumpRatio = cos(d * (PI / 2));

            // Compute the perturbed normal vector in tangent space.
            vec3 tanPertubedNormal = normalize(vec3(p.x, p.y, bumpRatio));

            // Compute the perturbed normal vector in eye space.
            vec3 ecPerturbedNormal = tanPertubedNormal.x * T + tanPertubedNormal.y * B + tanPertubedNormal.z * N;

            // Compute reflective vector R.
            vec3 R = reflect(normalize(ecPosition), ecPerturbedNormal);

            // Use R to compute reflective color.
            vec3 color = texture(EnvMap, R).rgb;
            FragColor = vec4(color, 1);
            return;
        }else {

            // If outside bump area, just compute the liaghting of a wood cube.
            // Diffuse
            float diff = max(dot(necNormal, lightVec), 0.0);
            vec3 diffuse = texture(WoodDiffuseMap, uv).rgb;
            vec3 diffuseCol = diffuse * LightDiffuse.rgb * diff;

            // I googled and find that ambient factor of wooden texture can be 0. 
            //vec3 ambientCol = LightAmbient.rgb * 1 * diffuseCol;

            // Specular
            float spec = max(dot(reflect(-lightVec, ecNormal), viewVec), 0.0);
            vec3 specularCol = LightSpecular.rgb * pow(spec, MatlShininess);

            FragColor = vec4(diffuseCol + specularCol, 1); // output col;
            }        
    }
    else discard;
}



void main()
{
    switch(WhichObj) {
        case 0: drawSkybox(); break;
        case 1: drawBrickCube(); break;
        case 2: drawWoodenCube(); break;
    }
}
