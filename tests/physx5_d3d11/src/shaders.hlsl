// Minimal shaders for particle point rendering
// Used by d3d11_renderer.cpp when rendering is enabled (non-headless mode)

cbuffer Constants : register(b0) {
    float4x4 viewProj;
    float pointSize;
    float3 padding;
};

struct VSInput {
    float4 position : POSITION;  // xyz = particle pos, w = 1
};

struct VSOutput {
    float4 position : SV_POSITION;
    float4 color    : COLOR;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = mul(viewProj, float4(input.position.xyz, 1.0));
    // Color by height (y) for visual debugging
    float t = saturate(input.position.y * 0.5 + 0.5);
    output.color = float4(t, 0.3, 1.0 - t, 1.0);
    return output;
}

float4 PSMain(VSOutput input) : SV_TARGET {
    return input.color;
}
