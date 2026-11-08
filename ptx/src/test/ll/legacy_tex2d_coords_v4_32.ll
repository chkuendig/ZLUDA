%struct.texture = type { %struct.textureReference }
%struct.textureReference = type { i32, i32, i32, [3 x i32], %struct.hipChannelFormatDesc, i32, i32, i32, float, float, float, ptr, i32, i32 }
%struct.hipChannelFormatDesc = type { i32, i32, i32, i32, i32 }

@legacy_texture = addrspace(1) externally_initialized global %struct.texture undef, align 8

; Function Attrs: denormal_fpenv(dynamic)
declare hidden <4 x i32> @__zluda_ptx_impl_texref_2d_v4_s32_f32(ptr addrspace(1), <2 x float>) #0

; Function Attrs: denormal_fpenv(preservesign)
define amdgpu_kernel void @legacy_tex2d_coords_v4(ptr addrspace(4) byref(i32) %"52", ptr addrspace(4) byref(i64) %"78") #1 {
  %"53" = alloca i32, align 4, addrspace(5)
  %"54" = alloca i32, align 4, addrspace(5)
  %"55" = alloca i32, align 4, addrspace(5)
  %"56" = alloca i32, align 4, addrspace(5)
  %"57" = alloca i32, align 4, addrspace(5)
  %"58" = alloca float, align 4, addrspace(5)
  %"59" = alloca float, align 4, addrspace(5)
  %"60" = alloca float, align 4, addrspace(5)
  %"61" = alloca float, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"51"

"51":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"78", align 8
  %3 = load i32, ptr addrspace(4) %"52", align 4
  store i32 %3, ptr addrspace(5) %"53", align 4
  store float 5.000000e-01, ptr addrspace(5) %"58", align 4
  store float 5.000000e-01, ptr addrspace(5) %"59", align 4
  store float 1.000000e+02, ptr addrspace(5) %"60", align 4
  store float -1.000000e+02, ptr addrspace(5) %"61", align 4
  %4 = load float, ptr addrspace(5) %"58", align 4
  %5 = load float, ptr addrspace(5) %"59", align 4
  %6 = insertelement <2 x float> undef, float %4, i8 0
  %"49" = insertelement <2 x float> %6, float %5, i8 1
  %"48" = call <4 x i32> @__zluda_ptx_impl_texref_2d_v4_s32_f32(ptr addrspace(1) @legacy_texture, <2 x float> %"49")
  %"69" = extractelement <4 x i32> %"48", i8 0
  %"70" = extractelement <4 x i32> %"48", i8 1
  %"71" = extractelement <4 x i32> %"48", i8 2
  %"72" = extractelement <4 x i32> %"48", i8 3
  store i32 %"69", ptr addrspace(5) %"54", align 4
  store i32 %"70", ptr addrspace(5) %"55", align 4
  store i32 %"71", ptr addrspace(5) %"56", align 4
  store i32 %"72", ptr addrspace(5) %"57", align 4
  %7 = load i32, ptr addrspace(5) %"54", align 4
  %8 = load i32, ptr addrspace(5) %"55", align 4
  %9 = load i32, ptr addrspace(5) %"56", align 4
  %10 = load i32, ptr addrspace(5) %"57", align 4
  %11 = insertelement <4 x i32> undef, i32 %7, i8 0
  %12 = insertelement <4 x i32> %11, i32 %8, i8 1
  %13 = insertelement <4 x i32> %12, i32 %9, i8 2
  %"50" = insertelement <4 x i32> %13, i32 %10, i8 3
  %14 = load i32, ptr addrspace(5) %"53", align 4
  %15 = zext i32 %14 to i64
  %16 = inttoptr i64 %2 to ptr addrspace(1)
  %"80" = getelementptr inbounds i8, ptr addrspace(1) %16, i64 %15
  store <4 x i32> %"50", ptr addrspace(1) %"80", align 16
  ret void
}

attributes #0 = { denormal_fpenv(dynamic) "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { denormal_fpenv(preservesign) "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
