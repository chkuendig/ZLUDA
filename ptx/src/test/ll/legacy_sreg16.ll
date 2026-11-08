; Function Attrs: denormal_fpenv(dynamic)
declare hidden i32 @__zluda_ptx_impl_sreg_tid(i8) #0

; Function Attrs: denormal_fpenv(dynamic)
declare hidden i32 @__zluda_ptx_impl_sreg_ntid(i8) #0

; Function Attrs: denormal_fpenv(dynamic)
declare hidden i32 @__zluda_ptx_impl_sreg_ctaid(i8) #0

; Function Attrs: denormal_fpenv(dynamic)
declare hidden i32 @__zluda_ptx_impl_sreg_nctaid(i8) #0

; Function Attrs: denormal_fpenv(preservesign)
define amdgpu_kernel void @legacy_sreg16(ptr addrspace(4) byref(i64) %"103", ptr addrspace(4) byref(i64) %"104") #1 {
  %"105" = alloca i64, align 8, addrspace(5)
  %"106" = alloca i32, align 4, addrspace(5)
  %"107" = alloca i32, align 4, addrspace(5)
  %"108" = alloca i32, align 4, addrspace(5)
  %"109" = alloca i32, align 4, addrspace(5)
  %"110" = alloca i32, align 4, addrspace(5)
  %"111" = alloca i32, align 4, addrspace(5)
  %"112" = alloca i32, align 4, addrspace(5)
  %"113" = alloca i32, align 4, addrspace(5)
  %"114" = alloca i32, align 4, addrspace(5)
  %"115" = alloca i32, align 4, addrspace(5)
  %"116" = alloca i32, align 4, addrspace(5)
  %"117" = alloca i16, align 2, addrspace(5)
  %"118" = alloca i16, align 2, addrspace(5)
  %"119" = alloca i16, align 2, addrspace(5)
  %"120" = alloca i16, align 2, addrspace(5)
  %"121" = alloca i1, align 1, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"98"

"98":                                             ; preds = %1
  %"56" = call i32 @__zluda_ptx_impl_sreg_ctaid(i8 0)
  store i32 %"56", ptr addrspace(5) %"106", align 4
  %"58" = call i32 @__zluda_ptx_impl_sreg_nctaid(i8 0)
  store i32 %"58", ptr addrspace(5) %"107", align 4
  %2 = load i32, ptr addrspace(5) %"107", align 4
  %"124" = sub i32 %2, 1
  store i32 %"124", ptr addrspace(5) %"108", align 4
  %3 = load i32, ptr addrspace(5) %"106", align 4
  %4 = load i32, ptr addrspace(5) %"108", align 4
  %5 = icmp ne i32 %3, %4
  store i1 %5, ptr addrspace(5) %"121", align 1
  %6 = load i1, ptr addrspace(5) %"121", align 1
  br i1 %6, label %"13", label %"32"

"32":                                             ; preds = %"98"
  %"60" = call i32 @__zluda_ptx_impl_sreg_tid(i8 0)
  %"61" = trunc i32 %"60" to i16
  store i16 %"61", ptr addrspace(5) %"117", align 2
  %"63" = call i32 @__zluda_ptx_impl_sreg_ntid(i8 0)
  %"64" = trunc i32 %"63" to i16
  store i16 %"64", ptr addrspace(5) %"118", align 2
  %"66" = call i32 @__zluda_ptx_impl_sreg_ctaid(i8 0)
  %"67" = trunc i32 %"66" to i16
  store i16 %"67", ptr addrspace(5) %"119", align 2
  %"69" = call i32 @__zluda_ptx_impl_sreg_nctaid(i8 0)
  %"70" = trunc i32 %"69" to i16
  store i16 %"70", ptr addrspace(5) %"120", align 2
  %7 = load i16, ptr addrspace(5) %"117", align 2
  %"134" = zext i16 %7 to i32
  store i32 %"134", ptr addrspace(5) %"109", align 4
  %8 = load i16, ptr addrspace(5) %"118", align 2
  %"136" = zext i16 %8 to i32
  store i32 %"136", ptr addrspace(5) %"110", align 4
  %9 = load i16, ptr addrspace(5) %"119", align 2
  %"138" = zext i16 %9 to i32
  store i32 %"138", ptr addrspace(5) %"111", align 4
  %10 = load i16, ptr addrspace(5) %"120", align 2
  %"140" = zext i16 %10 to i32
  store i32 %"140", ptr addrspace(5) %"112", align 4
  %"72" = call i32 @__zluda_ptx_impl_sreg_tid(i8 0)
  %"73" = trunc i32 %"72" to i16
  %"142" = zext i16 %"73" to i32
  store i32 %"142", ptr addrspace(5) %"113", align 4
  %"75" = call i32 @__zluda_ptx_impl_sreg_ntid(i8 0)
  %"76" = trunc i32 %"75" to i16
  %"143" = zext i16 %"76" to i32
  store i32 %"143", ptr addrspace(5) %"114", align 4
  %"78" = call i32 @__zluda_ptx_impl_sreg_ctaid(i8 0)
  %"79" = trunc i32 %"78" to i16
  %"144" = zext i16 %"79" to i32
  store i32 %"144", ptr addrspace(5) %"115", align 4
  %"81" = call i32 @__zluda_ptx_impl_sreg_nctaid(i8 0)
  %"82" = trunc i32 %"81" to i16
  %"145" = zext i16 %"82" to i32
  store i32 %"145", ptr addrspace(5) %"116", align 4
  %11 = load i64, ptr addrspace(4) %"104", align 8
  store i64 %11, ptr addrspace(5) %"105", align 8
  %12 = load i64, ptr addrspace(5) %"105", align 8
  %13 = load i32, ptr addrspace(5) %"109", align 4
  %"163" = inttoptr i64 %12 to ptr addrspace(1)
  store i32 %13, ptr addrspace(1) %"163", align 4
  %14 = load i64, ptr addrspace(5) %"105", align 8
  %"164" = inttoptr i64 %14 to ptr addrspace(1)
  %"85" = getelementptr inbounds i8, ptr addrspace(1) %"164", i64 4
  %15 = load i32, ptr addrspace(5) %"110", align 4
  store i32 %15, ptr addrspace(1) %"85", align 4
  %16 = load i64, ptr addrspace(5) %"105", align 8
  %"165" = inttoptr i64 %16 to ptr addrspace(1)
  %"87" = getelementptr inbounds i8, ptr addrspace(1) %"165", i64 8
  %17 = load i32, ptr addrspace(5) %"111", align 4
  store i32 %17, ptr addrspace(1) %"87", align 4
  %18 = load i64, ptr addrspace(5) %"105", align 8
  %"166" = inttoptr i64 %18 to ptr addrspace(1)
  %"89" = getelementptr inbounds i8, ptr addrspace(1) %"166", i64 12
  %19 = load i32, ptr addrspace(5) %"112", align 4
  store i32 %19, ptr addrspace(1) %"89", align 4
  %20 = load i64, ptr addrspace(5) %"105", align 8
  %"167" = inttoptr i64 %20 to ptr addrspace(1)
  %"91" = getelementptr inbounds i8, ptr addrspace(1) %"167", i64 16
  %21 = load i32, ptr addrspace(5) %"113", align 4
  store i32 %21, ptr addrspace(1) %"91", align 4
  %22 = load i64, ptr addrspace(5) %"105", align 8
  %"168" = inttoptr i64 %22 to ptr addrspace(1)
  %"93" = getelementptr inbounds i8, ptr addrspace(1) %"168", i64 20
  %23 = load i32, ptr addrspace(5) %"114", align 4
  store i32 %23, ptr addrspace(1) %"93", align 4
  %24 = load i64, ptr addrspace(5) %"105", align 8
  %"169" = inttoptr i64 %24 to ptr addrspace(1)
  %"95" = getelementptr inbounds i8, ptr addrspace(1) %"169", i64 24
  %25 = load i32, ptr addrspace(5) %"115", align 4
  store i32 %25, ptr addrspace(1) %"95", align 4
  %26 = load i64, ptr addrspace(5) %"105", align 8
  %"170" = inttoptr i64 %26 to ptr addrspace(1)
  %"97" = getelementptr inbounds i8, ptr addrspace(1) %"170", i64 28
  %27 = load i32, ptr addrspace(5) %"116", align 4
  store i32 %27, ptr addrspace(1) %"97", align 4
  br label %"13"

"13":                                             ; preds = %"32", %"98"
  ret void
}

attributes #0 = { denormal_fpenv(dynamic) "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { denormal_fpenv(preservesign) "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
