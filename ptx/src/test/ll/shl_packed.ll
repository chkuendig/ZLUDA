define amdgpu_kernel void @shl_packed(ptr addrspace(4) byref(i64) %"72", ptr addrspace(4) byref(i64) %"73") #0 {
  %"74" = alloca i64, align 8, addrspace(5)
  %"75" = alloca i64, align 8, addrspace(5)
  %"76" = alloca i32, align 4, addrspace(5)
  %"77" = alloca i32, align 4, addrspace(5)
  %"78" = alloca i32, align 4, addrspace(5)
  %"79" = alloca i32, align 4, addrspace(5)
  %"80" = alloca i32, align 4, addrspace(5)
  %"81" = alloca i32, align 4, addrspace(5)
  %"82" = alloca i32, align 4, addrspace(5)
  %"83" = alloca i32, align 4, addrspace(5)
  %"84" = alloca i32, align 4, addrspace(5)
  %"85" = alloca i32, align 4, addrspace(5)
  %"86" = alloca i32, align 4, addrspace(5)
  %"87" = alloca i32, align 4, addrspace(5)
  %"88" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"71"

"71":                                             ; preds = %1
  fence seq_cst
  %2 = load i64, ptr addrspace(4) %"72", align 8
  store i64 %2, ptr addrspace(5) %"74", align 8
  %3 = load i64, ptr addrspace(4) %"73", align 8
  store i64 %3, ptr addrspace(5) %"75", align 8
  %4 = load i64, ptr addrspace(5) %"74", align 8
  %"136" = inttoptr i64 %4 to ptr
  %5 = load i32, ptr %"136", align 4
  store i32 %5, ptr addrspace(5) %"76", align 4
  %6 = load i64, ptr addrspace(5) %"74", align 8
  %"137" = inttoptr i64 %6 to ptr
  %"51" = getelementptr inbounds i8, ptr %"137", i64 4
  %7 = load i32, ptr %"51", align 4
  store i32 %7, ptr addrspace(5) %"77", align 4
  %8 = load i64, ptr addrspace(5) %"74", align 8
  %"138" = inttoptr i64 %8 to ptr
  %"53" = getelementptr inbounds i8, ptr %"138", i64 8
  %9 = load i32, ptr %"53", align 4
  store i32 %9, ptr addrspace(5) %"78", align 4
  %10 = load i64, ptr addrspace(5) %"74", align 8
  %"139" = inttoptr i64 %10 to ptr
  %"55" = getelementptr inbounds i8, ptr %"139", i64 12
  %11 = load i32, ptr %"55", align 4
  store i32 %11, ptr addrspace(5) %"79", align 4
  %12 = load i32, ptr addrspace(5) %"76", align 4
  %13 = shl i32 %12, 4
  %"140" = select i1 false, i32 0, i32 %13
  store i32 %"140", ptr addrspace(5) %"80", align 4
  %14 = load i32, ptr addrspace(5) %"77", align 4
  %15 = shl i32 %14, 4
  %"142" = select i1 false, i32 0, i32 %15
  store i32 %"142", ptr addrspace(5) %"81", align 4
  %16 = load i32, ptr addrspace(5) %"78", align 4
  %17 = shl i32 %16, 4
  %"144" = select i1 false, i32 0, i32 %17
  store i32 %"144", ptr addrspace(5) %"82", align 4
  %18 = load i32, ptr addrspace(5) %"79", align 4
  %19 = shl i32 %18, 4
  %"146" = select i1 false, i32 0, i32 %19
  store i32 %"146", ptr addrspace(5) %"83", align 4
  %20 = load i32, ptr addrspace(5) %"80", align 4
  %"148" = and i32 %20, 240
  store i32 %"148", ptr addrspace(5) %"80", align 4
  %21 = load i32, ptr addrspace(5) %"81", align 4
  %"150" = and i32 %21, 240
  store i32 %"150", ptr addrspace(5) %"81", align 4
  %22 = load i32, ptr addrspace(5) %"82", align 4
  %"152" = and i32 %22, 240
  store i32 %"152", ptr addrspace(5) %"82", align 4
  %23 = load i32, ptr addrspace(5) %"83", align 4
  %"154" = and i32 %23, 240
  store i32 %"154", ptr addrspace(5) %"83", align 4
  store i32 1, ptr addrspace(5) %"88", align 4
  %24 = load i32, ptr addrspace(5) %"88", align 4
  %25 = load i32, ptr addrspace(5) %"80", align 4
  %26 = icmp uge i32 %25, 32
  %27 = shl i32 %24, %25
  %"156" = select i1 %26, i32 0, i32 %27
  store i32 %"156", ptr addrspace(5) %"84", align 4
  %28 = load i32, ptr addrspace(5) %"88", align 4
  %29 = load i32, ptr addrspace(5) %"81", align 4
  %30 = icmp uge i32 %29, 32
  %31 = shl i32 %28, %29
  %"158" = select i1 %30, i32 0, i32 %31
  store i32 %"158", ptr addrspace(5) %"85", align 4
  %32 = load i32, ptr addrspace(5) %"88", align 4
  %33 = load i32, ptr addrspace(5) %"82", align 4
  %34 = icmp uge i32 %33, 32
  %35 = shl i32 %32, %33
  %"160" = select i1 %34, i32 0, i32 %35
  store i32 %"160", ptr addrspace(5) %"86", align 4
  %36 = load i32, ptr addrspace(5) %"88", align 4
  %37 = load i32, ptr addrspace(5) %"83", align 4
  %38 = icmp uge i32 %37, 32
  %39 = shl i32 %36, %37
  %"162" = select i1 %38, i32 0, i32 %39
  store i32 %"162", ptr addrspace(5) %"87", align 4
  %40 = load i64, ptr addrspace(5) %"75", align 8
  %41 = load i32, ptr addrspace(5) %"84", align 4
  %"164" = inttoptr i64 %40 to ptr
  store i32 %41, ptr %"164", align 4
  %42 = load i64, ptr addrspace(5) %"75", align 8
  %"165" = inttoptr i64 %42 to ptr
  %"66" = getelementptr inbounds i8, ptr %"165", i64 4
  %43 = load i32, ptr addrspace(5) %"85", align 4
  store i32 %43, ptr %"66", align 4
  %44 = load i64, ptr addrspace(5) %"75", align 8
  %"166" = inttoptr i64 %44 to ptr
  %"68" = getelementptr inbounds i8, ptr %"166", i64 8
  %45 = load i32, ptr addrspace(5) %"86", align 4
  store i32 %45, ptr %"68", align 4
  %46 = load i64, ptr addrspace(5) %"75", align 8
  %"167" = inttoptr i64 %46 to ptr
  %"70" = getelementptr inbounds i8, ptr %"167", i64 12
  %47 = load i32, ptr addrspace(5) %"87", align 4
  store i32 %47, ptr %"70", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
