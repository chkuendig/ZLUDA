// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(non_snake_case, improper_ctypes, improper_ctypes_definitions)]
#[cfg(zluda_wine_generation_phase_0)]
compile_error!("bridge generation failed: shipping admission");
#[cfg(zluda_wine_generation_phase_1)]
compile_error!("bridge generation failed: canonical CUDA types");
#[cfg(zluda_wine_generation_phase_2)]
compile_error!("bridge generation failed: canonical dark metadata");
#[cfg(zluda_wine_generation_phase_3)]
compile_error!("bridge generation failed: PE32 declarations");
#[cfg(zluda_wine_generation_phase_4)]
compile_error!("bridge generation failed: canonical integrity view");
#[cfg(zluda_wine_generation_phase_5)]
compile_error!("bridge generation failed: public declaration and backend refusal guard");
#[cfg(zluda_wine_generation_phase_6)]
compile_error!("bridge generation failed: typed public surface");
#[cfg(zluda_wine_generation_phase_7)]
compile_error!("bridge generation failed: generated transport outputs");
#[cfg(not(zluda_wine_generation_failed))]
include!("bridge.rs");
