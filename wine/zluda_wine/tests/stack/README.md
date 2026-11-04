# Compiler calling-stack measurement

This fixture measures the real `compiler` crate's `zoc`: checked PTX parsing,
`ptx::to_llvm_module`, LLVM O3/codegen and in-process LLD. An explicit
`--arch gfx1030` bypasses HIP discovery. No GPU is accessed or kernel launched.
Its result qualifies the compiler pipeline, not Wine's complete cuModuleLoad
stack, every possible PTX program or LLVM worker-thread stacks. Those remain W1.

Build/retain zoc (`cargo build --release --locked --offline -p compiler -j3`),
its exact ELF interpreter/DT_NEEDED closure and `build-probe.sh`'s test-only
preload in the approved hosted SDK. Keep the loader, runtime/source hashes and
compiler versions. Never install the preload in a tool or add a stack override.

Run `measure.py --help` only after coordinator clearance for the exact local
container. Its inputs are absolute raw zoc, loader, library directory, preload,
local corpus manifest, selected module SHA256 and fresh results directory.
It invokes the retained loader with `--preload`; do not preload a shell wrapper.
One invocation measures one module serially with three paint seeds, a known
64 KiB recursion calibration and a checked guard fault. Each compile has a 900s
watchdog. A compile failure, timeout or guard failure leaves qualification pending.

`stack-probe.c` interposes libc startup and uses ucontext on the **same OS
thread/TLS**. Its1 MiB mapping includes a4 KiB low guard; an additional high guard
lies outside that reserve. After returning to the original stack, the probe
scans stack paint. It retains touched high-water, a page-rounded observation
with 128-byte red-zone allowance, thread IDs, raw statuses and products' hashes.
Paint observes writes, not every untouched stack-pointer reservation. Three
seeds and calibration reduce blind spots; this is measured corpus evidence,
never a universal maximum-depth proof. Rust/LLVM helper threads are outside
the measured calling stack. Overflow selftest runs before Rust initialization.

Game/NVIDIA PTX and resulting compiler products stay in local evidence. Commit
only harness code and metadata/hashes; do not upload PTX, LLVM IR, assembly or
compiled game ELF products to GitHub. Choose largest/deepest retained PhysX
modules with provenance, including Knight and available Mafia/APEX cases.
Module size and lexical nesting guide selection; they do not predict stack use.

Compare the largest observed bound against1 MiB minus guard/minimum Wine frame
margin. W1 must confirm the full Wine/cuModuleLoad path and game-thread C
baseline. Only measured headroom below approximately 2x warrants proposing the
smallest floor giving at least 2x, with explicit review and on-root-only launcher
policy. No floor is set by this fixture. At Wine46b29104, native kernel stacks
are allocated with 4 GiB as a **lower** bound, including WoW64; the 32-bit game
stack is a separate allocation with a32-bit upper bound. Native stack reserves
do not consume Asylum's low4 GiB address space, but still consume native memory.

## Retained local measurement, 2026-10-30

The immutable zoc/runtime capsule from run41034209983 (source 51ec74f1) was
measured locally on gfx1030 under separate exact-command clearance. Six
retained modules passed three compiles each, plus one recursion calibration
and one checked guard fault per module. Each compile stayed on its original
OS thread and completed ELF code generation/linking.

| Module SHA256 prefix | Selection | Rounded observed bound | Conservative pipeline headroom |
|---|---|---:|---:|
| 5dea9d868d53 | Largest retained IOFX,3,388,001 bytes | 76 KiB | 13.26x |
| 8358e54f033b | Large retained PhysX,3,102,719 bytes | 76 KiB | 13.26x |
| 948d1118dc59 | Large retained PhysX,3,030,651 bytes | 76 KiB | 13.26x |
| a38e6f5648b4 | Fluid/runtime PhysX,2,896,563 bytes | 76 KiB | 13.26x |
| b634591a8d2f | Knight interop,2,148,296 bytes | 108 KiB | 9.33x |
| 9bd0f08b8c2a | Knight interop,218,975 bytes | 76 KiB | 13.26x |

Worst touched high-water was 106,536 bytes; worst page-rounded bound 110,592
bytes includes the 128-byte red-zone allowance. Comparison budget is 1,032,192
bytes: 1 MiB minus 4 KiB guard and 12 KiB Wine minimum-frame margin. The exact
module/toolchain/runtime/product hashes, observations and raw-receipt hashes
are in MEASURED-41034209983.json. Only metadata/hashes are committed; all PTX
and compiler products remain local. No Mafia attribution is inferred for the
unattributed corpus modules. This result gives no reason for a stack floor;
full Wine/cuModuleLoad and C effective-stack confirmation remain W1.
