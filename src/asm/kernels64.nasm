; AVX kernels for x86-64, double precision (prototypes in src/kernels.h).
;
; System V AMD64 calling convention: arguments in RDI, RSI, RDX, floating-point
; result in XMM0. Only caller-saved registers are used, so nothing is saved.
; Every exit runs VZEROUPPER, so the SSE code compiled by GCC in the caller does
; not pay the AVX-SSE transition penalty.

default rel
section .text

global dist2_simd
global fill_labels_simd

; double dist2_simd(const double *a, const double *b, int d)
; a and b 32-byte aligned, d a positive multiple of 4.
dist2_simd:
        movsxd  rdx, edx
        shl     rdx, 3                  ; row length in bytes
        add     rdi, rdx                ; negative offset loop, as in kernels32.nasm
        add     rsi, rdx
        neg     rdx
        vxorpd  ymm0, ymm0, ymm0        ; 4 partial sums
.loop:
        vmovapd ymm1, [rdi+rdx]         ; 4 coordinates of a
        vsubpd  ymm1, ymm1, [rsi+rdx]   ; minus 4 coordinates of b
        vmulpd  ymm1, ymm1, ymm1
        vaddpd  ymm0, ymm0, ymm1
        add     rdx, 32
        jnz     .loop
        ; VHADDPD on YMM adds within each 128-bit half, so fold the halves first
        vextractf128 xmm1, ymm0, 1
        vaddpd  xmm0, xmm0, xmm1
        vhaddpd xmm0, xmm0, xmm0
        vzeroupper
        ret

; void fill_labels_simd(int *labels, int n, int value)
; labels 32-byte aligned.
fill_labels_simd:
        vmovd   xmm0, edx
        vpshufd xmm0, xmm0, 0           ; value in 4 lanes
        vinsertf128 ymm0, ymm0, xmm0, 1 ; and in 8 (AVX has no 256-bit integer shuffles)
.vector:
        cmp     esi, 8
        jl      .tail
        vmovdqa [rdi], ymm0             ; 8 labels per store
        add     rdi, 32
        sub     esi, 8
        jmp     .vector
.tail:
        test    esi, esi
        jle     .done
        mov     [rdi], edx
        add     rdi, 4
        dec     esi
        jmp     .tail
.done:
        vzeroupper
        ret

section .note.GNU-stack noalloc noexec nowrite progbits
