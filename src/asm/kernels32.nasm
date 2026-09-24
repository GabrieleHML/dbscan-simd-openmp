; SSE kernels for x86-32, single precision (prototypes in src/kernels.h).
;
; cdecl calling convention: arguments on the stack, float results in st(0),
; EBX, ESI, EDI and EBP preserved. These functions only use EAX, ECX, EDX and
; XMM registers, which are all caller-saved, so they save nothing.

section .text

global dist2_simd
global fill_labels_simd

; float dist2_simd(const float *a, const float *b, int d)
; a and b 16-byte aligned, d a positive multiple of 4.
dist2_simd:
        mov     eax, [esp+4]            ; a
        mov     ecx, [esp+8]            ; b
        mov     edx, [esp+12]           ; d
        shl     edx, 2                  ; row length in bytes
        add     eax, edx                ; point past the end of both rows and walk them
        add     ecx, edx                ; with a negative offset: the loop ends when it
        neg     edx                     ; reaches zero, with no extra compare
        xorps   xmm0, xmm0              ; 4 partial sums
.loop:
        movaps  xmm1, [eax+edx]         ; 4 coordinates of a
        subps   xmm1, [ecx+edx]         ; minus 4 coordinates of b
        mulps   xmm1, xmm1
        addps   xmm0, xmm1
        add     edx, 16
        jnz     .loop
        haddps  xmm0, xmm0              ; SSE3 horizontal sum of the partial sums
        haddps  xmm0, xmm0
        ; The x87 stack can only be loaded from memory: use a slot on the stack, which,
        ; unlike a global variable, is private to each thread
        sub     esp, 4
        movss   [esp], xmm0
        fld     dword [esp]
        add     esp, 4
        ret

; void fill_labels_simd(int *labels, int n, int value)
; labels 16-byte aligned.
fill_labels_simd:
        mov     eax, [esp+4]            ; labels
        mov     ecx, [esp+8]            ; n
        mov     edx, [esp+12]           ; value
        movd    xmm0, edx
        pshufd  xmm0, xmm0, 0           ; value in all 4 lanes
.vector:
        cmp     ecx, 4
        jl      .tail
        movdqa  [eax], xmm0             ; 4 labels per store
        add     eax, 16
        sub     ecx, 4
        jmp     .vector
.tail:
        test    ecx, ecx
        jle     .done
        mov     [eax], edx
        add     eax, 4
        dec     ecx
        jmp     .tail
.done:
        ret

section .note.GNU-stack noalloc noexec nowrite progbits
