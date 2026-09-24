; ---------------------------------------------------------
; Regression con istruzioni AVX a 64 bit
; ---------------------------------------------------------
; F. Angiulli
; 23/11/2017
;

;
; Software necessario per l'esecuzione:
;
;     NASM (www.nasm.us)
;     GCC (gcc.gnu.org)
;
; entrambi sono disponibili come pacchetti software 
; installabili mediante il packaging tool del sistema 
; operativo; per esempio, su Ubuntu, mediante i comandi:
;
;     sudo apt-get install nasm
;     sudo apt-get install gcc
;
; potrebbe essere necessario installare le seguenti librerie:
;
;     sudo apt-get install lib32gcc-4.8-dev (o altra versione)
;     sudo apt-get install libc6-dev-i386
;
; Per generare file oggetto:
;
;     nasm -f elf64 regression64.nasm
;

%include "sseutils64.nasm"

section .data			; Sezione contenente dati inizializzati
	align 32
	unclassified dd -2, -2, -2, -2, -2, -2, -2, -2
	value dq 5.5
section .bss			; Sezione contenente dati non inizializzati

alignb 32
sc		resq		1
temp_float resq 1  ; Definisci una variabile temporanea in memoria per il float

section .text			; Sezione contenente il codice macchina

; ----------------------------------------------------------
; macro per l'allocazione dinamica della memoria
;
;	getmem	<size>,<elements>
;
; alloca un'area di memoria di <size>*<elements> bytes
; (allineata a 16 bytes) e restituisce in EAX
; l'indirizzo del primo bytes del blocco allocato
; (funziona mediante chiamata a funzione C, per cui
; altri registri potrebbero essere modificati)
;
;	fremem	<address>
;
; dealloca l'area di memoria che ha inizio dall'indirizzo
; <address> precedentemente allocata con getmem
; (funziona mediante chiamata a funzione C, per cui
; altri registri potrebbero essere modificati)

extern get_block
extern free_block

%macro	getmem	2
	mov	rdi, %1
	mov	rsi, %2
	call	get_block
%endmacro

%macro	fremem	1
	mov	rdi, %1
	call	free_block
%endmacro

; ------------------------------------------------------------
; Funzione prova
; ------------------------------------------------------------
global riempi_out_con_meno_due

global prova
global distanza_euclidea_assembly

msg	db 'sc:',32,0
nl	db 10,0

prova:
		; ------------------------------------------------------------
		; Sequenza di ingresso nella funzione
		; ------------------------------------------------------------
		push		rbp				; salva il Base Pointer
		mov		rbp, rsp			; il Base Pointer punta al Record di Attivazione corrente
		pushaq						; salva i registri generali
		;è sbagliato fare pushaq e popaq perchè altrimenti si salvano e ripristinano tutti i registri
		;anche quelli xmm, e quindi non si riesce a restituire nulla...
		;Il template dunque è sbagliato.
		;G.

		; ------------------------------------------------------------
		; I parametri sono passati nei registri
		; ------------------------------------------------------------
		; rdi = indirizzo della struct input
		
		; esempio: stampa input->sc
        ; [RDI] input->ds; 			// dataset
		; [RDI + 8] input->labels; 	// etichette
		; [RDI + 16] input->out;	// vettore contenente risultato dim=(k+1)
		; [RDI + 24] input->sc;		// score dell'insieme di features risultato
		; [RDI + 32] input->k; 		// numero di features da estrarre
		; [RDI + 36] input->N;		// numero di righe del dataset
		; [RDI + 40] input->d;		// numero di colonne/feature del dataset
		; [RDI + 44] input->display;
		; [RDI + 48] input->silent;
		VMOVSD		XMM0, [RDI+24]
		VMOVSD		[sc], XMM0
		prints 		msg
		printsd		sc
		prints 		nl
		; ------------------------------------------------------------
		; Sequenza di uscita dalla funzione
		; ------------------------------------------------------------
		
		popaq				; ripristina i registri generali
		mov		rsp, rbp	; ripristina lo Stack Pointer
		pop		rbp		; ripristina il Base Pointer
		ret				; torna alla funzione C chiamante


;void riempi_out_con_meno_due(params* p)
riempi_out_con_meno_due:
		; ------------------------------------------------------------
		; Sequenza di ingresso nella funzione
		; ------------------------------------------------------------
		push		rbp				; salva il Base Pointer
		mov		rbp, rsp			; il Base Pointer punta al Record di Attivazione corrente
		pushaq						; salva i registri generali
		; ------------------------------------------------------------
		; legge i parametri dal Record di Attivazione corrente
		; ------------------------------------------------------------
		; rdi = indirizzo della struct input
		; [RDI + 16] input->out;	// vettore contenente risultato dim=(k+1)
		; [RDI + 36] input->N;		// numero di righe del dataset
		mov rax, [rdi+16]	;rax = p->out
		mov edx, [rdi+36]	;ebp = p->N
		; ------------------------------------------------------------
		; elaborazione
		; ------------------------------------------------------------
		vmovaps ymm0, [unclassified]		;ymm0 = [-2,-2,-2,-2,-2,-2,-2,-2] 
		xor rsi, rsi 				;rsi = i = 0
		cmp edx, 8					;if N<8 andiamo direttamente al for classico
		jl residui_for_riempi_out_con_meno_due 
		sub edx, 8					;facciamo N = N-4 così che non ci siano problemi di SegFault
									;Così, se N=7, senza N = N-4, succederebbe che al primo ciclo
									;assegnamo out[i] con i=0,1,2,3, al secondo ciclo verifichiamo
									;i = 4 < N = 7, dunque rimaniamo nel ciclo e assegnamo out[i] con
									;i=4,5,6,7, ma scrivendo in out[7] provochiamo una SegFault. Quindi
									;con N = N - 4 ciò non può succedere.
		for_riempi_out_con_meno_due:
			cmp esi, edx 				;if i>=N
			jge residui_riempi_out_con_meno_due
			vmovaps [rax+rsi*4], ymm0	;out[i]=-2,out[i+1]=-2,out[i+2]=-2,out[i+3]=-2, (rimane *4 perchè comunque gli interi sono a 32 bit, non 64)
			add esi, 8					;i+=4, perchè inseriamo 4 valori per volta.
			jmp for_riempi_out_con_meno_due
		residui_riempi_out_con_meno_due:
			add edx, 8						;ripristiniamo il valore originale di N
											;Non dovremmo anche decrementare i? No, perchè se prima N=3,
											;allora bbaimo allocato out[i] con i=0,1,2,3, dunque abbiamo
											;allocato un numero di posizioni pari a quelle che avremmo
											;allocato se N fosse stato pari a 4. Dunque non bisogna decrementare
											;i, va bene così.
			residui_for_riempi_out_con_meno_due:
				cmp esi, edx				;if i>=N
				jge fine_for_riempi_out_con_meno_due
				mov [rax+rsi*4], dword -2		;out[i]=-2
				inc esi						;i++
				jmp residui_for_riempi_out_con_meno_due
		fine_for_riempi_out_con_meno_due:
		; ------------------------------------------------------------
		; Sequenza di uscita dalla funzione
		; ------------------------------------------------------------
		popaq				; ripristina i registri generali
		mov		rsp, rbp	; ripristina lo Stack Pointer
		pop		rbp		; ripristina il Base Pointer
		ret				; torna alla funzione C chiamante



;type distanza_euclidea(params* input, int p1, int p2)
distanza_euclidea_assembly:
		; ------------------------------------------------------------
		; Sequenza di ingresso nella funzione
		; ------------------------------------------------------------
		push		rbp				; salva il Base Pointer
		mov		rbp, rsp			; il Base Pointer punta al Record di Attivazione corrente
		push rax					;salviamo i registri generali
		push rbx
		push rcx
		push rdi
		; ------------------------------------------------------------
		; legge i parametri dal Record di Attivazione corrente
		; ------------------------------------------------------------
		; rdi = indirizzo della struct input
		; [RDI] input->ds;	// vettore contenente risultato dim=(k+1)
		; [RDI + 40] input->d;		// numero di righe del dataset
		; rsi = p1
		; rdx = p2
		mov rax, [rdi]	;rax = &ds
		mov rbx, rsi	;rbx = p1
		mov rcx, rdx	;rcx = p2
		mov edi, [rdi+40]	;rdi = p->d
		; ------------------------------------------------------------
		; elaborazione
		; ------------------------------------------------------------
		vpxor ymm1, ymm1					;ymm1 = somma = 0.0		
		xor rsi, rsi 				;esi = i = 0
		imul ebx, edi    ; ebx = ebx * edi, quindi facciamo "p1*input->d"
		imul ecx, edi    ; ecx = ecx * edi, quindi facciamo "p2*input->d"

		cmp edi, 4					;if d<4 andiamo direttamente al for dei residui
		jl residui_for_distanza_euclidea 
		sub edi, 4					;facciamo d = d-4 così che non ci siano problemi di SegFault
		for_distanza_euclidea:
			cmp esi, edi 				;if i>=d
			jge residui_distanza_euclidea
			add ebx, esi	 ; ebx = ebx + esi, quindi abbiamo ebx = (p1*input->d)+i
			add ecx, esi	 ; ecx = ecx + esi, quindi abbiamo ecx = (p2*input->d)+i
			vmovapd ymm0, [rax+rbx*8]	;xmm0 = input->ds[(p1*input->d)+i], con i=0,1,2,3
										;c'è il 4 che moltiplica perchè non dobbiamo muoverci di
										;un indirizzo per ogni elemento di "ds", ma si 4 indirizzi
										;per ogni elemento di "ds" perchè "ds" contiene valori
										;floating point, funque valori memorizzati in 32 bit, e non 8!	
			vsubpd ymm0, [rax+rcx*8]		;xmm0 = input->ds[(p1*input->d)+i] - input->ds[(p2*input->d)+i]
			vmulpd ymm0, ymm0 			;xmm0 = differenza * differenza
			vaddpd ymm1, ymm0			;cumuliamo la somma, quindi xmm1 += differenza * differenza 
										;la riduzione poi la faremo fuori dal ciclo.
			sub ebx, esi	 ; ebx = ebx - esi, quindi abbiamo ebx = (p1*input->d)-i
			sub ecx, esi	 ; ecx = ecx - esi, quindi abbiamo ecx = (p2*input->d)-i
							 ; se non facessimo ciò, non aggiungeremmo "i" ad ogni iterazione, ma
							 ; alla prima aggiungeremmo 1, alla seconda 3 (1+2), alla terza 6 (3+3)
							 ; e dunque sarebbe sbagliato
			add esi, 4					;i+=4, perchè inseriamo 4 valori per volta.
			jmp for_distanza_euclidea
		residui_distanza_euclidea:
			add edi, 4						;ripristiniamo il valore originale di d
			residui_for_distanza_euclidea:
				cmp esi, edi 				;if i>=d
				jge fine_for_distanza_euclidea
				add ebx, esi	 ; ebx = ebx + esi, quindi abbiamo ebx = (p1*input->d)+i
				add ecx, esi	 ; ecx = ecx + esi, quindi abbiamo ecx = (p2*input->d)+i
				movsd xmm0, [rax+rbx*8]		;xmm0 = input->ds[(p1*input->d)+i], con i=0,1,2,3
				subsd xmm0, [rax+rcx*8]		;xmm0 = input->ds[(p1*input->d)+i] - input->ds[(p2*input->d)+i]
				mulsd xmm0, xmm0 			;xmm0 = differenza * differenza
				addsd xmm1, xmm0			;cumuliamo la somma, quindi xmm1 += differenza * differenza 
											;la riduzione la facciamo alla fine
				sub ebx, esi	 ; ebx = ebx - esi, quindi abbiamo ebx = (p1*input->d)-i
				sub ecx, esi	 ; ecx = ecx - esi, quindi abbiamo ecx = (p2*input->d)-i
								; se non facessimo ciò, non aggiungeremmo "i" ad ogni iterazione, ma
								; alla prima aggiungeremmo 1, alla seconda 3 (1+2), alla terza 6 (3+3)
								; e dunque sarebbe sbagliato
				inc esi						;i++
				jmp residui_for_distanza_euclidea
		fine_for_distanza_euclidea:
		;ora dobbiamo fare la riduzione
		vhaddpd ymm1, ymm1

		vextractf128 xmm8,ymm1,0; La vhaddpd funziona in modo brutto, perciò non si può iterare come a
    	vextractf128 xmm9,ymm1,1; 32 bit, ovvero come nell'estensione ss3, dunque per ottenere lo stesso
    	addsd xmm8,xmm9 		; risultato facciamo così.

		sqrtsd xmm8, xmm8		;calcoliamo la radice quadrata del valore presente in xmm1 e lo memorizziamo in xmm1
		;ora dobbiamo restituire questo valore a 32 bit
  		vmovsd xmm0, xmm8	
		; Carichiamo il valore 5.5 dal segmento dati nel registro xmm0
		; ------------------------------------------------------------
		; Sequenza di uscita dalla funzione
		; ------------------------------------------------------------
		pop rax
		pop rbx
		pop rcx
		pop rdi				; ripristina i registri generali
		;ASSOLUTAMENTE NON FARE PUSHAD E POPAD PERCHè SENNò POPPANO PURE I REGISTRI xmmi E QUINDI NON SI RESTITUISCE PROPRIO NULLA
		mov		rsp, rbp	; ripristina lo Stack Pointer
		pop		rbp		; ripristina il Base Pointer
		ret				; torna alla funzione C chiamante

