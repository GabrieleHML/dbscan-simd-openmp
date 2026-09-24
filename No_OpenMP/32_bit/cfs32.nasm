; ---------------------------------------------------------
; Regressione con istruzioni SSE a 32 bit
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
;     nasm -f elf32 fss32.nasm 
;
%include "sseutils32.nasm"

section .data			; Sezione contenente dati inizializzati
	align 16
	unclassified dd -2, -2, -2, -2

section .bss			; Sezione contenente dati non inizializzati
	alignb 16
	sc		resd		1
	temp_float resd 1  ; Definisci una variabile temporanea in memoria per il float

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
	mov	eax, %1
	push	eax
	mov	eax, %2
	push	eax
	call	get_block
	add	esp, 8
%endmacro

%macro	fremem	1
	push	%1
	call	free_block
	add	esp, 4
%endmacro

; ------------------------------------------------------------
; Funzioni
; ------------------------------------------------------------
global riempi_out_con_meno_due

global prova
global distanza_euclidea_assembly

input		equ		8

msg	db	'sc:',32,0
nl	db	10,0



prova:
		; ------------------------------------------------------------
		; Sequenza di ingresso nella funzione
		; ------------------------------------------------------------
		push		ebp		; salva il Base Pointer
		mov		ebp, esp	; il Base Pointer punta al Record di Attivazione corrente
		push		ebx		; salva i registri da preservare
		push		esi
		push		edi
		; ------------------------------------------------------------
		; legge i parametri dal Record di Attivazione corrente
		; ------------------------------------------------------------

		; elaborazione
		
		; esempio: stampa input->sc
		mov EAX, [EBP+input]	; indirizzo della struttura contenente i parametri
        ; [EAX] input->ds; 			// dataset
		; [EAX + 4] input->labels; 	// etichette
		; [EAX + 8] input->out;	// vettore contenente risultato dim=(k+1)
		; [EAX + 12] input->sc;		// score dell'insieme di features risultato
		; [EAX + 16] input->k; 		// numero di features da estrarre
		; [EAX + 20] input->N;		// numero di righe del dataset
		; [EAX + 24] input->d;		// numero di colonne/feature del dataset
		; [EAX + 28] input->display;
		; [EAX + 32] input->silent;
		MOVSS XMM0, [EAX+12]
		MOVSS [sc], XMM0 
		prints msg            
		printss sc     
		prints nl
		; ------------------------------------------------------------
		; Sequenza di uscita dalla funzione
		; ------------------------------------------------------------

		pop	edi		; ripristina i registri da preservare
		pop	esi
		pop	ebx
		mov	esp, ebp	; ripristina lo Stack Pointer
		pop	ebp		; ripristina il Base Pointer
		ret			; torna alla funzione C chiamante

;void riempi_out_con_meno_due(params* p)
riempi_out_con_meno_due:
		; ------------------------------------------------------------
		; Sequenza di ingresso nella funzione
		; ------------------------------------------------------------
		push		ebp		; salva il Base Pointer
		mov		ebp, esp	; il Base Pointer punta al Record di Attivazione corrente
		push		ebx		; salva i registri da preservare
		push		esi
		push		edi
		; ------------------------------------------------------------
		; legge i parametri dal Record di Attivazione corrente
		; ------------------------------------------------------------
		mov eax, [ebp+input];ricordiamo che il parametro passatò è un indirizzo, quindi bisogna sempre fare un doppio accesso alla memoria
		mov edi, [eax+20]	;edi = p->N
		mov eax, [eax+8]	;eax = p->out
		; ------------------------------------------------------------
		; elaborazione
		; ------------------------------------------------------------
		movaps xmm0, [unclassified]		;xmm0 = [-2,-2,-2,-2] 
		xor esi, esi 				;esi = i = 0
		cmp edi, 4					;if N<4 andiamo direttamente al for classico
		jl classic_for_riempi_out_con_meno_due 
		sub edi, 4					;facciamo N = N-4 così che non ci siano problemi di SegFault
									;Così, se N=7, senza N = N-4, succederebbe che al primo ciclo
									;assegnamo out[i] con i=0,1,2,3, al secondo ciclo verifichiamo
									;i = 4 < N = 7, dunque rimaniamo nel ciclo e assegnamo out[i] con
									;i=4,5,6,7, ma scrivendo in out[7] provochiamo una SegFault. Quindi
									;con N = N - 4 ciò non può succedere.
		for_riempi_out_con_meno_due:
			cmp esi, edi 				;if i>=N
			jge residui_riempi_out_con_meno_due
			movaps [eax+esi*4], xmm0	;out[i]=-2,out[i+1]=-2,out[i+2]=-2,out[i+3]=-2,
			add esi, 4					;i+=4, perchè inseriamo 4 valori per volta.
			jmp for_riempi_out_con_meno_due
		residui_riempi_out_con_meno_due:
			add edi, 4						;ripristiniamo il valore originale di N
											;Non dovremmo anche decrementare i? No, perchè se prima N=3,
											;allora bbaimo allocato out[i] con i=0,1,2,3, dunque abbiamo
											;allocato un numero di posizioni pari a quelle che avremmo
											;allocato se N fosse stato pari a 4. Dunque non bisogna decrementare
											;i, va bene così.
			residui_for_riempi_out_con_meno_due:
				cmp esi, edi 				;if i>=N
				jge fine_for_riempi_out_con_meno_due
				mov [eax+esi*4], dword -2		;out[i]=-2
				inc esi						;i++
				jmp residui_for_riempi_out_con_meno_due
		classic_for_riempi_out_con_meno_due:
			cmp esi, edi 				;if i>=N
			jge fine_for_riempi_out_con_meno_due
			mov [eax+esi*4], dword -2			;out[i]=-2
			inc esi						;i++
			jmp classic_for_riempi_out_con_meno_due
		fine_for_riempi_out_con_meno_due:
		; ------------------------------------------------------------
		; Sequenza di uscita dalla funzione
		; ------------------------------------------------------------
		pop	edi		; ripristina i registri da preservare
		pop	esi
		pop	ebx
		mov	esp, ebp	; ripristina lo Stack Pointer
		pop	ebp		; ripristina il Base Pointer
		ret			; torna alla funzione C chiamante


;type distanza_euclidea(params* input, int p1, int p2)
distanza_euclidea_assembly:
		; ------------------------------------------------------------
		; Sequenza di ingresso nella funzione
		; ------------------------------------------------------------
		push		ebp		; salva il Base Pointer
		mov		ebp, esp	; il Base Pointer punta al Record di Attivazione corrente
		push		ebx		; salva i registri da preservare
		push		esi
		push		edi
		; ------------------------------------------------------------
		; legge i parametri dal Record di Attivazione corrente
		; ------------------------------------------------------------
		mov eax, [ebp+input];ricordiamo che il parametro passatò è un indirizzo, quindi bisogna sempre fare un doppio accesso alla memoria
		mov ebx, [ebp+12]	;ebx = p1
		mov ecx, [ebp+16]	;ecx = p2
		mov edi, [eax+24]	;edi = p->d
		mov eax, [eax]	;eax = &ds
		; ------------------------------------------------------------
		; elaborazione
		; ------------------------------------------------------------
		pxor xmm1, xmm1					;xmm1 = somma = 0.0		
		xor esi, esi 				;esi = i = 0
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
			movaps xmm0, [eax+ebx*4]	;xmm0 = input->ds[(p1*input->d)+i], con i=0,1,2,3
										;c'è il 4 che moltiplica perchè non dobbiamo muoverci di
										;un indirizzo per ogni elemento di "ds", ma si 4 indirizzi
										;per ogni elemento di "ds" perchè "ds" contiene valori
										;floating point, funque valori memorizzati in 32 bit, e non 8!					
			subps xmm0, [eax+ecx*4]		;xmm0 = input->ds[(p1*input->d)+i] - input->ds[(p2*input->d)+i]
			mulps xmm0, xmm0 			;xmm0 = differenza * differenza
			addps xmm1, xmm0			;cumuliamo la somma, quindi xmm1 += differenza * differenza 
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
				movss xmm0, [eax+ebx*4]		;xmm0 = input->ds[(p1*input->d)+i], con i=0,1,2,3
				subss xmm0, [eax+ecx*4]		;xmm0 = input->ds[(p1*input->d)+i] - input->ds[(p2*input->d)+i]
				mulss xmm0, xmm0 			;xmm0 = differenza * differenza
				addss xmm1, xmm0			;cumuliamo la somma, quindi xmm1 += differenza * differenza 
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
		haddps xmm1, xmm1
		haddps xmm1, xmm1			;con queste due hadd abbiamo sommato i 4 elementi presenti in xmm1
		sqrtps xmm1, xmm1		;calcoliamo la radice quadrata del valore presente in xmm1 e lo memorizziamo in xmm1
		;ora dobbiamo restituire questo valore a 32 bit
  		movq [temp_float], xmm1		; Memorizza il float contenuto in xmm1 nella variabile temporanea
									;perchè per spostare un valore nello stack dei registri x87
									;serve prima spostare il valore in memoria
		fld dword [temp_float] 		; fld carica un valore a virgola mobile a precisione singola nello stack FPU

		; ------------------------------------------------------------
		; Sequenza di uscita dalla funzione
		; ------------------------------------------------------------
		pop	edi		; ripristina i registri da preservare
		pop	esi
		pop	ebx
		mov	esp, ebp	; ripristina lo Stack Pointer
		pop	ebp		; ripristina il Base Pointer
		ret			; torna alla funzione C chiamante


;type distanza_euclidea_rispetto_a(params* input, type* p1, int p2)
;distanza_euclidea_rispetto_a:
;		; ------------------------------------------------------------
;		; Sequenza di ingresso nella funzione
;		; ------------------------------------------------------------
;		push		ebp		; salva il Base Pointer
;		mov		ebp, esp	; il Base Pointer punta al Record di Attivazione corrente
;		push		ebx		; salva i registri da preservare
;		push		esi
;		push		edi
;		; ------------------------------------------------------------
;		; legge i parametri dal Record di Attivazione corrente
;		; ------------------------------------------------------------
;		mov eax, [ebp+input];ricordiamo che il parametro passatò è un indirizzo, quindi bisogna sempre fare un doppio accesso alla memoria
;		mov ebx, [ebp+12]	;ebx = &p1
;		mov ecx, [ebp+16]	;ecx = p2
;		mov edi, [eax+24]	;edi = p->d
;		mov eax, [eax]	;eax = &ds
;		; ------------------------------------------------------------
;		; elaborazione
;		; ------------------------------------------------------------
;		pxor xmm1, xmm1					;xmm1 = somma = 0.0		
;		xor esi, esi 				;esi = i = 0
;		imul ecx, edi    ; ecx = ecx * edi, quindi facciamo "p2*input->d"
;
;		cmp edi, 4					;if d<4 andiamo direttamente al for dei residui
;		jl residui_for_distanza_euclidea_rispetto_a 
;		sub edi, 4					;facciamo d = d-4 così che non ci siano problemi di SegFault
;		for_distanza_euclidea_rispetto_a:
;			cmp esi, edi 				;if i>=d
;			jge residui_distanza_euclidea_rispetto_a
;			add ecx, esi	 ; ecx = ecx + esi, quindi abbiamo ecx = (p2*input->d)+i
;			movups xmm0, [ebx+esi*4]	;xmm0 = p1[i], con i=0,1,2,3 (usiamo movUps perchè non è detto che sia allineato p1)
;										;c'è il 4 che moltiplica perchè non dobbiamo muoverci di
;										;un indirizzo per ogni elemento di "ds", ma si 4 indirizzi
;										;per ogni elemento di "ds" perchè "ds" contiene valori
;										;floating point, funque valori memorizzati in 32 bit, e non 8!	
;			subps xmm0, [eax+ecx*4]		;xmm0 = input->ds[(p1*input->d)+i] - input->ds[(p2*input->d)+i]
;			mulps xmm0, xmm0 			;xmm0 = differenza * differenza
;			addps xmm1, xmm0			;cumuliamo la somma, quindi xmm1 += differenza * differenza 
;										;la riduzione poi la faremo fuori dal ciclo.
;			sub ecx, esi	 ; ecx = ecx - esi, quindi abbiamo ecx = (p2*input->d)-i
;							 ; se non facessimo ciò, non aggiungeremmo "i" ad ogni iterazione, ma
;							 ; alla prima aggiungeremmo 1, alla seconda 3 (1+2), alla terza 6 (3+3)
;							 ; e dunque sarebbe sbagliato
;			add esi, 4					;i+=4, perchè inseriamo 4 valori per volta.
;			jmp for_distanza_euclidea_rispetto_a
;		residui_distanza_euclidea_rispetto_a:
;			add edi, 4						;ripristiniamo il valore originale di d
;			residui_for_distanza_euclidea_rispetto_a:
;				cmp esi, edi 				;if i>=d
;				jge fine_for_distanza_euclidea_rispetto_a
;				add ecx, esi	 ; ecx = ecx + esi, quindi abbiamo ecx = (p2*input->d)+i
;				movss xmm0, [ebx+esi*4]		;xmm0 = input->ds[(p1*input->d)+i], con i=0,1,2,3
;				subss xmm0, [eax+ecx*4]		;xmm0 = input->ds[(p1*input->d)+i] - input->ds[(p2*input->d)+i]
;				mulss xmm0, xmm0 			;xmm0 = differenza * differenza
;				addss xmm1, xmm0			;cumuliamo la somma, quindi xmm1 += differenza * differenza 
;											;la riduzione la facciamo alla fine
;				sub ecx, esi	 ; ecx = ecx - esi, quindi abbiamo ecx = (p2*input->d)-i
;								; se non facessimo ciò, non aggiungeremmo "i" ad ogni iterazione, ma
;								; alla prima aggiungeremmo 1, alla seconda 3 (1+2), alla terza 6 (3+3)
;								; e dunque sarebbe sbagliato
;				inc esi						;i++
;				jmp residui_for_distanza_euclidea_rispetto_a
;		fine_for_distanza_euclidea_rispetto_a:
;		;ora dobbiamo fare la riduzione
;		haddps xmm1, xmm1
;		haddps xmm1, xmm1			;con queste due hadd abbiamo sommato i 4 elementi presenti in xmm1
;		sqrtps xmm1, xmm1		;calcoliamo la radice quadrata del valore presente in xmm1 e lo memorizziamo in xmm1
;		;ora dobbiamo restituire questo valore a 32 bit
;  		movq [temp_float], xmm1		; Memorizza il float contenuto in xmm1 nella variabile temporanea
;									;perchè per spostare un valore nello stack dei registri x87
;									;serve prima spostare il valore in memoria
;		fld dword [temp_float] 		; fld carica un valore a virgola mobile a precisione singola nello stack FPU
;
;		; ------------------------------------------------------------
;		; Sequenza di uscita dalla funzione
;		; ------------------------------------------------------------
;		pop	edi		; ripristina i registri da preservare
;		pop	esi
;		pop	ebx
;		mov	esp, ebp	; ripristina lo Stack Pointer
;		pop	ebp		; ripristina il Base Pointer
;		ret			; torna alla funzione C chiamante