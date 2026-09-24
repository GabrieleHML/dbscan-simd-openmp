 /**************************************************************************************
* 
* CdL Magistrale in Ingegneria Informatica
* Corso di Architetture e Programmazione dei Sistemi di Elaborazione - a.a. 2020/21
* 
* Progetto dell'algoritmo Attention mechanism 221 231 a
* in linguaggio assembly x86-32 + SSE
* 
* Fabrizio Angiulli, novembre 2022
* 
**************************************************************************************/

/*
* 
* Software necessario per l'esecuzione:
* 
*    NASM (www.nasm.us)
*    GCC (gcc.gnu.org)
* 
* entrambi sono disponibili come pacchetti software 
* installabili mediante il packaging tool del sistema 
* operativo; per esempio, su Ubuntu, mediante i comandi:
* 
*    sudo apt-get install nasm
*    sudo apt-get install gcc
* 
* potrebbe essere necessario installare le seguenti librerie:
* 
*    sudo apt-get install lib32gcc-4.8-dev (o altra versione)
*    sudo apt-get install libc6-dev-i386
* 
* Per generare il file eseguibile:
* 
* nasm -f elf32 att32.nasm && gcc -m32 -msse -O0 -no-pie sseutils32.o att32.o att32c.c -o att32c -lm && ./att32c $pars
* 
* oppure
* 
* ./runatt32
* 
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <libgen.h>
#include <xmmintrin.h>
#include <omp.h>

#define	type	float
#define	MATRIX	type*
#define	VECTOR	type*

#define bool int
#define false 0
#define true 1

/*
//Qui di seguito è riportata la struct dell'input del vecchio progetto, così da avere un confronto con la nuova e non confondersi

typedef struct {
	MATRIX ds; 		// dataset
	VECTOR labels; 	// etichette
	int* out;		// vettore contenente risultato dim=k
	type sc;		// score dell'insieme di features risultato
	int k;			// numero di features da estrarre
	int N;			// numero di righe del dataset
	int d;			// numero di colonne/feature del dataset
	int display;
	int silent;
} params;

*/

typedef struct {
	MATRIX ds; 		// dataset
	VECTOR labels; 	// etichette
	int* out;		// vettore contenente risultato dim=k
	type sc;		// score dell'insieme di features risultato
	int k;			// numero di features da estrarre
	int N;			// numero di righe del dataset
	int d;			// numero di colonne/feature del dataset

	int MinPts;     // numero minimo di punti che devono essere presenti in un intorno per far si che il punto centrato in quell'intorno sia un punto core
	type Eps;		// raggio dell'intorno
	
	int display;
	int silent;
} params;

/*
* 
*	Le funzioni sono state scritte assumento che le matrici siano memorizzate 
* 	mediante un array (float*), in modo da occupare un unico blocco
* 	di memoria, ma a scelta del candidato possono essere 
* 	memorizzate mediante array di array (float**).
* 
* 	In entrambi i casi il candidato dovr� inoltre scegliere se memorizzare le
* 	matrici per righe (row-major order) o per colonne (column major-order).
*
* 	L'assunzione corrente � che le matrici siano in row-major order.
* 
*/

void* get_block(int size, int elements) { 
	return _mm_malloc(elements*size,16); 
}

void free_block(void* p) { 
	_mm_free(p);
}

MATRIX alloc_matrix(int rows, int cols) {
	return (MATRIX) get_block(sizeof(type),rows*cols);
}

int* alloc_int_matrix(int rows, int cols) {
	return (int*) get_block(sizeof(int),rows*cols);
}

void dealloc_matrix(void* mat) {
	free_block(mat);
}

/*
* 
* 	load_data
* 	=========
* 
*	Legge da file una matrice di N righe
* 	e M colonne e la memorizza in un array lineare in row-major order
* 
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero
* 	successivi 4 byte: numero di colonne (M) --> numero intero
* 	successivi N*M*4 byte: matrix data in row-major order --> numeri floating-point a precisione singola
* 
*****************************************************************************
*	Se lo si ritiene opportuno, � possibile cambiare la codifica in memoria
* 	della matrice. 
*****************************************************************************
* 
*/
MATRIX adjust_matrix(MATRIX data, int rows, int cols, int* k); //stub
MATRIX load_data(char* filename, int *n, int *k) {
	FILE* fp;
	int rows, cols, status, i;
	
	fp = fopen(filename, "rb");
	
	if (fp == NULL){
		printf("'%s': bad data file name!\n", filename);
		exit(0);
	}
	
	status = fread(&cols, sizeof(int), 1, fp);
	status = fread(&rows, sizeof(int), 1, fp);
	
	MATRIX data = alloc_matrix(rows,cols);
	status = fread(data, sizeof(type), rows*cols, fp);
	fclose(fp);
	
	*n = rows;
	*k = cols;

	data = adjust_matrix(data,rows,cols,k);
	
	return data;
}

/*
Aggiungiamo questa funzione perchè ci siamo accorti che
quando la matrice ha un numero di colonne >4 ma con cols%4!=0
allora non viene allineata per bene in memoria, e dunque
non possiamo usare il parallelismo SIMD senza causare
SegmentationFault.
Dunque, l'idea è di aggiungere 4-(cols%4) colonne vuote
(conteneti 0.0) ad ogni riga.
*/
MATRIX adjust_matrix(MATRIX data, int rows, int cols, int* k)
{
	/*
	Se il numero di colonne è già multiplo di 4 non va fatto nulla
	*/
	if (cols%4==0)
	{
		return data;
	}//if
	int colonne_da_aggiungere = 4-(cols%4);
	int new_cols = cols+colonne_da_aggiungere;
	MATRIX new_data = alloc_matrix(rows,new_cols);
	for (int i = 0; i<rows; i++)
	{
		for (int j=0; j<new_cols; j++)
		{
			if (j>=cols)
			{
				new_data[i*new_cols+j]=0.0;
			}else{
				new_data[i*new_cols+j]=data[i*cols+j];
			}// if (j>=cols)
		}// for (int j=0; j<new_cols; j++)
	}// for (int i = 0; i<rows; i++)
	*k = new_cols;
	return new_data;
}

/*
* 	save_data
* 	=========
* 
*	Salva su file un array lineare in row-major order
*	come matrice di N righe e M colonne
* 
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero a 32 bit
* 	successivi 4 byte: numero di colonne (M) --> numero intero a 32 bit
* 	successivi N*M*4 byte: matrix data in row-major order --> numeri interi o floating-point a precisione singola
*/
void save_data(char* filename, void* X, int n, int k) {
	FILE* fp;
	int i;
	fp = fopen(filename, "wb");
	if(X != NULL){
		fwrite(&k, 4, 1, fp);
		fwrite(&n, 4, 1, fp);
		for (i = 0; i < n; i++) {
			fwrite(X, sizeof(type), k, fp);
			//printf("%i %i\n", ((int*)X)[0], ((int*)X)[1]);
			X += sizeof(type)*k;
		}
	}
	else{
		int x = 0;
		fwrite(&x, 4, 1, fp);
		fwrite(&x, 4, 1, fp);
	}
	fclose(fp);
}

/*
* 	save_out
* 	=========
* 
*	Salva su file un array lineare composto da k+1 elementi.
* 
* 	Codifica del file:
* 	primi 4 byte: contenenti l'intero 1 		--> numero intero a 32 bit
* 	successivi 4 byte: numero di elementi (k+1) --> numero intero a 32 bit
* 	successivi byte: elementi del vettore 		--> 1 numero floating-point a precisione singola e k interi
*/
void save_out(char* filename, type sc, int* X, int k) {
	FILE* fp;
	int i;
	int n = 1;
	k++;
	fp = fopen(filename, "wb");
	if(X != NULL){
		fwrite(&n, 4, 1, fp);
		fwrite(&k, 4, 1, fp);
		fwrite(&sc, sizeof(type), 1, fp);
		fwrite(X, sizeof(int), k, fp);
		//printf("%i %i\n", ((int*)X)[0], ((int*)X)[1]);
	}
	fclose(fp);
}

// PROCEDURE ASSEMBLY
extern void riempi_out_con_meno_due(params* p);
extern type distanza_euclidea_assembly(params* input, int p1, int p2);
//extern void prova(params* input);
// FINE PROCEDURE ASSEMBLY

/*
	-----------------------------------Costruzione e gestione oggetto Punti----------------------------------
	
	L'oggetto "Punti" rappresenta il risultato di una regionQuery.
*/

	/*
	L'array di interi contiene gli indici delle colonne di "ds", dunque contiene i punti.
	Size è la numerosità dell'array.
	*/
	typedef struct {
		int* indici;  // Puntatore al vettore di interi
		int size;     // Dimensione del vettore
		int capacity; //serve per tenere traccia di quanti spazi di memoria disponiamo
	} Punti;

	/*
	Costruttore

	L'idea è che come parametro "size" venga passato params->N, così di default
	allochiamo una posizione per ogni punto (nel peggiore dei casi al massimo
	"N" punti potra contenere); poi, man mano che aggiungiamo i valori dei punti,
	incrementiamo size.

	Questa funzione ci si aspetta che verrà usata dentro la regionQuery.
	*/
	Punti* initPunti(int initialCapacity)
	{
		// Alloca memoria per la struttura Punti
		Punti* p = (Punti*)malloc(sizeof(Punti));
		if (p == NULL) {
			// Gestire l'errore di allocazione
			fprintf(stderr, "Errore di allocazione della memoria per punti\n");
			return NULL;
		}

		p->indici = (int*)malloc(initialCapacity * sizeof(int));
		if (p->indici == NULL) {
			fprintf(stderr, "Errore: allocazione della memoria fallita.\n");
			free(p); // Libera la memoria della struttura in caso di errore di allocazione
        	return NULL; // Gestione dell'errore di allocazione
		}
		p->capacity = initialCapacity;
		p->size = 0;
		return p;
	}
	/*
	Di default allochiamo solo 5000 posizioni.
	*/
	Punti* initPuntiDefault()
	{
		return initPunti(5000);
	}

	/*
	Distruttore
	*/
	void destroyPunti(Punti* p) {
		if (p != NULL) {
			// Libera la memoria del vettore di interi
			if (p->indici != NULL) {
				free(p->indici);
			}
			// Libera la memoria della struttura
			free(p);
		}
	}

	/*
	Funzione per impostare un valore nel vettore.
	Non serve per inserire un valore in qualsiasi posizione, ma per inserirlo
	nella ultima posizione (dunque va usata solo dentro la funzione "append").
	*/
	/*
	void setPuntiValue(Punti* p, int index, int value) {
		if (index < 0 || index > p->size) {
			fprintf(stderr, "Errore: indice fuori dai limiti.\n");
			exit(EXIT_FAILURE);
		}
		p->indici[index] = value;
		p->size+=1;
	}
	*/
	void setPuntiValue(Punti* p, int index, int value) {
		if (index < 0) {
			fprintf(stderr, "Errore: indice negativo.\n");
			return;
		}

		if (index >= p->capacity) {
			// Raddoppia la capacità fino a coprire l'indice richiesto
			int newCapacity = p->capacity * 2;
			while (index >= newCapacity) {
				newCapacity *= 2;
			}

			int* newIndici = (int*)realloc(p->indici, newCapacity * sizeof(int));
			if (newIndici == NULL) {
				fprintf(stderr, "Errore: riallocazione della memoria per indici fallita.\n");
				return;
			}

			p->indici = newIndici;
			p->capacity = newCapacity;
		}

		p->indici[index] = value;
		p->size += 1;
	}

	/*
	Funzione per aggiungere in coda
	*/
	void append(Punti* p, int value){
		setPuntiValue(p,p->size,value);
	}

	/*
	Funzione per ottenere un valore dal vettore
	*/
	int getPuntiValue(Punti* p, int index) {
		if (index < 0 || index > p->size) {
			fprintf(stderr, "Errore: indice fuori dai limiti.\n");
			exit(EXIT_FAILURE);
		}
		return p->indici[index];
	}

	/*
	Prende l'ultimo elemento della collezione, e diminuisce la dimensione (come se
	liberasse il posto dell'elemento restituito). Ricordiamo che l'elemento in coda sta
	in posizione size-1, perciò è comodo fare prima size-- e poi la return.
	*/
	int pop(Punti* p)
	{
		p->size-=1;
		return p->indici[p->size];
	}

	/*
	Implementiamo la distanza euclidea, ovvero la norma 2.
	*/
	/*
	type distanza_euclidea(params* input, int p1, int p2){
		type somma = 0.0;
		for (int i = 0; i < input->d; i++) {
			//se i punti sono le colonne
			//type differenza = input->ds[p1+(i*input->d)] - input->ds[p2+(i*input->d)];

			//se i punti sono le righe
			type differenza = input->ds[(p1*input->d)+i] - input->ds[(p2*input->d)+i];
			somma += differenza * differenza;
		}
		return sqrt(somma);
	}
	*/

	/*
	Per evitare di calcolare ogni volta le stesse distanze, le calcoliamo una volta
	e le salviamo in una matrice. Questa matrice sarà simmetrica e a diagonale nulla
	(perchè la distanza di un punto da se stesso è nulla). Quando una distanza non è
	ancora stata calcolata la settiamo a -1.0, tanto la distanza che usiamo è quella
	euclidea, dunque non può essere nulla, perchè è prodotto di una radice.
	In generale le distanze sono indotte dalle norme negli spazi vettoriali, dunque
	non possono essere nulle.
	L'algebra serve sempre, ed è pure bella.
	*/
	//void inizializza_matrice_distanze_euclidee(params* input)
	//{
	//	unsigned long rows = input->N;
	//	type* matrice = (type*) malloc(rows*rows*sizeof(type));
	//	for (unsigned long i=0;i<rows;i++)
	//	{
	//		for (unsigned long j=0;j<rows;j++)
	//		{
	//			if(i==j){
	//				matrice[(i*rows)+j] = 0.0f; //la distanza di un punto da se stesso è nulla
	//			}else{
	//				matrice[(i*rows)+j] = -1.0f; //-1.0 vuol dire che la distanza ancora non è stata calcolata;
	//			}//if
	//		}//for j
	//		printf("i=%d\n",i);
	//	}//for i
	//	input->distanze_euclidee = matrice;
	//}//inizializza_matrice_distanze_euclidee

	type distanza_euclidea(params* input, int p1, int p2)
	{
		if (p1 == p2)
		{
			return 0.0f;
		}
		type distanza = distanza_euclidea_assembly(input, p1, p2);

		return distanza;
	}

	/*
	Implementiamo la distanza euclidea, ovvero la norma 2.
	A differenza della funzione distanza_euclidea(), la funzione
	distanza_euclidea_rispetto_a() prevede che uno dei due punti
	venga passato come array di type.

	Precondizioni: il punto p1 deve avere lo stesso numero di dimensioni
				   del punto p2.
	Perchè non la abbiamo anche scritta in assembly? La abbiamo fatta e ci
	siamo resi conto che non si guadagnava praticamente nulla, perchè
	"p1" di solito non è allineato in memoria, e quindi non possiamo
	usare le operazioni SIMD per ottimizzare il tutto.
	*/
	type distanza_euclidea_rispetto_a(params* input, type* p1, int p2){
		type somma = 0.0;
		for (int i = 0; i < input->d; i++) {
			//se i punti sono le colonne
			//type differenza = p1[i] - input->ds[p2+(i*input->d)];

			//se i punti sono le righe
			type differenza = p1[i] - input->ds[(p2*input->d)+i];
			somma += differenza * differenza;
		}
		return sqrt(somma);
	}
	
/*
	---------------------------------Fine Costruzione e gestione oggetto Punti-------------------------------
*/


/*
	-------------------------------------Costruzione e gestione Ball-Tree------------------------------------
*/
	#define LEAFSIZE 300	//per capire cos'è bisogna leggere la funzione calcolo_partizione
	/*
	Un Ball-Tree è una struttura dati utilizzata per la ricerca efficiente di vicini 
	più prossimi in spazi metrici ad alta dimensione. La partizione dei punti in due 
	sotto cerchi è un passo chiave nella costruzione di un Ball-Tree.

	Un nodo di un BallTree rappresenta una iper-sfera, dunque è caratterizzato da un
	centro (un punto di più dimensioni) ed un raggio. Questi nodi costruiscono un
	albero binario. I nodi foglia rappresentano delle iper-sfere che contengono i
	dei punti, e non altre iper-sfere, questo perchè il numero di punti è troppo
	piccolo per determinare una loro partizione in due sotto iper-sfere. 
	*/
	typedef struct BallTreeNode {
		type* centro;                // Centro della ball
		type raggio;               	 // Raggio della ball
		struct BallTreeNode *left;   // Sotto-albero sinistro
		struct BallTreeNode *right;  // Sotto-albero destro
		bool isLeaf;
		Punti* punti;				 // Punti in questa ball (solo foglie)
	} BallTreeNode;

		/*
		Costruttore di BallTreeNode
		*/
		BallTreeNode* initBallTreeNode(type* centro, type raggio, bool isLeaf, Punti* punti) {
			BallTreeNode* node = (BallTreeNode*) malloc(sizeof(BallTreeNode));
			if (node == NULL) {
				return NULL; // Gestione dell'errore di allocazione
			}
			
			// Inizializzazione dei membri della struttura
			node->centro = centro;
			node->raggio = raggio;
			node->left = NULL;
			node->right = NULL;
			node->isLeaf = isLeaf;
			
			
			node->punti = punti;
			

			return node;
		}

		/*
		Dato un'insieme di punti è necessario riuscire a trovare il centroide di tale
		insieme.
		Per fare ciò si sommano le coordinate, lungo tutte le dimensioni, dei punti
		appartenenti a questo insieme, per poi calcolarne la media, lungo ogni
		dimensione.

		La funzione calcolo_centro() fa ciò e restituisce un punto, che noi gestiamo
		come un array di tipo type; questo array contiene le coordinate del punto.
		*/
		type* calcolo_centro(Punti* punti, params* input) {
			int dim = input->d; //quante dimensioni ha ogni punto
			type* center = (type*)malloc(dim*sizeof(type));

			for (int j = 0; j < dim; j++) {	//per ogni dimensione dei punti
				center[j]=0.0;	//invariante della somma
				for (int k = 0; k < punti->size; k++) {	//per ogni punto
					int i=punti->indici[k];	//estrapoliamo l'indice del punto
					center[j]+=input->ds[(i*dim)+j];
				}
			}
			//dopodichè dividiamo per il numero di punti ogni dimensione, così da avere la media
			for (int j = 0; j < dim; j++) {
				center[j] /= punti->size;
			}
			return center;
		}//calcolo_centro

		/*
		Dato un'insieme di punti ed il suo centroide, è necessario trovare il raggio
		di questa iper-sfera.
		Per fare ciò si cerca la distanza massima tra il centro ed un qualsiasi altro
		punto appartenente alla iper-sfera.

		La funzione calcolo_raggio() fa ciò e restituisce un valore in virgola mobile.

		Precondizione: Il parametro "punti" deve contenere almeno un punto.
		*/
		type calcolo_raggio(type* centro, Punti* punti, params* input){
			type max_dist=distanza_euclidea_rispetto_a(input, centro, punti->indici[0]);
			for (int i=1; i<punti->size;i++)	//un semplice calcolo del massimo
			{
				type dist=distanza_euclidea_rispetto_a(input, centro, punti->indici[i]);
				if (dist>max_dist)
				{
					max_dist=dist;
				}//if
			}//for
			return max_dist;
		}//calcolo_raggio

		/*
		Questa funzione calcola il punto con distanza maggiore dal punto "centro".

		Precondizione: "punti" deve contenere almeno un punto.
		*/
		type* calcolo_punto_piu_distante_da(type* centro, Punti* punti, params* input)
		{
			type max_dist=distanza_euclidea_rispetto_a(input, centro, punti->indici[0]);
			int max_punto=punti->indici[0];
			for (int i=1; i<punti->size;i++)	//un semplice calcolo del massimo
			{
				type dist=distanza_euclidea_rispetto_a(input, centro, punti->indici[i]);
				if (dist>max_dist)
				{
					max_dist=dist;
					max_punto=punti->indici[i];
				}//if
			}//for
			type* ret=&(input->ds[(max_punto*input->d)]);	//operatore di dereferenziazione, prendiamo l'indirizzo del punto
			return ret;
		}//calcolo_punto_piu_distante_da

		/*
		Funzione che effettua la partizione di un nodo in due sotto nodi.

		Il partizionamento avviene come descritto di seguito:
		-trovare il punto p1 più distante dal centro del nodo;
		-trovare il punto p2 più distante da p1;
		-creare due nodi, uno centrato in p1, l'altro in p2;
		-per ogni nodo p calcolare la distanza tra esso e p1, e tra esso e p2;
		-p viene messo nel nodo la cui distanza dal centro (p1 o p2) è minima;
		-ricalcolo dei centroidi e dei raggi dei due sotto nodi per motivi di efficienza;

		La partizione di un nodo avviene solo se il nodo contiene un numero 
		di punti maggiore di una certa soglia predefinita. Questo parametro 
		di soglia è spesso chiamato capacity o leaf size e determina il numero 
		massimo di punti che un nodo foglia può contenere prima che sia 
		necessario dividerlo in sotto-nodi.
		*/
		bool calcolo_partizione(BallTreeNode* parent, params* input)
		{
			type* centro = parent->centro;
			Punti* punti = parent->punti;

			if (punti->size<=LEAFSIZE) //<---SegFault (Errore: il puntatore 'punti' è NULL)
			{
				/*
				Se il nodo contiene pochi nodi, allora non ha senso
				partizionarlo. Dunque notifichiamo che la partizione
				non è andata a buon fine. Sarà compito della funzione
				chiamante gestire il nodo "parent" in modo tale che
				il parametro "isLeaf" venga impostato a "true".
				*/
				return false;	
			}
			type* p1 = calcolo_punto_piu_distante_da(centro, punti, input);
			type* p2 = calcolo_punto_piu_distante_da(p1, punti, input);

			/*
			Nella peggiore delle ipotesi tutti i punti vanno in p1 o in p2,
			dunque allochiamo, tramite initPunti() una porzione di memoria
			pari a punti->size*sizeof(int), oltre ad altro.
			*/
			Punti* punti_p1 = initPunti(punti->size);
			Punti* punti_p2 = initPunti(punti->size);
			for (int i=0;i<punti->size;i++)
			{
				int p = punti->indici[i];
				type distanza_da_p1 = distanza_euclidea_rispetto_a(input, p1, p);
				type distanza_da_p2 = distanza_euclidea_rispetto_a(input, p2, p);
				/*
				p1 o p2 stessi verranno valutati ed inseriti rispettivamente
				in punti_p1 e punti_p2. Perchè facciamo ciò visto che p1 e p2
				sono i centroidi dei corrispettivi nodi? Perchè in realtà,dopo
				la partizione si ricalcolano i centroidi e i raggi per ottimizzare
				il tutto.
				*/
				if (distanza_da_p1<distanza_da_p2)	
				{
					append(punti_p1,p);
				}else{
					append(punti_p2,p);
				}//if
			}//for

			type* centroide_p1 = calcolo_centro(punti_p1, input);
			type* centroide_p2 = calcolo_centro(punti_p2, input);

			type raggio_p1 = calcolo_raggio(centroide_p1, punti_p1, input);
			type raggio_p2 = calcolo_raggio(centroide_p2, punti_p2, input);

			parent->left = initBallTreeNode(centroide_p1, raggio_p1, false, punti_p1);
			parent->right = initBallTreeNode(centroide_p2, raggio_p2, false, punti_p2);

			return true; //true rappresenta che la partizione è andata a buon fine
		}//calcolo_partizione

	/*
	Il BallTree non è nient'altro che un puntatore ad un nodo radice.
	*/
	typedef struct {
		BallTreeNode* root;  // Radice dell'albero
	} BallTree;

		/*
		Costruttore
		*/
		BallTree* initBallTree(params* input) {
			BallTree* tree = (BallTree*) malloc(sizeof(BallTree));
			if (tree == NULL) {
				return NULL;
			}

			//Punti* punti = initPunti(input->N);
			Punti* punti = initPuntiDefault();
			/*
			aggiungiamo tutti i punti del dataset 
			dunque gli indici da 0 a (input->N)-1
			*/
			for (int i=0; i<input->N; i++)	
			{
				append(punti, i);
			}//for

			type* centro = calcolo_centro(punti, input);
			type raggio = calcolo_raggio(centro, punti, input);

			tree->root = initBallTreeNode(centro, raggio, false, punti);
			return tree;
		}//initBallTree

		/*
		Questa funzione prende il nodo contenente tutti i punti del dataset
		e da esso vi crea il BallTree vero e proprio. Dunque tramite
		un approccio ricorsivo di partizione.

		La vera e proprio funzione ricorsiva è expandBallTree_recursive,
		expandBallTree è solo una funzione stub per il setup di questa
		elaborazione.
		*/
		void expandBallTree_recursive(BallTreeNode* parent, params* input);
		void expandBallTree(BallTree* ballTree, params* input)
		{
			BallTreeNode* root = ballTree->root;
			expandBallTree_recursive(root, input); //<----SegFault
		}//expandBallTree

		void expandBallTree_recursive(BallTreeNode* parent, params* input)
		{
			bool partizione_eseguita = calcolo_partizione(parent, input); //<---SegFault

			if(partizione_eseguita)
			{
				destroyPunti(parent->punti); //bisogna deallocare questa parte di memoria sennò non ci basta
				parent->punti=NULL;
				expandBallTree_recursive(parent->left, input);
				expandBallTree_recursive(parent->right, input);
			}else{
				parent->isLeaf=true;
			}//if
		}//expandBallTree_recursive

		/*
		Questa funzione fa da stub per la vera e propria query,
		che ha un approccio di ricerca ricorsivo.
		*/
		void ballTreeQuery_recursive(int punto, params* input, BallTreeNode* parent, Punti* punti);
		void ballTreeQuery(int punto, params* input, BallTree* ballTree, Punti* punti)
		{
			BallTreeNode* root = ballTree->root;
			ballTreeQuery_recursive(punto, input, root, punti);
		}//ballTreeQuery

		void ballTreeQuery_recursive(int punto, params* input, BallTreeNode* parent, Punti* punti)
		{
			type* centro = parent->centro;
			type raggio = parent->raggio;
			bool isLeaf = parent->isLeaf;

			type distanza_punto_centro = distanza_euclidea_rispetto_a(input, centro, punto);

			/*
			Se la distanza è maggiore di raggio+Eps, nessun punto nel 
			nodo può essere entro la distanza Eps dal punto di query.
			Quindi, scarta questo nodo.
			*/
			if (distanza_punto_centro>(raggio+(input->Eps))){
				return ;
			}else{
				/*
				Può esserci almeno un punto nel nodo che è entro la distanza Eps.
				*/
				if (isLeaf)
				{
					Punti* spazio_di_ricerca = parent->punti;
					#pragma omp parallel for
					for (int k=0;k<spazio_di_ricerca->size;k++)
					{	
						int i = spazio_di_ricerca->indici[k];	//estraiamo il punto, "k" fa da indice per l'array che contiene i punti
						if(i!=punto)
						{
							if(distanza_euclidea(input,punto,i)<=input->Eps) 
							#pragma omp critical
							{
								append(punti,i);
							}
						}//if i!=punto
					}//for
					return ;
				}else{
					#pragma omp parallel sections
					{
						#pragma omp section
						{
							ballTreeQuery_recursive(punto, input, parent->left, punti);
						}
						#pragma omp section
						{
							ballTreeQuery_recursive(punto, input, parent->right, punti);
						}
					}
				}//if isLeaf
			}//if distanza_punto_centro>(raggio+(input->Eps))
		}//ballTreeQuery_recursive

		/*
		Funzione col solo scopo di debug. Stampa su stdout una rappresentazione
		del BallTree.
		*/
		void stampaBallTree_debuggind_recursive(BallTreeNode* parent, params* input, int id);
		void stampaBallTree_debugging(BallTree* ballTree, params* input)
		{
			stampaBallTree_debuggind_recursive(ballTree->root, input, 0);
		}//stampaBallTree_debuggind

		void stampaBallTree_debuggind_recursive(BallTreeNode* parent, params* input, int id)
		{
			printf("[------Nodo %d------]\n",id);
			printf("isLeaf=%d\n",parent->isLeaf);
			printf("raggio=%f",parent->raggio);
			Punti* punti = parent->punti;

			printf("punti=");
			printf("[");
			for (int i=0;i<punti->size; i++)
			{
				printf("%d,",punti->indici[i]);
			}
			printf("]\n");

			if (parent->isLeaf)
				return ;
			stampaBallTree_debuggind_recursive(parent->left, input, 2*id+1);
			stampaBallTree_debuggind_recursive(parent->right, input, 2*id+2);
		}//stampaBallTree_debuggind_recursive

/*
	-----------------------------------Fine Costruzione e gestione Ball-Tree---------------------------------
*/


/*
	------------------------------------------Implementazione DBScan-------------------------------------------
*/
	/* 
		Funzione per riempire il vettore "out" con elementi pari a -2.

		Per richiamare la funzione si fa: 
			riempi_out_con_meno_due(&p);

		Dunque, nessun cluster avrà l'id "-2" perchè lo usiamo per indicare i punti non ancora classificati
	*/
	#define UNCLASSIFIED -2
	/*
	void riempi_out_con_meno_due(params* p) {
		for (int i = 0; i < p->N; i++) {
			p->out[i] = UNCLASSIFIED;
		}
	}
	*/

	/*
		I punti appartententi al cluster con ID=-1=NOISE sono dei rumori.
		Sklearn utilizza -1 per i punti di rumore, quindi così facendo la 
		numerazione parte da 0. Noi facciamo lo stesso.
	*/
	#define NOISE -1	
	int nextId(int currentId)
	{
		return currentId+1;
	}

	/*
	------------------------------------------Stup declarations---------------------------------------------
	*/
	bool expandCluster(params* input, int punto, int clusterId, BallTree* ballTree);
	void changeClId(params* input, int punto, int id);
	void changeClId_for_seeds(params* input, Punti* seeds ,int id);
	Punti* regionQuery(int punto, params* input, BallTree* ballTree);
	void linearRegionQuery(int punto, params* input, Punti* punti);
	/*
	----------------------------------------end of Stup declarations----------------------------------------
	*/
	
	/*
		Funzione chiave. Per ogni punto, vediamo se è stato classificato. Se è già
		stato classificato si passa vanti, altrimenti si cerca di classificarlo.
		Il parametro "input" contiene sia Eps che MinPts.
	*/
	void DBSCAN(params* input) {
		printf("N:%d , d:%d , k:%d \n",input->N,input->d,input->k);
		BallTree* ballTree = initBallTree(input);
		expandBallTree(ballTree, input); 

		//stampaBallTree_debugging(ballTree, input);
		riempi_out_con_meno_due(input); 

		int clusterId = nextId(NOISE);
		
		for (int j = 0; j < input->N; j++) {
			/*
			"j" rappresenta la "j-esima" riga della matrice "ds", dunque ds[j] è 
			la prima coordinata del punto, ds[j+i] è la "i-esima" coordinata del
			punto.
			*/
			if (input->out[j]==UNCLASSIFIED)
			{
				if (expandCluster(input, j, clusterId, ballTree)==true) //==true equivale a ==1
				{
					clusterId = nextId(clusterId);
				}
			}//if UNCLASSIFIED
		}//for
	}//DBSCAN

	bool expandCluster(params* input, int punto, int clusterId, BallTree* ballTree)
	{
		int MinPts = input->MinPts;
		Punti* seeds=regionQuery(punto, input, ballTree);
		/*
		si presuppone che regionQuery non includa in "seeds" il punto stesso nell'intorno
		del quale effettuiamo al query, per questo c'è size+1, per questo dopo
		faremo "changeClId_for_seeds(input,seeds ,clusterId);" e "changeClId(input, punto, id);"
		insieme, per questo dopo non elimineremo il "punto" da "seeds".
		*/
		if (seeds->size+1<MinPts) 
		{
			changeClId(input, punto, NOISE);
			return false; //return false == return 0
		}//if seeds->size<MinPts
		//else
		changeClId(input, punto, clusterId);
		changeClId_for_seeds(input, seeds, clusterId);
		while(seeds->size>0)
		{
			int currentP = pop(seeds);
			Punti* result = regionQuery(currentP, input, ballTree); 
			/*
			Come prima, si assume che regionQuery non aggiunga "currentP" a "result".
			*/
			if(result->size+1>=MinPts)
			{
				for(int i=0;i<result->size;i++)
				{
					int resultP = getPuntiValue(result, i);
					if(input->out[resultP]==UNCLASSIFIED || input->out[resultP]==NOISE)
					{
						if(input->out[resultP]==UNCLASSIFIED)
						{
							append(seeds, resultP);
						}
						changeClId(input, resultP, clusterId);
					}//if input->out[resultP]==UNCLASSIFIED || input->out[resultP]==NOISE
				}//for
			}//if result->size+1>=MinPts
			/*
			Qui non c'è bisogno di eliminare "currentP" da "seeds" perchè lo 
			abbiamo già eliminato facendo la "pop".
			*/

			destroyPunti(result);

		}//while seeds->size>0

		destroyPunti(seeds);

		return true; //true==1
	}//expandCluster

	/*
	Assegna il punto al cluster di id "id".
	*/
	void changeClId(params* input, int punto, int id)
	{
		input->out[punto]=id;
	}//changeClId

	/*
	Assegna tutti i punti contenuti in "seeds" al cluster di id "id".
	*/
	void changeClId_for_seeds(params* input, Punti* seeds ,int id)
	{
		#pragma omp parallel for
		for(int i=0; i<seeds->size;i++)
		{
			input->out[seeds->indici[i]]=id;
		}
	}//changeClId_for_seeds

	/*
	In base alla logica della regionQuery cambia la forma geometrica dei cluster e
	cambia anche la complessità dell'algoritmo, per questo la seguente funzione
	si limita a richiamare una regionQuery specifica. Dunque definiremo più
	logiche di ricerca.

	Postcondizioni: Nella struct Punti che verrà restituita 
	il "punto" preso in input non dovrà essere incluso, altrimenti
	l'algoritmo "expandCluster" non funzionerà correttamente.
	*/
	Punti* regionQuery(int punto, params* input, BallTree* ballTree){
		Punti* punti = initPuntiDefault();

		ballTreeQuery(punto, input, ballTree, punti);

		//linearRegionQuery(punto, input, punti); 

		return punti;
	}
	
	/*
	Algoritmo di ricerca a forza bruta. Effettua una ricerca esaustiva.
	Offre le prestazioni peggiori ma la semplicità maggiore.
	*/
	void linearRegionQuery(int punto, params* input, Punti* punti)
	{
		for (int j=0;j<input->N;j++)
		{	
			if(j!=punto)
			{
				if(distanza_euclidea(input,punto,j)<=input->Eps) 
				{
					append(punti,j);
				}
			}//if j!=punto
		}//for
	}//linearRegionQuery


	
/*
	----------------------------------------Fine Implementazione DBScan----------------------------------------
*/

int main(int argc, char** argv) {

	char fname[256];
	char* dsfilename = NULL;
	char* labelsfilename = NULL;
	clock_t t;
	float time;
	
	//
	// Imposta i valori di default dei parametri
	//

	params* input = malloc(sizeof(params));

	input->ds = NULL;
	input->labels = NULL;
	input->k = -1;
	input->sc = -1;

	input->MinPts = -1;
	input->Eps = -1;

	input->silent = 0;
	input->display = 0;

	//
	// Visualizza la sintassi del passaggio dei parametri da riga comandi
	//

	if(argc <= 1){
		printf("%s -ds <DS> -labels <LABELS> -k <K> -MinPts <MinPts> -Eps <Eps> [-s] [-d]\n", argv[0]);
		printf("\nParameters:\n");
		printf("\tDS: il nome del file ds2 contenente il dataset\n");
		printf("\tLABELS: il nome del file ds2 contenente le etichette\n");
		printf("\tk: numero di features da estrarre\n");
		printf("\nOptions:\n");
		printf("\t-s: modo silenzioso, nessuna stampa, default 0 - false\n");
		printf("\t-d: stampa a video i risultati, default 0 - false\n");
		exit(0);
	}

	//
	// Legge i valori dei parametri da riga comandi
	//

	int par = 1;
	while (par < argc) {
		if (strcmp(argv[par],"-s") == 0) {
			input->silent = 1;
			par++;
		} else if (strcmp(argv[par],"-d") == 0) {
			input->display = 1;
			par++;
		} else if (strcmp(argv[par],"-ds") == 0) {
			par++;
			if (par >= argc) {
				printf("Missing dataset file name!\n");
				exit(1);
			}
			dsfilename = argv[par];
			par++;
		} else if (strcmp(argv[par],"-labels") == 0) {
			par++;
			if (par >= argc) {
				printf("Missing labels file name!\n");
				exit(1);
			}
			labelsfilename = argv[par];
			par++;
		} else if (strcmp(argv[par],"-k") == 0) {
			par++;
			if (par >= argc) {
				printf("Missing k value!\n");
				exit(1);
			}
			input->k = atoi(argv[par]);
			par++;
		} else if (strcmp(argv[par],"-MinPts")==0) {
			par++;
			if (par >= argc){
				printf("Missing MinPts value!\n");
				exit(1);
			}
			input->MinPts = atoi(argv[par]);
			par++;
		} else if (strcmp(argv[par],"-Eps")==0) {
			par++;
			if (par >= argc){
				printf("Missing Eps value!\n");
				exit(1);
			}
			input->Eps = atof(argv[par]); //ipotizziamo che Eps sia intero
			par++;
		} else{
			printf("WARNING: unrecognized parameter '%s'!\n",argv[par]);
			par++;
		}
	}

	//
	// Legge i dati e verifica la correttezza dei parametri
	//

	if(dsfilename == NULL || strlen(dsfilename) == 0){
		printf("Missing ds file name!\n");
		exit(1);
	}

	if(labelsfilename == NULL || strlen(labelsfilename) == 0){
		printf("Missing labels file name!\n");
		exit(1);
	}


	input->ds = load_data(dsfilename, &input->N, &input->d);

	/*
	-------------------------Codice non presente nel template del Feature Selection--------------------------
	*/
	input->k=input->N; //<--------Così facendo annulliamo il parametro k
	/*
	-----------------Fine sezione di codice non presente nel template del Feature Selection------------------
	*/

	int nl, dl;
	/*
	input->labels = load_data(labelsfilename, &nl, &dl);
	
	if(nl != input->N || dl != 1){
		printf("Invalid size of labels file, should be %ix1!\n", input->N);
		exit(1);
	} 
	*/

	if(input->k <= 0){
		printf("Invalid value of k parameter!\n");
		exit(1);
	}

	input->out = alloc_int_matrix(input->k, 1); 

	//
	// Visualizza il valore dei parametri
	//

	if(!input->silent){
		printf("Dataset file name: '%s'\n", dsfilename);
		printf("Labels file name: '%s'\n", labelsfilename);
		printf("Dataset row number: %d\n", input->N);
		printf("Dataset column number: %d\n", input->d);
		printf("Number of features to extract: %d\n", input->k);
	}

	// COMMENTARE QUESTA RIGA!
	//prova(input);
	//

	//
	// Correlation Features Selection
	//
	t = clock();
	DBSCAN(input);
	t = clock() - t;
	time = ((float)t)/CLOCKS_PER_SEC;

	if(!input->silent)
		printf("CFS time = %.3f secs\n", time);
	else
		printf("%.3f\n", time);

	//
	// Salva il risultato
	//
	sprintf(fname, "out32_%d_%d_%d.ds2", input->N, input->d, input->k);
	save_out(fname, input->sc, input->out, input->k);
	if(input->display){
		if(input->out == NULL)
			printf("out: NULL\n");
		else{
			int i,j;
			printf("sc: %f, out: [", input->sc);
			for(i=0; i<input->k; i++){
				printf("%i,", input->out[i]);
			}
			printf("]\n");
		}
	}

	if(!input->silent)
		printf("\nDone.\n");

	dealloc_matrix(input->ds);
	dealloc_matrix(input->labels);
	dealloc_matrix(input->out);
	free(input);

	return 0;
}
