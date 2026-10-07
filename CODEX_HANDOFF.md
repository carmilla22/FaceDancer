# Handoff Codex — prova GPU FaceDancer con self-ai

Documento aggiornato il 2026-10-07. Lingua di lavoro: italiano.

## 1. Obiettivo e autorizzazione corrente

Eseguire una prova di addestramento di FaceDancer su una GPU economica di Vast.ai, usando self-ai per creare l'istanza e accedervi. Budget massimo desiderato: **10 USD complessivi**. La prova prevista è di **tre iterazioni**, una epoca, immagini 256×256, batch 2 (batch 1 se necessario), validation e salvataggio dei checkpoint.

L'utente ha confermato di voler mantenere **TensorFlow/Keras**, senza migrare a PyTorch. Generatore e discriminatore partono da zero omettendo `--load`; ArcFace, VGG16 ed ExpressionEmbedder restano componenti preaddestrati della pipeline.

La richiesta che ha prodotto questo documento autorizza la scrittura dell'handoff, non l'avvio immediato di risorse a pagamento. Non noleggiare né distruggere istanze mentre si sta soltanto leggendo/verificando questo documento. Riprendere la creazione quando l'utente chiede di procedere.

## 2. Percorsi e struttura

### self-ai

Root e directory di lavoro:

```text
/mnt/c/Users/c.galasso/Desktop/self-ai
```

File rilevanti:

- `create-instance`: cerca offerte, propone `Launch? [yn]`, crea l'istanza, stampa il contratto e attende SSH.
- `common`: funzioni Bash, variabile `SSH_PREFIX`, caricamento opzionale del file `configuration`.
- `flake.nix`, `nix/base-image.nix`, `nix/container-image.nix`: infrastruttura Nix e container.
- `nix/services/`: servizi esistenti; non contiene un servizio FaceDancer dedicato.
- `pyproject.toml`, `uv.lock`: ambiente locale di gestione, include la CLI Vast.ai.
- `launch-llm`, `launch-whisper`: esecuzione di modelli pronti; non sono launcher di training FaceDancer.
- `workspace/`: progetto di esempio sui numeri primi, non il progetto FaceDancer.
- `CODEX_HANDOFF.md`: questo documento.

### FaceDancer originale

```text
/mnt/c/Users/c.galasso/Desktop/FaceDancer
```

Il percorso inizialmente fornito `/mnt/c/c.galasso/Desktop/FaceDancer` non esiste. Il percorso corretto include `/Users/`.

File rilevanti:

- `train/train.py`: training TensorFlow/Keras e parser degli argomenti.
- `train/checkpointing.py`, `utils/utils.py`: gestione checkpoint, stato e salvataggi atomici.
- `networks/generator.py`, `networks/discriminator.py`, `networks/layers.py`: architetture.
- `utils/loss.py`: loss, inclusa VGG16 con pesi ImageNet.
- `utils/hand_occlusion.py`, `models/hand_landmarker.task`: rilevamento maschere MediaPipe.
- `dataset/tf_records_parser.py`: caricamento TFRecord source e target.
- `devenv.nix`, `devenv.yaml`, `devenv.lock`, `pyproject.toml`, `uv.lock`: ambiente del modello.
- `scripts/check_environment.py`: controllo GPU specifico di un precedente host, da non usare invariato su Vast.ai.

## 3. Ambiente locale verificato

- Windows con WSL2; utente Linux `cgalasso`.
- Prompt dell'utente: `(self-ai) cgalasso@BVG8ZS674`.
- Distribuzione osservata: Ubuntu 26.04.1 LTS.
- Kernel osservato: `6.18.33.2-microsoft-standard-WSL2`.
- Python globale osservato: 3.14.4.
- Python di `self-ai/.venv`: **3.11.17**.
- CLI presente: `/mnt/c/Users/c.galasso/Desktop/self-ai/.venv/bin/vastai`.
- Nix presente: `/home/cgalasso/.nix-profile/bin/nix`.
- devenv presente: `/home/cgalasso/.nix-profile/bin/devenv`; versione locale osservata 2.3.1+2418e1b.
- `rsync`, `ssh`, `timeout` presenti nel sistema.
- `jq` è stato usato con successo nel terminale dell'utente, ma non era nel PATH della shell non interattiva di Codex al momento del controllo. **DA VERIFICARE** il PATH della nuova shell.
- Le variabili della shell dell'utente non sono automaticamente visibili alle shell degli strumenti Codex. Reimpostarle esplicitamente.

Il training remoto deve usare Python 3.11, TensorFlow 2.15.1 e Keras legacy definiti da FaceDancer, non il Python globale locale.

## 4. Stato confermato e comandi riusciti

L'utente ha dichiarato di avere completato i punti 1–5 della procedura: strumenti locali, preparazione della copia ridotta e ricerca delle offerte. È stato verificato direttamente quanto segue:

1. La CLI locale e l'ambiente `.venv` esistono.
2. La copia ridotta esiste nel percorso indicato nella sezione successiva.
3. La copia contiene codice di training, i tre asset necessari e un TFRecord per ciascuna delle quattro partizioni.
4. Nella copia, `devenv.yaml` non contiene gli input `celeb`, `hands`, `hands_masks`, e `devenv.nix` non contiene dichiarazioni `files."..."`.
5. L'utente ha eseguito con successo:

```bash
vastai search offers "$QUERY" --storage 60 --raw > /tmp/facedancer-offer.json

jq '.[] | {
  id,
  gpu_name,
  dph_total,
  storage_cost,
  inet_down_cost,
  inet_up_cost
}' /tmp/facedancer-offer.json

printf 'QUERY=%s\n' "${QUERY-}"
```

6. Il controllo seguente è stato eseguito e ha restituito **false**, correttamente:

```bash
jq -e '
  length == 1
  and .[0].id == 53644732
  and (.[0].dph_total | type) == "number"
  and .[0].dph_total <= 0.20
' /tmp/facedancer-offer.json
```

Non è un successo di selezione: ha evitato di proseguire con un'offerta diversa da quella prevista.

## 5. Copia di lavoro ridotta già preparata

Percorso verificato:

```text
/tmp/facedancer-gpu-test.7xkTQs
```

Variabili da ripristinare:

```bash
FD_LOCAL="/mnt/c/Users/c.galasso/Desktop/FaceDancer"
FD_STAGE="/tmp/facedancer-gpu-test.7xkTQs"
export FD_STAGE
```

La copia esclude ambienti locali, Git, dataset completi, checkpoint/log/config di esperimenti precedenti e risultati. Contiene:

- Codice e configurazioni dell'ambiente Python.
- `arcface_model/ArcFace-Res50.h5` (circa 155 MiB).
- `expressionembedder_model/ExpressionEmbedder-B0.h5` (circa 274 MiB).
- `models/hand_landmarker.task` (circa 7,5 MiB).
- `assets/dataset/source_processed/train/*.records` (un file).
- `assets/dataset/source_processed/validation/*.records` (un file).
- `assets/dataset/target_processed/train/*.records` (un file).
- `assets/dataset/target_processed/validation/*.records` (un file).

I TFRecord originali occupano complessivamente circa 14 MiB. **DA VERIFICARE** il numero di esempi; non è stato contato nella sessione.

Nella copia sono state rimosse le dichiarazioni di asset in `devenv.nix` e gli input asset in `devenv.yaml`, per evitare download inutili sul server. Le configurazioni originali di FaceDancer non sono state modificate da Codex. La copia non è ancora stata validata con una completa attivazione `devenv shell`: **DA VERIFICARE**.

`/tmp` può essere ripulita o persa dopo un riavvio. Prima di noleggiare verificare che questa directory esista ancora. Se manca, ricrearla dall'originale prima di spendere.

## 6. SSH e Vast.ai: verifiche senza segreti

Non sono riportati né devono essere riportati contenuti di chiavi, API key, token, password o credenziali.

### Stato osservato

- `self-ai/configuration`: **non presente** al controllo finale.
- `/home/cgalasso/.ssh/id_ed25519`: **non presente** al controllo finale.
- `/home/cgalasso/.ssh/id_ed25519.pub`: **non presente** al controllo finale.
- File di configurazione API della CLI presente in `/home/cgalasso/.config/vastai/vast_api_key`; il suo contenuto non è stato letto né copiato.
- La ricerca delle offerte è riuscita nel terminale dell'utente.
- Percorso effettivo di una chiave alternativa, registrazione della chiave su Vast.ai e stato dell'agent SSH: **DA VERIFICARE**.
- Saldo, eventuali altre istanze e impostazioni di fatturazione automatica: **DA VERIFICARE**.

Il percorso `~/.ssh/id_ed25519` nei messaggi precedenti era un esempio, non una configurazione confermata. Non assumerlo valido.

### Prima di noleggiare

Individuare la chiave scelta dall'utente senza visualizzare contenuti privati. Impostare nel file `configuration`:

```bash
SSH_KEY=/percorso/reale/della/chiave_privata
```

Il file è caricato come Bash da `common`. Non inserirvi segreti aggiuntivi per questo test. Non committare file contenenti credenziali.

`common` usa `SSH_PREFIX=(ssh -o "StrictHostKeyChecking no")` e non passa `-i "$SSH_KEY"`. Per usare lo script invariato, caricare la chiave nell'agent:

```bash
eval "$(ssh-agent -s)"
ssh-add /percorso/reale/della/chiave_privata
ssh-add -l
```

Se la chiave pubblica non è registrata, il comando suggerito è:

```bash
vastai create ssh-key /percorso/reale/della/chiave_privata.pub
```

Eseguirlo soltanto dopo aver verificato il percorso e la necessità. Il container self-ai riceve anche la chiave pubblica tramite `SELF_AI_SSH_PUBLIC_KEY` da `create-instance`.

## 7. Stato della ricerca GPU

La variabile confermata dall'utente era ancora una ricerca generale:

```bash
QUERY='num_gpus=1 gpu_ram>=24 compute_cap>=800 compute_cap<=890 cuda_max_good>=12.2 verified=true rentable=true direct_port_count>=1 inet_down>=300'
```

Non includeva ID specifico, limite di prezzo o limiti di traffico. Questo spiega le molte offerte.

Il file `/tmp/facedancer-offer.json` contiene **64 offerte**, non una. Non usarlo come conferma che un'offerta sia stata selezionata o noleggiata.

### Offerte discusse

| ID offerta | GPU | Tariffa osservata | Stato |
|---|---|---|---|
| 53644732 | RTX 3090 | 0,1394 USD/h in una ricerca precedente | Non presente nell'ultimo elenco; disponibilità DA VERIFICARE |
| **46021771** | **RTX 3090** | **0,1877778 USD/h con `--storage 60`** | Candidata raccomandata; non ancora selezionata con successo |
| 49607479 | RTX 3090 | 0,1833333 USD/h con `--storage 60` | Alternativa leggermente più economica all'ora, traffico più caro |

La 46021771 aveva `inet_down_cost` e `inet_up_cost` pari a circa 0,0000130208 USD/GB. Si è preferita come candidata per il basso costo di trasferimento durante il setup. Tutti i valori sono una fotografia della ricerca, non prezzi o disponibilità garantiti per la nuova sessione.

L'ultimo messaggio dell'assistente ha proposto la ricerca specifica della 46021771 riportata sotto. **L'utente non ha ancora confermato di averla eseguita né di avere ottenuto `true`.**

## 8. Decisioni e motivazioni

- Mantenere TensorFlow/Keras: scelta esplicita dell'utente.
- Riutilizzare self-ai manualmente: per la prima prova non serve creare un launcher dedicato o modificare le architetture.
- Una GPU: il training seleziona un dispositivo e non implementa training distribuito.
- 24 GB: scelta prudente, **non requisito minimo misurato**.
- 12–16 GB con batch 1: alternativa possibile da provare, non garantita; per la candidata attuale si mantiene batch 2.
- Ampere/Ada (`compute_cap` 800–890): filtro prudente per l'ambiente CUDA esistente.
- Host verificato, disponibile e SSH diretto: facilità di accesso.
- 60 GB: allocazione proposta per la copia ridotta e le dipendenze; sufficienza DA VERIFICARE.
- Tre iterazioni: verifica funzionale, non valutazione della qualità del modello.
- Omettere `--load`: nuova inizializzazione di generatore e discriminatore.
- Usare `vast_smoke_001` come nome nuovo; se esiste già sul server scegliere un altro nome.
- Trasferire una copia ridotta e materializzare i symlink: evitare download completi e link rotti verso il Nix store locale.

## 9. Problemi e limiti noti

1. **Percorso FaceDancer errato:** risolto aggiungendo `/Users/`.
2. **Ricerca generica invece dell'ID preciso:** il controllo `false` era corretto; reimpostare `QUERY` e aggiornare anche l'ID nel controllo jq.
3. **SSH/configurazione non confermati:** verificare prima della creazione, come descritto sopra.
4. **`create-instance` aspetta indefinitamente:** manca un timeout interno e non gestisce bene elenco vuoto/offerte esaurite. Non lanciarlo su una ricerca vuota. Un timeout esterno ferma il client, non distrugge l'istanza.
5. **Prezzo nel riepilogo di `create-instance`:** lo script ricerca senza `--storage "$DISK_SIZE"` e arrotonda il costo a due decimali. La verifica preliminare deve usare `--storage 60`; il riepilogo non è una stima esatta con 60 GB.
6. **`fd-check-env` non portabile:** richiede RTX 4500 Ada, compute capability 8.9 e `/run/opengl-driver/lib/libcuda.so.1`. Usare il controllo generico più sotto.
7. **Driver nel container:** aggiungere i percorsi librerie Ubuntu a `LD_LIBRARY_PATH`; compatibilità reale DA VERIFICARE.
8. **README FaceDancer:** alcuni comandi usano `assets/...` invece di `assets/dataset/...`; i comandi di questo documento usano i percorsi osservati.
9. **Checkpoint:** salvataggi atomici, architetture JSON, pesi H5 e stato contatori. Gli ottimizzatori Adam vengono ricreati alla ripresa; non è una ripresa completa dello stato del training.
10. **H5 dei checkpoint:** `gen_3.h5` contiene pesi separati dall'architettura; non trattarlo automaticamente come modello completo caricabile da ogni script di inferenza.
11. **Vecchio file `error`:** nell'originale contiene un errore storico HDF5 per nomi duplicati; il codice attuale include nomi espliciti nei layer e controlli. Non dichiarare risolto su GPU finché la prova non salva correttamente.

## 10. Budget: protezioni e loro limiti

Budget richiesto: **10 USD massimo**. Strategia proposta: prezzo entro 0,20 USD/h con 60 GB, traffico entro 0,01 USD/GB per direzione, prova breve, controllo manuale Billing e timer di distruzione dopo due ore.

- A 0,1877778 USD/h, due ore sono circa 0,3756 USD alla tariffa osservata, più traffico.
- Il filtro di prezzo non è un limite di spesa cumulativa.
- `create-instance` non spegne/distrugge automaticamente la macchina.
- Nella CLI installata non è stato trovato un flag di creazione per imporre un tetto totale di 10 USD.
- **Non promettere un tetto rigido garantito** con i comandi attuali.
- Il timer locale richiede Windows/WSL acceso, senza sospensione, con rete disponibile; una chiamata API fallita può impedirne l'effetto.
- `nohup` protegge dalla chiusura del terminale, non dallo spegnimento di WSL/Windows.
- `stop` mantiene i costi del disco; `destroy` elimina la macchina e tutti i dati remoti.
- Non affidarsi all'esaurimento del credito: storage e gestione dei saldi negativi possono continuare a generare addebiti.
- Controllare eventuale autoricarica, altri workload e saldo: DA VERIFICARE. Non modificarli automaticamente.
- Soglia operativa prudente suggerita: intervenire a 1 USD di spesa della prova; è un controllo manuale, non automatizzato.
- Impostare anche un promemoria esterno a 90 minuti; recuperare i risultati prima delle due ore.

Riferimenti consultati:

- https://docs.vast.ai/guides/reference/billing
- https://docs.vast.ai/cli/hello-world
- https://www.tensorflow.org/guide/checkpoint

## 11. Ripresa: verifiche locali e selezione dell'offerta

Tutti i comandi di questa sezione sono **locali WSL** e non creano istanze.

```bash
cd /mnt/c/Users/c.galasso/Desktop/self-ai
source .venv/bin/activate

FD_LOCAL="/mnt/c/Users/c.galasso/Desktop/FaceDancer"
FD_STAGE="/tmp/facedancer-gpu-test.7xkTQs"
export FD_STAGE

command -v vastai
command -v jq
command -v ssh
command -v rsync
command -v timeout

test -f "$FD_STAGE/train/train.py"
test -f "$FD_STAGE/models/hand_landmarker.task"
test -f "$FD_STAGE/arcface_model/ArcFace-Res50.h5"
test -f "$FD_STAGE/expressionembedder_model/ExpressionEmbedder-B0.h5"
du -sh "$FD_STAGE"
```

Criterio: strumenti presenti, tutti i `test` restituiscono successo e la copia esiste. Controllare anche `configuration` e agent SSH senza leggere segreti.

Se mancano gli strumenti Nix, aprire la shell suggerita in precedenza, poi riattivare `.venv`:

```bash
nix shell \
  nixpkgs#uv \
  nixpkgs#python311 \
  nixpkgs#jq \
  nixpkgs#rsync \
  nixpkgs#openssh \
  --command bash

source .venv/bin/activate
```

Non rieseguire `uv sync` se l'ambiente già funzionante non ne ha bisogno. Il comando di bootstrap proposto era `uv sync --frozen --python 3.11`.

Selezione precisa:

```bash
QUERY='id=46021771 num_gpus=1 gpu_name=RTX_3090 gpu_ram>=24 compute_cap>=800 compute_cap<=890 cuda_max_good>=12.2 verified=true rentable=true direct_port_count>=1 inet_down>=300 disk_space>=60 dph<=0.20 inet_down_cost<=0.01 inet_up_cost<=0.01'

printf 'QUERY=%s\n' "$QUERY"

vastai search offers "$QUERY" --storage 60 --raw \
  > /tmp/facedancer-offer.json

jq '.[] | {
  id,
  gpu_name,
  num_gpus,
  gpu_ram,
  cuda_max_good,
  dph_total,
  inet_down_cost,
  inet_up_cost
}' /tmp/facedancer-offer.json

jq -e '
  length == 1
  and .[0].id == 46021771
  and (.[0].dph_total | type) == "number"
  and .[0].dph_total <= 0.20
' /tmp/facedancer-offer.json
```

Criterio: un solo risultato, ID corretto, controllo `true` con exit code 0. Se `false`, `[]` o errore, fermarsi prima del noleggio. Se serve una nuova offerta, mantenere i criteri di costo/compatibilità e aggiornare sia la query sia il controllo dell'ID; non rimuovere i limiti silenziosamente.

## 12. Creazione e timer — non ancora eseguiti

Solo dopo selezione riuscita, chiave/configurazione verificate e istruzione dell'utente a procedere.

Preparare un secondo terminale WSL. Disabilitare temporaneamente la sospensione automatica del computer. Nel primo terminale locale:

```bash
timeout --foreground 15m \
  ./create-instance \
  'ghcr.io/aleclearmind/vast-ai-nix-container-base' \
  "$QUERY" \
  60
```

Lo script chiede `Launch? [yn]`. Verificare offerta e prezzo prima di `y`. L'immagine è referenziata dagli script esistenti, ma la sua disponibilità e il provisioning non sono stati verificati su una macchina reale in questa sessione.

Appena appare `Successfully created contract NUMERO`, nel **secondo terminale locale**:

```bash
cd /mnt/c/Users/c.galasso/Desktop/self-ai

INSTANCE_ID=NUMERO
VAST_CLI="$PWD/.venv/bin/vastai"

mkdir -p runs

nohup bash -c '
  sleep 7200
  "$1" destroy instance "$2"
' _ "$VAST_CLI" "$INSTANCE_ID" \
  > "runs/auto-destroy-$INSTANCE_ID.log" 2>&1 < /dev/null &

GUARD_PID=$!
echo "Timer avviato: PID $GUARD_PID"
ps -p "$GUARD_PID" -o pid,etime,args

"$VAST_CLI" show instance "$INSTANCE_ID"
```

`NUMERO` è il contratto/istanza appena creato, **non l'ID offerta**. Il timer cancella anche i risultati remoti alla scadenza; il backup va completato prima.

Criteri: ID annotato, processo timer presente, istanza visibile e stato controllato. Il timer non è una garanzia di esecuzione remota e il suo exit code/API response va verificato se raggiunge la scadenza.

Quando l'istanza è pronta:

```bash
SSH_URL=$("$VAST_CLI" ssh-url "$INSTANCE_ID")
echo "$SSH_URL"
ssh "$SSH_URL"
```

Criterio: collegamento SSH riuscito. Se il timeout del primo comando scade, verificare l'istanza già creata; non crearne automaticamente un'altra.

## 13. Preparazione e trasferimento — non ancora eseguiti

### Server remoto

```bash
export PATH="$HOME/.nix-profile/bin:$PATH"

nix profile install nixpkgs#rsync
nix profile install --accept-flake-config github:cachix/devenv/latest

mkdir -p /root/FaceDancer
devenv --version

exit
```

Questa installazione segue il README esistente; `latest` non è una versione fissata. Compatibilità con lock e container: **DA VERIFICARE**. Non aggiornare le dipendenze del modello per risolvere problemi senza prima analizzarli.

### Locale WSL

Ricavare host e porta dall'URL reale restituito da `ssh-url`:

```bash
FD_HOST="HOST_REALE"
FD_PORT="PORTA_REALE"

rsync -avL \
  -e "ssh -p $FD_PORT" \
  "$FD_STAGE/" \
  "root@$FD_HOST:/root/FaceDancer/"

ssh "$SSH_URL"
```

Criteri: `rsync` exit code 0; file/modelli/TFRecord presenti sul server. Non usare host o porta degli esempi della conversazione.

## 14. Ambiente remoto e prova GPU — non ancora eseguiti

### Server remoto

```bash
cd /root/FaceDancer

export PATH="$HOME/.nix-profile/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

devenv shell
```

Dentro la shell devenv:

```bash
python - <<'PY'
import os
import tensorflow as tf
import tf_keras

assert os.environ.get("TF_USE_LEGACY_KERAS") == "1"
assert tf.keras.Model is tf_keras.Model

gpus = tf.config.list_physical_devices("GPU")
assert gpus, "TensorFlow non vede la GPU"

with tf.device("/GPU:0"):
    value = tf.matmul(tf.ones((64, 64)), tf.ones((64, 64)))

value.numpy()
assert "GPU:0" in value.device, value.device

from utils.hand_occlusion import create_hand_landmarker

landmarker = create_hand_landmarker("models/hand_landmarker.task")
landmarker.close()

print("TensorFlow:", tf.__version__)
print("GPU:", gpus)
print("Calcolo GPU e MediaPipe: OK")
PY
```

Criteri: TensorFlow 2.15.1 atteso, Keras legacy, GPU rilevata, matmul effettivamente su GPU, modello MediaPipe caricabile. Se fallisce, non avviare il training e controllare tempo/spesa residui.

Opzionale prima del training: contare gli esempi e verificare che le quattro partizioni non siano vuote:

```bash
python - <<'PY'
from pathlib import Path
import tensorflow as tf

for role in ("source", "target"):
    for split in ("train", "validation"):
        folder = Path(f"assets/dataset/{role}_processed/{split}")
        files = sorted(str(p) for p in folder.glob("*.records"))
        assert files, f"Nessun TFRecord in {folder}"
        count = sum(1 for _ in tf.data.TFRecordDataset(files))
        print(folder, "esempi:", count)
        assert count >= 2, f"Dataset troppo piccolo: {folder}"
PY
```

## 15. Training breve e verifica — non ancora eseguiti

Sul server, in `/root/FaceDancer`, dentro devenv:

```bash
python -u train/train.py \
  --data_dir 'assets/dataset/target_processed/train/target_occluded_train_*-of-*.records' \
  --eval_dir 'assets/dataset/target_processed/validation/target_occluded_validation_*-of-*.records' \
  --source_data_dir 'assets/dataset/source_processed/train/source_occluded_train_*-of-*.records' \
  --eval_source_dir 'assets/dataset/source_processed/validation/source_occluded_validation_*-of-*.records' \
  --hand_task_path models/hand_landmarker.task \
  --arcface_path arcface_model/ArcFace-Res50.h5 \
  --eval_model_expface expressionembedder_model/ExpressionEmbedder-B0.h5 \
  --batch_size 2 \
  --eval_batch_size 1 \
  --num_epochs 1 \
  --iterations_per_epoch 3 \
  --checkpoint_interval 3 \
  --log_dir logs/runs \
  --chkp_dir checkpoints \
  --log_name vast_smoke_001
```

Non aggiungere `--load`. Per OOM, valutare batch 1 e un nuovo nome di esperimento, senza modificare architettura/risoluzione automaticamente. La prima iterazione può includere compilazione dei grafi e download dei pesi VGG16.

Mantenere la connessione aperta per la prova breve. Se serve un processo persistente, valutarlo esplicitamente, senza prolungare il noleggio inavvertitamente.

Criteri:

- Tre iterazioni completate senza eccezioni; loss finite da verificare nell'output.
- Validation e immagini di esempio non producono errori.
- Presenti architetture JSON, pesi G/D e stato:

```bash
ls -lh \
  checkpoints/vast_smoke_001/gen/ \
  checkpoints/vast_smoke_001/dis/ \
  checkpoints/vast_smoke_001/state/

cat checkpoints/vast_smoke_001/state/3.json
```

File attesi:

```text
gen/gen.json
gen/gen_3.h5
dis/dis.json
dis/dis_3.h5
state/3.json
```

Stato atteso:

```json
{
  "version": 1,
  "iteration": 3,
  "epoch": 1,
  "epoch_iteration": 0,
  "iterations_per_epoch": 3
}
```

Questo verifica l'esecuzione, non la qualità del modello né il ripristino degli ottimizzatori.

## 16. Recupero risultati e fine noleggio — non ancora eseguiti

Uscire prima da devenv e poi da SSH (`exit` per ciascuna shell). Nel terminale locale ripristinare le variabili reali, se necessario.

```bash
mkdir -p runs/vast_smoke_001

rsync -av \
  -e "ssh -p $FD_PORT" \
  "root@$FD_HOST:/root/FaceDancer/checkpoints/vast_smoke_001" \
  runs/vast_smoke_001/

rsync -av \
  -e "ssh -p $FD_PORT" \
  "root@$FD_HOST:/root/FaceDancer/config/vast_smoke_001" \
  runs/vast_smoke_001/

rsync -av \
  -e "ssh -p $FD_PORT" \
  "root@$FD_HOST:/root/FaceDancer/logs/" \
  runs/vast_smoke_001/logs/

ls -lh runs/vast_smoke_001/vast_smoke_001/gen/
ls -lh runs/vast_smoke_001/vast_smoke_001/dis/
cat runs/vast_smoke_001/vast_smoke_001/state/3.json
```

Nota: i primi due rsync uniscono checkpoint e configurazione nella stessa sottocartella locale `vast_smoke_001`. La duplicazione del nome nel percorso è prevista dai comandi sopra.

Criteri: rsync exit code 0, entrambi i pesi e le due architetture presenti localmente, stato coerente e log/config recuperati. Per una verifica più forte confrontare checksum dei file remoti e locali.

Solo dopo il recupero/verifica, eliminare l'istanza scelta:

```bash
"$VAST_CLI" destroy instance "$INSTANCE_ID"
```

Verificare nella console che l'istanza sia realmente eliminata. Solo dopo annullare il timer ancora pendente, nello stesso terminale in cui `GUARD_PID` è definito:

```bash
kill "$GUARD_PID"
```

Non usare un PID vecchio in una nuova sessione: verificare il processo prima. Se il timer è già terminato, leggere il log e controllare lo stato nella console.

## 17. Cosa è fatto e cosa manca

### Fatto / osservato

- Analisi di self-ai e FaceDancer.
- Decisione di mantenere TensorFlow/Keras.
- Ambiente locale `.venv` presente e ricerca Vast.ai riuscita.
- Copia ridotta preparata e verificata nei suoi file essenziali.
- Individuata candidata RTX 3090 46021771 nell'elenco.
- Identificata la causa del controllo `false`: query ancora generale e controllo su un altro ID.
- Scritto questo handoff; nessuna modifica funzionale a self-ai o FaceDancer eseguita da Codex.

### Non fatto / non confermato

- Query specifica aggiornata con risultato `true`.
- Configurazione e chiave SSH effettive.
- Attivazione completa dell'ambiente della copia ridotta.
- Creazione di una nuova istanza per questa prova: **nessuna creazione confermata**; account/istanze reali DA VERIFICARE prima di procedere.
- ID istanza, URL SSH, host e porta: **DA VERIFICARE**, non disponibili.
- Timer: non avviato; directory `runs/` non presente al controllo finale.
- Installazione ambiente remoto, trasferimento, verifica GPU, training, backup e distruzione.
- Costo reale della prova e tempo per iterazione.
- Launcher automatico FaceDancer, container dedicato, checkpoint completi degli ottimizzatori: non implementati e fuori dalla prima prova manuale.

## 18. Cosa NON fare

- Non scambiare ID offerta e ID istanza.
- Non creare risorse se jq restituisce `false`, se i dati sono vecchi o se la ricerca è vuota.
- Non usare una query generale quando si intende selezionare un'offerta specifica.
- Non dare per esistente la chiave SSH d'esempio.
- Non leggere/stampare/copiare credenziali in questo documento, nei log, nei tool output o nei commit.
- Non assumere che il saldo di 10 USD sia un limite rigido alla spesa.
- Non creare istanze duplicate dopo un timeout senza verificare quella precedente.
- Non lasciare la macchina attiva o soltanto fermata dopo la prova senza controllare i costi.
- Non spegnere/sospendere Windows o WSL confidando nel timer locale.
- Non aspettare il timer dopo aver concluso: recuperare e distruggere subito.
- Non distruggere la macchina prima del backup, salvo decisione esplicita di rinunciare ai risultati per fermare i costi.
- Non cancellare directory/checkpoint originali di FaceDancer.
- Non scaricare dataset completi per la prova di tre iterazioni.
- Non trasferire `.devenv`, `.direnv`, virtualenv locali o symlink irrisolti al Nix store.
- Non migrare a PyTorch, aggiornare TensorFlow/Keras o cambiare architettura per questa prova.
- Non avviare il training lungo del README (es. 100 epoche × 10.000 iterazioni) al posto della prova.
- Non modificare automaticamente lock, fatturazione/account o configurazioni di altri esperimenti.

## NEXT STEP

**Prossimo passo concreto, senza costi:** ripristinare il terminale locale e le variabili, verificare `FD_STAGE`, poi eseguire la ricerca specifica di **46021771** con `--storage 60` e il controllo jq della sezione 11.

Il risultato deve essere **`true`**. Se non lo è, analizzare il nuovo output e scegliere un'offerta disponibile prima di procedere.

**Prima di qualsiasi noleggio, verificare inoltre la chiave SSH reale e creare `configuration`: questi prerequisiti non risultano presenti nel filesystem osservato.** Dopo queste verifiche e la richiesta dell'utente di proseguire, eseguire la sezione 12 con il secondo terminale pronto per il timer.
