# Diseño del TP Integrador — LoRa/FSCM

**Materia**: Procesamiento de Señales
**Grupo 8**: "Testigos de Vangelista" — Castro, Victor José Arturo · Carreras, Leonor Victoria

Este documento fija, parte por parte, qué se debe usar y qué se debe hacer en el
entregable. No reemplaza al código: es la referencia de diseño contra la que se
revisa `tp_final.ipynb`.

**Referencia teórica**: L. Vangelista, *Frequency Shift Chirp Modulation: The LoRa
Modulation* (IEEE SPL, 2017). Ecuaciones 1, 2/3, 15 y 16.

---

## 0. Convenciones y restricciones

- **Un solo archivo**: `tp_final.ipynb`, formato de cuaderno (celdas markdown con el
  texto y celdas de código, `plt.show()` por bloque, sin `argparse` ni `main()`).
- **Sin clases.** Las funciones se reservan para la lógica que se usa más de una vez o
  que se lee mejor con nombre (`up_chirp`, `modular`, `demodular`,
  `construir_trama`, `correlacion_tiempo`, `correlacion_fft`, `detectar_tiempo`,
  `detectar_frecuencia`, `fer_simulada`); las que se usan una sola vez van inline en
  su sección. El archivo se lee de arriba hacia abajo, con variables sueltas por
  bloque, y los bucles se permiten.
- **Identificadores en español** y comentarios en español.
- **Librerías**: `numpy` (importada como `np`), `matplotlib.pyplot` (importada como
  `plt`) y `adi` (pyadi-iio), que es el driver con el que se habla con el Pluto.
  Las dos primeras son las del enunciado; `adi` entra en la configuración del SDR y
  en el FER con hardware, y el análisis del transmisor no depende de ella.
- **Banda usable**: 900–920 MHz; la portadora se elige en 915 MHz.
- **Alcance**: el FER del preámbulo se simula en Monte Carlo con canal FLAT y
  selectivo en frecuencia, y además se **mide** sobre el enlace real repitiendo la
  recepción (20 000 capturas).
- El transmisor y el receptor comparten los parámetros del PHY de la sección 1.

### 0.1 Uso de numpy

Todo el procesamiento es vectorial: no hay listas de Python para señales ni bucles
por muestra. Estas son las familias de numpy que se usan, y por qué.

| API | Para qué | Aparece en |
|---|---|---|
| `np.arange` | índices de tiempo y vector de bins | chirps, portadora, figuras |
| `np.asarray`, `np.zeros` | construir vectores y fijar el tipo | todo el archivo |
| `np.complex128`, `np.int64` | señal en complejo; vector de TX en enteros de 16 bits | chirps, capturas, §2 |
| `np.exp`, `np.pi` | exponenciales complejas de fase cuadrática | chirp (Eq. 3), portadora |
| `np.fft.fft`, `.ifft` | correlación por FFT y espectro del dechirp | §2, §4, §5 |
| `np.abs`, `np.real`, `np.imag` | módulo para métricas y figuras | §3, §4, §5 |
| `np.conj` | filtro adaptado: conjugado del chirp | dechirp (§2), correlación (§4) |
| `np.dot` | símbolo contra la base | §2, §4 |
| `np.concatenate`, `np.tile` | armar la trama y el preámbulo | §2, §5 |
| `np.roll` | desplazar la captura y los ecos del canal | §5 |
| `np.cumsum` | energía acumulada para la métrica de detección | §4 |
| `np.argmax` | decidir el símbolo y ubicar el inicio | §2, §4 |
| `np.sqrt`, `np.sum`, `np.linalg.norm` | normalizar amplitudes y medir la métrica | §1, §4 |
| `np.mean` | potencia media de la señal y del ruido | §1, §5 |
| `np.random.default_rng` | semilla determinista para bits, ruido y desplazamientos | §2, §5 |

---

## 1. Parámetros del PHY (comunes)

| Parámetro | Valor | Rol |
|---|---|---|
| `SF` | 8 | bits por símbolo |
| `M = 2**SF` | 256 | **derivado**: muestras por símbolo = símbolos posibles |
| `B` | 1 MHz | ancho de banda del chirp |
| `SamplingRate` (`Fs` en las ecuaciones) | 1 MSPS | tasa del SDR (una muestra por chip) |
| `T = 1/B` | 1 µs | **derivado**: período de chip |
| `Ts = M/B` | 256 µs | **derivado**: duración de un símbolo |
| `fc` | 915 MHz | portadora dentro de la banda 900–920 MHz (solo la usa el SDR) |
| `n_up` | 8 | up-chirps del preámbulo |
| `n_down` | 2 | down-chirps completos |
| `cuarto_down` | `True` | agregar el último 1/4 de down-chirp |
| `n_simbolos` | 10 | símbolos del payload |
| `tramas_en_buffer` | 2 | tramas que captura el receptor |
| `NFLAT` | 50000 | tramas por punto del canal FLAT |
| `NSELECTIVE` | 50000 | tramas por punto del canal selectivo |
| `snr_fer_db` | −12..−1 | barrido de `Eb/N0` en pasos de 1 dB |
| `n_tramas_enlace` | 20000 | capturas del FER con el SDR |

Los últimos parámetros (`n_up`, `n_down`, `cuarto_down`, `n_simbolos`,
`tramas_en_buffer`, `snr_fer_db`, `n_tramas_enlace`) se escriben en minúsculas;
`NFLAT` y `NSELECTIVE` quedan en mayúsculas para distinguirlos. Los físicos
(`SF`, `M`, `B`, `T`, `Ts`, `fc`) van en mayúsculas; la tasa del SDR es
`SamplingRate`.

**Dependencia de `SF`**: todo lo derivado se calcula en el código a partir de sus
bases, nunca como constante escrita a mano:

```
M                    = 2**SF
T                    = 1/B
Ts                   = M/B
longitud_preambulo   = (n_up + n_down)*M + (M//4 si cuarto_down)
longitud_trama       = longitud_preambulo + n_simbolos*M
tamaño_buffer_rx     = tramas_en_buffer*longitud_trama
```

Cambiar `SF` debe propagarse solo a `M`, `Ts` y a todas las longitudes. Las
cifras de la sección 2.4 son el caso `SF = 8`; las fórmulas son las que mandan.

---

## 2. Parte 1 — Transmisor

**Responsabilidad**: convertir un bloque de bits en una trama lista para entregar
al SDR. No conoce la portadora ni el hardware.

### 2.1 Cadena

```
bits → símbolos → chirps (FSCM) → trama = preámbulo + payload
```

### 2.2 Bits → símbolo (Ecuación 1)

Cada símbolo es la suma ponderada de `SF` bits, con el **bit menos significativo
primero**:

```
s = Σ(h = 0..SF-1) w_h · 2^h
```

- Entrada: vector de `SF` bits (0/1), LSB primero.
- Salida: entero en `[0, M)`.
- Debe existir la conversión inversa (símbolo → bits) para verificar el roundtrip.

**Bloque de bits**: aleatorio pero determinista. El generador produce los **bits**,
no los símbolos: se crea una matriz de `(n_simbolos, SF)` bits con
`numpy.random.default_rng(semilla_bits)` e `integers(0, 2, size=(n_simbolos, SF))`.
Con la misma `semilla_bits` la corrida da siempre los mismos bits; al cambiarla
cambian los valores binarios. Los símbolos se obtienen recién después aplicando la
Ecuación 1 a cada fila. `semilla_bits` es un parámetro editable de esta sección,
no de la tabla del PHY.

### 2.3 Modulación FSCM (Ecuaciones 2 y 3)

El chirp del símbolo `s` sobre `k = 0..M-1`:

```
c_s(k) = (1/√M) · exp( j·2π·[((s + k) mod M) · k] / M )
```

- Amplitud por muestra: `1/√M` (energía unitaria por símbolo).
- El índice va módulo `M`: el chirp da la vuelta al pasar el final. Eso es lo que
  hace ortogonales a los `M` símbolos.
- El transmisor **no aplica portadora**: entrega banda base compleja. El `fc` es
  del enlace.

**El chirp es exactamente el de Vangelista** (Ecuación 3), sin ventanas ni
tapering:

- El up-chirp base es `s = 0`; el down-chirp es su conjugado.
- `Fs = B`: una muestra por chip. No hay sobremuestreo (`sps = 1`), así que no hace
  falta remuestrear ni decimar.
- La normalización `1/√M` se aplica por muestra. El demodulador usa la réplica sin
  normalizar y solo compara el `argmax`, así que la escala no cambia el símbolo.

Verificado sobre el código: `modular([s])` coincide muestra a muestra con la
Ecuación 3 (error máximo 0) y `|⟨c_s, c_r⟩| < 3e-17` para `s ≠ r`, con energía 1
por símbolo.

### 2.4 Armado de la trama

| Segmento | Contenido | Muestras (SF = 8) | Offset |
|---|---|---:|---:|
| Preámbulo, up | `n_up` up-chirps idénticos | `n_up·M = 2048` | 0 |
| Preámbulo, down | `n_down` down-chirps (conjugado del up) | `n_down·M = 512` | 2048 |
| Preámbulo, cola | 1/4 de down-chirp (`M/4`), si `cuarto_down` | `64` | 2560 |
| Payload | `n_simbolos` símbolos modulados | `n_simbolos·M = 2560` | 2624 |
| **Total** | | **5184** | |

Valores para `SF = 8` y los parámetros por defecto. En el código las longitudes se
calculan con las fórmulas de la sección 1, no con estos números.

- `longitud_preambulo = (n_up + n_down)·M + (M/4 si cuarto_down)`.
- `longitud_trama = longitud_preambulo + n_simbolos·M`.
- El preámbulo es una **firma única**: los 8 up-chirps solos darían un pico de
  correlación por cada copia; los down-chirps finales rompen esa repetición y
  dejan un solo pico, que marca el comienzo de la trama.

### 2.5 Contrato de salida

- Vector complejo 1-D, `longitud_trama` muestras, `complex128`.
- Sin ruido, sin portadora, sin escalado del DAC.
- El escalado a la escala del Pluto (`2**15 - 1`) lo hace la Parte 2.

### 2.6 Criterios de aceptación

1. `bits → símbolo → bits` reproduce exactamente el vector original.
2. Preámbulo = 2624 y trama = 5184 muestras para `SF = 8`; ambas escalan con `M`.
3. Un símbolo modulado y demodulado sin ruido devuelve el mismo símbolo.
4. La trama completa, demodulada sin ruido, devuelve los `n_simbolos` símbolos.
5. El bloque de bits es reproducible: dos corridas con la misma `semilla_bits` dan
   los mismos bits, y con otra semilla cambian.
6. Los `M` chirps son ortogonales: `|⟨c_s, c_r⟩| ≈ 0` para `s ≠ r` (dentro del
   redondeo numérico).
7. Cambiar `SF` cambia `M`, `Ts` y las longitudes sin editar ninguna constante.

### 2.7 Gráficos del transmisor

Cuatro figuras generadas dentro del `.ipynb`, sin guardar imágenes aparte, con las
funciones `plotSignal` y `plotSpect` tomadas del HowTo:

1. **Señal del preámbulo**: `Re{}` e `Im{}` en el tiempo, en dos subplots
   (`figsize=(20, 8)`).
2. **PSD del preámbulo**: `plt.psd` con `NFFT=len(signal)` y `Fs=SamplingRate`
   (`figsize=(15, 3)`).
3. **Señal de la trama completa**: preámbulo + payload, con la misma `plotSignal`.
4. **PSD de la trama completa**: la misma `plotSpect`.

Todas las figuras usan `plt.show()` y no escriben archivos.

---

## 3. Parte 2 — SDR: configuración y emisión

**Responsabilidad**: configurar el ADALM-PLUTO y dejar la trama emitiendo en forma
cíclica, para que el receptor la capture en cualquier instante. Esta parte **no**
captura ni decodifica: eso es la Parte 3.

> Estado: implementado y verificado en loopback digital (captura de 10 368 muestras,
> inicio 3612, métrica 1.0); falta la corrida final irradiando.
>
> **Orden implementado en el `.ipynb`**: transmisor (celdas 1–13) → receptor
> (14–16: señal, PSD y FER con el SDR) → cierre del enlace (17).

### 3.1 Conexión

| Dato | Valor |
|---|---|
| VPN de la cátedra | `200.16.19.5:443` (usuario provisto por la cátedra) |
| JupyterHub | `192.168.1.60` |
| SDR del grupo 8 | `192.168.1.34` |
| URI de pyadi-iio | `ip:192.168.1.34` |

### 3.2 Configuración (orden del HowTo)

| Paso | Parámetro | Valor |
|---|---|---|
| 1 | `loopback` | `0` (deshabilitado) |
| 1b | `ensm_mode` | `fdd` (full duplex; ver nota) |
| 2 | `sample_rate` | `1e6` SPS |
| 3 | `tx_rf_bandwidth` / `rx_rf_bandwidth` | `2e6` Hz |
| 4 | `tx_hardwaregain_chan0` | `-60` dB (primer valor; el script usaba `-10`) |
| 4 | `gain_control_mode_chan0` | `slow_attack` (AGC, sin ganancia manual) |
| 5 | `tx_lo` / `rx_lo` | `915e6` Hz |
| 6 | `tx_cyclic_buffer` | `True` (**antes** de transmitir) |
| 6 | `rx_buffer_size` | `2 · longitud_trama = 10368` muestras |

Notas de la tabla:

- `ensm_mode = fdd` es obligatorio para recibir. Si otra sesión dejó el chip en
  `alert`, el TX puede configurarse pero el RX no entrega muestras: `sdr.rx()`
  termina en `Unable to dequeue block` (timeout 110) aunque el buffer se cree
  bien. La configuración lo fuerza en cada corrida.
- La recepción va con el AGC del Pluto (`slow_attack`): el código **no** fija
  ganancia manual. No es un olvido: en pyadi-iio el setter de
  `rx_hardwaregain_chan0` ignora el valor si el modo no es `manual`, así que
  escribirlo con `slow_attack` sería una línea muerta. Si alguna vez se pasa a
  `manual`, hay que fijar el modo **antes** que la ganancia.
- **Si se usa ganancia manual, nunca el máximo.** Indicación del profesor: llevar
  la ganancia manual al extremo superior puede dañar el dispositivo.
- `TxAtten` es el único control de nivel del enlace: se ajusta en la medición
  según saturación y SNR. El HowTo usa `-60 dB`; el script de laboratorio usó
  `-10 dB`.
- `tx_cyclic_buffer = True` se escribe **antes** de `sdr.tx()`, y `rx_buffer_size`
  **antes** del primer `sdr.rx()`: el driver crea los buffers la primera vez que se
  usan, así que después ya no se redimensionan.
- `RxBufferSize = tramas_en_buffer · longitud_trama` (10368 muestras para
  `SF = 8`), muy por debajo del límite de buffer del Pluto. Se configura acá, pero
  quien captura y decodifica es la Parte 3.

### 3.3 Secuencia de emisión

1. Configurar el SDR en el orden de la tabla 3.2.
2. Escalar la trama y entregarla: `sdr.tx(trama * (2**15 - 1))`. El Pluto espera
   muestras entre `-(2**15 - 1)` y `+2**15 - 1` (enteros de 16 bits en complemento
   a dos), no en el rango `[-1, 1]`: por eso se escala. La trama es un vector
   complejo 1-D de `longitud_trama` muestras.
3. Como el buffer es cíclico, esa única llamada deja la trama repitiéndose sin
   pausa: el Pluto la reinyecta sola, no hay que llamar a `tx()` en un bucle.
4. **No cerrar acá**: el enlace queda abierto para que la Parte 3 capture y para que
   el receptor mida el FER con capturas repetidas. El cierre es obligatorio y va en
   un bloque aparte, al final del archivo, después del receptor (sección 7).

### 3.4 Requisito de hardware

Las secciones del SDR y del FER con hardware necesitan el Pluto y `pyadi-iio`, igual
que el HowTo: no hay `try/except` ni degradación automática. Si no está la placa, la
ejecución se corta en la sección del SDR y no sigue hasta el receptor ni el cierre.
La Parte 1 corre sin hardware.

### 3.5 Contrato de salida y nivel de señal

Qué entra y qué sale de esta parte, y con qué criterio se ajusta la señal.

**TX**

- Entrada: `trama`, complejo 1-D de `longitud_trama` muestras (5184 para `SF = 8`).
- Escalado: `trama * (2**15 - 1)`, porque el Pluto espera enteros de 16 bits y no
  el rango `[-1, 1]`.
- El buffer es cíclico: la trama se repite espalda con espalda, sin gap ni cabecera.
  La señal en el aire queda periódica de período `longitud_trama`, y eso es lo que
  explota la Parte 3.
- No hay trigger ni marca de inicio: el aire solo tiene la señal repetida.

**Captura (Parte 3)**

- El `sdr.rx()` y la ventana de `rx_buffer_size` son responsabilidad de la Parte 3,
  aunque el tamaño se configura en la tabla 3.2. Ver 4.2.

**Nivel**

- La captura tiene que entrar en el rango del ADC. Si los picos muestran mesetas
  planas (recorte), se baja `TxAtten` (más negativa) o se aleja la antena; si la
  señal es débil, se acerca la antena o se sube `TxAtten` hacia 0.
- Con `slow_attack` el AGC ajusta solo: no hay ganancia manual que tocar. El nivel
  se controla con `TxAtten`.
- No se busca maximizar la amplitud: se busca no recortar. Dechirp + FFT no necesita
  señal grande, necesita señal limpia.

**Frecuencia**

- `tx_lo = rx_lo = 915 MHz` (banda usable 900–920 MHz).
- TX y RX son la misma placa y comparten la referencia de reloj, así que no hay
  corrimiento de frecuencia entre ellos. El receptor no estima ni corrige CFO:
  sincroniza por correlación y demodula con dechirp + FFT.
- El lazo local prueba el camino digital completo (TX, canal, RX, sincronización y
  decodificación), que es lo que se pide.

### 3.6 Criterios de aceptación

1. Con el Pluto encendido y `pyadi-iio` instalado, la sección del SDR configura el
   enlace sin excepciones.
2. `sdr.tx()` acepta la trama escalada y el buffer cíclico queda activo: la señal se
   repite sin volver a llamar a `tx()`.
3. El RX va con AGC (`slow_attack`), sin ganancia manual; `TxAtten` no se deja en
   0 dB salvo justificación medida.
4. La conexión **no** se cierra al terminar esta parte: el objeto `sdr` queda vivo
   para el receptor y el FER con hardware.

### 3.7 Gráficos del SDR

Esta parte no genera gráficos por sí sola: los muestra la Parte 3 sobre la captura
recibida (ver 4.4).

---

## 4. Parte 3 — Receptor

**Responsabilidad**: a partir de una captura del SDR, detectar el preámbulo y medir
cuántas tramas se detectan bien. La recuperación del payload ya quedó validada en el
transmisor (§2.3, celda de demodulación ida y vuelta) y en el loopback digital.

> Estado: detección implementada y verificada en simulación y en hardware (loopback
> digital); el conteo de FER con el SDR queda para la corrida final.

### 4.1 Cadena

```
captura de 2 tramas → detección del preámbulo
  en tiempo: correlación directa
  en frecuencia: correlación por FFT
→ inicio y métrica (0 a 1) → ¿métrica ≥ 0.5? → trama detectada o fallada
```

La demodulación del payload desde el inicio detectado es la misma de §2.3
(`dechirp + FFT`); el receptor del entregable la usa para validar la cadena, no para
el conteo de FER.

### 4.2 Puntos críticos ya decididos

- **Captura**: un `sdr.rx()` devuelve `rx_buffer_size = tramas_en_buffer ·
  longitud_trama = 10368` muestras, dos repeticiones consecutivas que arrancan en un
  instante cualquiera. Como la señal es periódica, al menos una copia completa cae
  dentro de la captura sin importar dónde arranque.
- **Detección en tiempo**: `correlacion_tiempo` calcula `|correlate(captura, patrón)|`
  en forma directa. Se usa en la demostración porque es la definición más legible.
- **Detección en frecuencia**: `correlacion_fft` multiplica las FFT y vuelve al tiempo
  con `ifft`; con `nfft` potencia de dos y `mode="valid"`. Es la que corre en el
  receptor porque es más rápida y escala a lotes 2D.
- **Métrica**: `pico / (||patrón|| · sqrt(energía de la ventana))`, entre 0 y 1. En el
  receptor real no se conoce la posición verdadera, así que la decisión es
  `métrica ≥ 0.5`; en la simulación se compara contra la posición real, que es más
  estricto porque una falsa alarma también cuenta como falla.
- **Sin CFO**: TX y RX comparten oscilador, así que no hay corrimiento de frecuencia
  que deforme la correlación. No se implementa estimación ni corrección de CFO.

### 4.3 Criterios de aceptación

1. `captura.size == tramas_en_buffer · longitud_trama`; si el driver entrega menos,
   el código lo advierte.
2. Loopback digital (`loopback = 1`) recupera exactamente los `n_simbolos` símbolos
   del payload: valida TX, RX y la demodulación sin irradiar.
3. En la demostración con la trama desplazada y ruido, los dos métodos informan el
   mismo inicio y una métrica alta.
4. Se puede repetir `sdr.rx()` sin reconfigurar y las capturas siguen decodificando el
   mismo payload.
5. El conteo de FER corre sobre 20 000 capturas sin reconfigurar el enlace.

### 4.4 Gráficos del receptor

- **Señal recibida**: `Re{captura}` e `Im{captura}` en el tiempo.
- **PSD de la señal recibida**: `plt.psd` con `NFFT` y `Fs` del HowTo.

No se grafica la señal transmitida: la trama ya se muestra en el transmisor y acá lo
que interesa es qué llegó.

---

## 5. FER en simulación

**Responsabilidad**: medir la fracción de tramas cuyo preámbulo el detector no
encuentra, en dos canales simulados, barriendo la SNR.

> Estado: **implementado**; la corrida de 50 000 tramas por punto tarda ≈55 min con `lote=50`.

### 5.1 Criterio de error

- Se genera la **trama completa** (preámbulo + payload) y se la rota circularmente una
  cantidad aleatoria de muestras (modela una captura que arranca en cualquier punto de
  la trama periódica).
- Se pasa por el canal **por convolución** con la respuesta al impulso, frame por
  frame, y se suma ruido gaussiano complejo.
- Se detecta con `detectar_frecuencia` en modo circular, usando **solo el preámbulo**
  como patrón, y se compara el inicio detectado con el desplazamiento real: si
  coincide, la trama se considera correcta; cualquier otro pico (incluida una falsa
  alarma) cuenta como falla.
- `FER = falladas / simuladas`. El payload está presente en la captura, pero no se
  demodula: el criterio es únicamente la detección del preámbulo. **A confirmar
  mañana** si el payload debe evaluarse de otra forma.
- **Acumulación y checkpoint**: el FER se acumula por bloques de 500 tramas. Después
  de cada bloque se guarda `fer_checkpoint.npz` con `fallas` y `hechas`; si el proceso
  se corta, se retoma desde el último bloque. El archivo incluye una versión y se
  reinicia solo si cambia el canal, el ruido o la captura.
- **Tramas fijas**: `NFLAT = 50 000` y `NSELECTIVE = 50 000` tramas por punto. No hay
  criterio adaptativo: se recorren los 12 valores de SNR con la misma cantidad de
  tramas. La resolución del FER es `1/50 000`.
- **Lote**: `lote_fer = 50` procesa 50 tramas por FFT. La memoria sigue siendo chica
  (~2 MB por captura) y el tiempo total es ≈55 min; el lote no cambia el estimador,
  solo la memoria y la velocidad.
- **Alcance de la detección**: el patrón de correlación es **solo el preámbulo**
  (2624 muestras). En el receptor real la búsqueda recorre **toda la captura** de dos
  tramas (10 368 muestras), incluido el payload; en la simulación del FER la captura
  es una trama completa (5184 muestras) rotada circularmente, así que la búsqueda
  recorre toda la trama pero el patrón sigue siendo el preámbulo.

### 5.2 Canales

| Canal | Modelo |
|---|---|
| FLAT | Solo AWGN: `r = s + n`, sin multitrayecto |
| Selectivo en frecuencia | `h(nT) = √0.8·δ(nT) + √0.2·δ(nT − T)`, dos taps separados 1 muestra, aplicado por convolución frame por frame. Las raíces dan energía unitaria (`0.8 + 0.2 = 1`) |

El eco llega una muestra (1 µs) después del camino directo. La coherencia de banda
es del orden de `1/T = 1 MHz`, comparable a `B`. El paper escribe `0.8` y `0.2` y
llama al canal "unit energy"; en el TP se usan `√0.8` y `√0.2` para que la energía
sea exactamente 1, como en `Untitled3.ipynb`. El ruido es gaussiano complejo de media
cero con `Pz = (1/M)/SNR` y `σ = √(Pz/2)` por parte real/imaginaria (ecuación (10)).

### 5.3 Barrido y figura

- `SNR = Eb/N0` de −12 a −1 dB en pasos de 1 dB. Con `SNR_por_muestra` en ese rango el
  FER sería 0 en todos los puntos; la relación usada es
  `SNR_por_muestra = Eb/N0 − 10·log10(M/SF)`, con un offset de 15.05 dB para
  `M = 256`, `SF = 8`.
- `NFLAT = NSELECTIVE = 50 000` tramas fijas por punto. `lote_fer = 50` y checkpoint
  cada 500 tramas. La corrida completa son 12 × 50 000 × 2 = 1 200 000 tramas.
- Detección por correlación **circular** por FFT: es la más rápida y es coherente con
  un TX que repite la trama sin pausa. Un detector lineal sobre una ventana corta
  subestimaría el FER porque tendría muchas menos posiciones donde producir falsas
  alarmas.
- Gráfico semilogarítmico con las dos curvas y un tick por cada SNR, de −12 a −1; un
  FER de 0 se dibuja en `1/(2·NFLAT)`.
- Resultado: la corrida de 50 000 tramas por punto se ejecuta al correr la celda y
  llena la tabla y el gráfico. Con `lote=50` tarda ≈55 min. Una prueba rápida con 200
  tramas por punto mostró el selectivo por encima del FLAT, como se espera; la corrida
  final de 50k da la curva suave.

### 5.4 Criterios de aceptación

1. Ambas curvas bajan al crecer `Eb/N0`.
2. La curva selectiva queda por encima de la FLAT, como en el canal selectivo del
   paper.
3. El resultado es reproducible con las semillas fijas.
4. La sección corre sin hardware: no toca el SDR.
5. El checkpoint permite retomar sin repetir los puntos ya completados.

---

## 6. FER con el SDR

**Responsabilidad**: medir empíricamente cuántas capturas reales se detectan bien.
Como el TX cicla la misma trama, cada `sdr.rx()` es una observación nueva del mismo
paquete.

### 6.1 Procedimiento

1. `n_tramas_enlace = 20000` capturas.
2. En cada una: `captura = sdr.rx()` y `detectar_frecuencia(captura)`.
3. Se cuenta como detectada si `métrica ≥ 0.5`; si no, fallada. No se usa la posición
   verdadera porque en el enlace real no se conoce.
4. Se informa el avance cada 2000 capturas y al final `FER = falladas / 20000`, junto
   con los dos contadores. Sin gráfico.

### 6.2 Tiempo

- Cada captura de 10 368 muestras tarda ≈60 ms (≈50 ms son overhead del backend de
  red, no de la señal); la detección agrega pocos ms.
- 20 000 capturas ≈ 20 minutos. La celda lo advierte antes de arrancar.

### 6.3 Criterios de aceptación

1. La celda no reconfigura el SDR: usa el `sdr` que dejó abierto la configuración.
2. Usa la misma `detectar_frecuencia` de §4, sin duplicar lógica.
3. Informa `FER` junto con `N` y los contadores.
4. En loopback digital el FER da 0: sirve de control de la cadena completa.

---

## 7. Cierre del enlace (obligatorio, al final del archivo)

El cierre no es opcional: si no se ejecuta, el Pluto queda conectado y emitiendo.
Va en un bloque aparte **al final del archivo**, después del receptor, porque el
archivo se ejecuta por bloques y mientras el enlace esté abierto se puede volver a
capturar.

- Orden del cierre, idéntico al HowTo: `tx_destroy_buffer()`,
  `tx_hardwaregain_chan0 = -89`, `tx_lo = 2400 MHz`, `rx_lo = 950 MHz`,
  `tx(np.zeros(2048))`. Sin guardas: si el enlace está abierto, el cierre corre.
- Motivo: el Pluto no tiene `tx_off`; para callarlo hay que bajarle la potencia y
  sacarlo de la banda. Dejarlo emitiendo en 915 MHz al terminar la práctica molesta
  a los demás grupos.
- Si se vuelve a correr la parte del SDR después del cierre, se crea un objeto nuevo
  y el enlace queda abierto otra vez.
- Criterio: el bloque corre sin excepciones y deja el TX atenuado (`-89 dB`) y en
  2400 MHz.

---

## 8. Estructura del entregable

El notebook tiene una celda por subpunto, en este orden:

| Celda | Bloque | Contenido |
|---|---|---|
| 1 | Transmisor | Parámetros del PHY y del preámbulo |
| 2 | Transmisor | De bits a símbolo |
| 3 | Transmisor | Modulación de símbolo (`up_chirp`, `down_chirp`, `modular`, `preambulo`) |
| 4 | Transmisor | Gráfico de la señal del preámbulo (`plotSignal`) |
| 5 | Transmisor | PSD del preámbulo (`plotSpect`) |
| 6 | Transmisor | Gráfico de la trama completa |
| 7 | Transmisor | PSD de la trama completa |
| 8 | Transmisor | Demodulación (ida y vuelta) |
| 9 | Transmisor | Armado de la trama (`construir_trama`) |
| 10 | Transmisor | Detección en tiempo y en frecuencia + demostración |
| 11 | Transmisor | FER vs SNR (FLAT y selectivo) |
| 12 | Transmisor | Configuración del SDR |
| 13 | Transmisor | Envío de la señal al SDR |
| 14 | Receptor | Gráfico de la señal recibida |
| 15 | Receptor | PSD de la señal recibida |
| 16 | Receptor | FER con 20 000 tramas |
| 17 | Cierre | Cierre del enlace |

**Deuda pendiente del entregable**:

- [x] Parámetros del PHY y dependencia de `SF`.
- [x] Transmisor: bits aleatorios deterministas, modulación, trama y demodulación.
- [x] Gráficos del transmisor: preámbulo, trama completa y sus PSD.
- [x] Detección en tiempo (correlación directa) y en frecuencia (FFT), por separado.
- [x] FER en simulación (FLAT y selectivo), SNR −12..−1, 50 000 tramas fijas por punto con checkpoint.
- [x] SDR: configuración del HowTo, IP del grupo y RX de 2 tramas.
- [x] Receptor: gráficos de señal y PSD, FER con 20 000 tramas.
- [x] Cierre obligatorio en bloque aparte, al final del archivo.
- [ ] Reemplazar los `def` por código lineal (restricción del entregable).
- [ ] Correr el FER con el SDR en el laboratorio (≈20 min).
- [ ] Ajustar `TxAtten` en la medición real (el RX va con AGC).

