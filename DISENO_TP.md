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
  que se lee mejor con nombre (`up_chirp`, `modular`, `demodular`, `correlacion_fft`,
  `sincronizar`, `demodular_buffer`); las que se usan una sola vez van inline en su
  sección. El archivo se lee de arriba hacia abajo, con variables sueltas por bloque,
  y los bucles se permiten.
- **Identificadores en español** y comentarios en español.
- **Librerías**: `numpy` (importada como `np`), `matplotlib.pyplot` (importada como
  `plt`) y `adi` (pyadi-iio), que es el driver con el que se habla con el Pluto.
  Las dos primeras son las del enunciado; `adi` entra en el bloque del SDR y en el
  del PER, y el análisis de las Partes 1 y 3 no depende de ella.
- **Alcance**: no se simulan canal AWGN/selectivo, curvas BER/PER ni Monte Carlo.
  El ruido y el canal los aporta el enlace real entre placas. El PER no se simula:
  se **mide** sobre el enlace real repitiendo la recepción (Parte 4).
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
| `np.fft.fft`, `.ifft`, `.fftshift` | correlación por FFT y espectro del dechirp | §4, §6, §8 |
| `np.abs`, `np.real`, `np.imag` | módulo para métricas y figuras | §3, §4, §6, §8 |
| `np.conj` | filtro adaptado: conjugado del chirp | dechirp (§4), correlación (§6) |
| `np.dot` | símbolo contra la base | §2, §4 |
| `np.concatenate`, `np.tile` | armar la trama y extender el buffer circular | §5, §8 |
| `np.argmax` | decidir el símbolo y ubicar el inicio | §4, §6 |
| `np.sqrt`, `np.sum`, `np.linalg.norm` | normalizar amplitudes y medir la métrica | §1, §6 |
| `np.mean` | potencia media del buffer simulado | §7 |
| `np.random.default_rng` | semilla determinista para bits y ruido | §2, §7 |

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
| `fc` | 928 MHz | portadora (solo la usa el SDR) |
| `n_up` | 8 | up-chirps del preámbulo |
| `n_down` | 2 | down-chirps completos |
| `cuarto_down` | `True` | agregar el último 1/4 de down-chirp |
| `n_simbolos` | 10 | símbolos del payload |
| `tramas_en_buffer` | 2 | tramas que captura el receptor |
| `n_paquetes_per` | 30000 | paquetes (tramas) evaluados para medir el PER |

Los últimos seis parámetros (`n_up`, `n_down`, `cuarto_down`, `n_simbolos`,
`tramas_en_buffer`, `n_paquetes_per`) se escriben en minúsculas. Los físicos
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

Dos figuras generadas dentro del `.ipynb`, sin guardar imágenes aparte:

1. **Último símbolo del payload modulado**: parte real e imaginaria en el tiempo y
   espectrograma. Se usa `simbolos_payload[-1]`, no un símbolo fijo, para que el
   gráfico siga al bloque de bits.
2. **Trama completa**: `Re{trama}` contra muestra, con líneas verticales en
   `n_up·M` (fin de los up-chirps) y en `longitud_preambulo` (fin del preámbulo e
   inicio del payload). Se acompaña con la vista por segmentos.

Todas las figuras usan `plt.show()` y no escriben archivos.

---

## 3. Parte 2 — SDR: configuración y emisión

**Responsabilidad**: configurar el ADALM-PLUTO y dejar la trama emitiendo en forma
cíclica, para que el receptor la capture en cualquier instante. Esta parte **no**
captura ni decodifica: eso es la Parte 3.

> Estado: implementado y revisado contra el HowTo; **no verificado con hardware**
> (las placas no estuvieron accesibles).
>
> **Orden implementado en el `.ipynb`**: transmisor (§1–§5) → SDR (§6) → receptor
> (§7–§9) → PER (§10) → cierre (§11) → conclusiones (§12). La configuración del SDR
> quedó antes del receptor y el cierre al final del archivo.

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
| 2 | `sample_rate` | `1e6` SPS |
| 3 | `tx_rf_bandwidth` / `rx_rf_bandwidth` | `2e6` Hz |
| 4 | `tx_hardwaregain_chan0` | `-60` dB (primer valor; el script usaba `-10`) |
| 4 | `gain_control_mode_chan0` | `slow_attack` (AGC, sin ganancia manual) |
| 5 | `tx_lo` / `rx_lo` | `928e6` Hz |
| 6 | `tx_cyclic_buffer` | `True` (**antes** de transmitir) |
| 6 | `rx_buffer_size` | `2 · longitud_trama = 10368` muestras |

Notas de la tabla:

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
   la Parte 4 mida el PER con capturas repetidas. El cierre es obligatorio y va en un
   bloque aparte, al final del archivo, después del receptor y del PER (sección 6).

### 3.4 Requisito de hardware

Las secciones del SDR, del receptor y del PER necesitan el Pluto y `pyadi-iio`, igual
que el HowTo: no hay `try/except` ni degradación automática. Si no está la placa, la
ejecución se corta en la sección del SDR y no sigue hasta el receptor, el PER ni las
conclusiones. La Parte 1 corre sin hardware.

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

- `tx_lo = rx_lo = 928 MHz`.
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
   para el receptor y el PER.

### 3.7 Gráficos del SDR

Esta parte no genera gráficos por sí sola: los muestra la Parte 3 sobre la captura
recibida (ver 4.4).

---

## 4. Parte 3 — Receptor

**Responsabilidad**: recuperar los `n_simbolos` símbolos del payload a partir de una
captura del SDR que puede empezar en cualquier punto del buffer.

> Estado: implementado y verificado en simulación; **no verificado con hardware**.

### 4.1 Cadena

```
buffer circular (2 tramas) → extensión circular
→ detección en tiempo: correlación con el preámbulo (calculada por FFT) → inicio
→ detección en frecuencia: dechirp + FFT sobre la ventana del payload → símbolo
→ bits
```

### 4.2 Puntos críticos ya decididos

- **Captura**: un `sdr.rx()` devuelve exactamente `rx_buffer_size` =
  `tramas_en_buffer · longitud_trama` = 10368 muestras. Son dos repeticiones
  consecutivas que arrancan en un instante cualquiera: pueden empezar en medio del
  payload y pueden dar la vuelta por el borde del buffer. Como la ventana mide dos
  tramas y la señal es periódica, al menos una copia completa cae dentro de la
  captura, sin importar dónde arranque. Esa es toda la sincronización que da el
  hardware. Se pueden pedir capturas seguidas sin reconfigurar: cada `rx()` entrega
  una ventana nueva de la misma señal.
- **Detección en tiempo (correlación)**: se correlaciona la captura con el
  preámbulo completo y el inicio es el máximo. La correlación se calcula por FFT y
  se normaliza para informar una `metrica` entre 0 y 1.
- **Buffer circular**: se extiende la captura por ambos lados para que una trama que
  da la vuelta aparezca contigua. Se lee siempre la copia central, la que cae dentro
  del buffer, para que la ventana del payload entre completa.
- **Detección en frecuencia (FFT)**: `Y = FFT(r · exp(-j2πk²/M))`; el índice del
  pico es el símbolo. Se aplica sobre la ventana del payload ya sincronizada en
  tiempo, con la misma FFT que usa la correlación.
- **Sin CFO**: TX y RX comparten oscilador, así que no hay corrimiento de frecuencia
  que deforme la correlación. Se decidió no implementar estimación ni corrección de
  CFO porque no forma parte del alcance de la materia y no hace falta para este
  montaje.

### 4.3 Criterios de aceptación

1. `captura.size == tramas_en_buffer · longitud_trama`; si el driver entrega menos,
   el código lo advierte.
2. Loopback digital (`loopback = 1`) recupera exactamente los `n_simbolos` símbolos
   del payload: valida TX, RX y la demodulación sin irradiar.
3. La captura no está recortada: sin mesetas planas en los picos de `Re{captura}` ni
   `Im{captura}`.
4. Se puede repetir `sdr.rx()` sin reconfigurar y las dos capturas decodifican el
   mismo payload: prueba de que el TX sigue ciclando y el enlace quedó vivo.
5. La trama puede cruzar cualquiera de los dos bordes del buffer y el inicio
   detectado se informa módulo la longitud del buffer.
6. Con offset y ruido hasta `SNR = 0 dB` los `n_simbolos` símbolos coinciden con los
   transmitidos; con `SNR = 0 dB` la métrica del pico no baja de 0.8.

> Los criterios 5 y 6 salían de pruebas sintéticas que ya se retiraron del
> entregable. La verificación real es con las placas y con el buffer capturado.

### 4.4 Gráficos del receptor

Un solo gráfico, de la **señal recibida** (antes figuraba en la parte del SDR):

- Arriba: `Re{captura}` e `Im{captura}` en el tiempo, las dos tramas completas.
- Abajo: `|captura|`, la envolvente. El rizado sobre la envolvente es el ruido del
  canal y el desvanecimiento del enlace; con el TX ciclando sin pausa no hay huecos
  de silencio, así que la diferencia con el ruido se ve como rizado, no como tramos
  vacíos.
- No se grafica la señal transmitida: la trama ya se muestra en la Parte 1 (§2.7) y
  acá lo que interesa es qué llegó.

---

## 5. Parte 4 — PER (Packet Error Rate)

**Responsabilidad**: medir cuántas de las tramas recibidas se decodifican mal,
repitiendo la recepción de la **misma** señal muchas veces. La señal transmitida es
periódica y siempre la misma trama, así que cada captura es una observación nueva del
mismo paquete y el receptor conoce de antemano los símbolos esperados.

> Estado: **implementado y verificado en simulación** con un SDR simulado; falta la
> corrida con hardware. No se puede validar el tiempo real de 30 000 capturas sin la
> placa.

### 5.1 Idea

Como el TX cicla la misma trama, recepciones sucesivas entregan tramas idénticas en
el aire. Comparando lo decodificado contra lo transmitido se cuenta, sobre `N`
paquetes, cuántos salieron con error. Eso da una estimación empírica del PER del
enlace real, sin simulaciones de Monte Carlo.

### 5.2 Procedimiento

1. Fijar `n_paquetes_per` (parámetro editable de la tabla 1): **30000 paquetes**.
2. Repetir la recepción hasta juntar `n_paquetes_per` paquetes: `captura = sdr.rx()`
   y decodificar con la Parte 3 (extensión circular → sincronización →
   demodulación). Cada captura aporta **un** paquete: la copia central que decodifica
   `demodular_buffer`.
3. Comparar los `n_simbolos` decodificados contra los símbolos transmitidos
   (`simbolos_payload`), que se conocen porque el receptor es del mismo TP.
4. Contar un paquete como erróneo si **al menos un** símbolo no coincide. El conteo
   de símbolos erróneos se informa como métrica complementaria, pero el PER que se
   informa es por paquete.
5. Acumular los contadores entre capturas y entre corridas de la celda:
   `paquetes_recibidos += 1` y `paquetes_erroneos += 1` si hubo error. Al volver a
   correr la celda se completan solo los paquetes que faltan hasta `n_paquetes_per`,
   así que una corrida cortada se puede retomar.
6. Informar `PER = paquetes_erroneos / paquetes_recibidos` junto con los dos
   contadores. Sin `N` grande el PER no es confiable: mostrarlo siempre con su `N`.

### 5.3 Cantidad de paquetes y tiempo

- `n_paquetes_per = 30000` es el objetivo: con esa `N`, un solo error da un PER de
  `3.3e-5`, y si no hay ninguno el intervalo superior (regla de tres) es `1e-4`. Es
  el orden de magnitud que hace medible un PER bajo.
- Cada paquete dura `longitud_trama / SamplingRate` = 5.184 ms, así que 30 000
  paquetes son ≈ 155 s de señal en el aire.
- Cada `sdr.rx()` trae `tramas_en_buffer = 2` copias, pero la implementación evalúa
  **una sola** (la copia central) para que las observaciones no estén correlacionadas:
  hacen falta 30 000 capturas. Cada captura ocupa 10.368 ms de buffer, así que el
  piso de tiempo es ≈ 311 s más el procesamiento por captura (unos 5 minutos de
  señal); conviene medirlo antes de fijar la corrida final.
- La segunda copia de cada captura queda sin usar. Aprovecharla reduciría las
  capturas a la mitad, pero las dos copias están separadas por 5 ms y no son
  observaciones independientes; se prefiere el conteo limpio.

### 5.4 Decisiones tomadas

- **Acumular entre corridas**: sí. Permite cortar y retomar sin reconfigurar y con el
  enlace caliente; el reinicio es explícito (tres asignaciones a cero comentadas al
  principio de la celda). Si cambia `TxAtten` o la distancia, hay que reiniciar para
  no mezclar condiciones.
- **Una copia por captura**: se evalúa solo la copia central, aunque el buffer traiga
  dos. Preferible a contar copias correlacionadas entre sí.
- **Criterio de error**: por símbolo (el paquete es erróneo si algún símbolo falla),
  con el conteo de símbolos erróneos como métrica extra.

### 5.5 Criterios de aceptación

1. La celda de PER no reconfigura el SDR: usa el `sdr` que dejó abierto la Parte 2.
2. Cada captura se decodifica con la misma cadena de la Parte 3, sin duplicar lógica.
3. `PER = paquetes_erroneos / paquetes_recibidos`, informado junto con `N`.
4. En loopback digital (`loopback = 1`) el PER da 0 sobre los 30 000 paquetes: sirve
   de control de la cadena completa.
5. Los contadores se pueden reiniciar sin cerrar el enlace.

### 5.6 Gráfico del PER

Implementado: `PER` acumulado contra paquetes recibidos, con un punto cada 10 % del
objetivo. Complementa el `print` con `N`, paquetes erróneos, símbolos erróneos y
`PER` final.

---

## 6. Cierre del enlace (obligatorio, al final del archivo)

El cierre no es opcional: si no se ejecuta, el Pluto queda conectado y emitiendo.
Va en un bloque aparte **al final del archivo**, después del receptor y del PER,
porque el archivo se ejecuta por bloques y mientras el enlace esté abierto se puede
volver a capturar para confirmar la recepción y sumar paquetes al PER.

- Orden del cierre, idéntico al HowTo: `tx_destroy_buffer()`,
  `tx_hardwaregain_chan0 = -89`, `tx_lo = 2400 MHz`, `rx_lo = 950 MHz`,
  `tx(np.zeros(2048))`. Sin guardas: si el enlace está abierto, el cierre corre.
- Motivo: el Pluto no tiene `tx_off`; para callarlo hay que bajarle la potencia y
  sacarlo de la banda. Dejarlo emitiendo en 928 MHz al terminar la práctica molesta
  a los demás grupos.
- Si se vuelve a correr la parte del SDR después del cierre, se crea un objeto nuevo
  y el enlace queda abierto otra vez.
- Criterio: el bloque corre sin excepciones y deja el TX atenuado (`-89 dB`) y en
  2400 MHz. Si se ejecuta dos veces, la segunda informa que no hay enlace abierto.

---

## 7. Plan de progreso

| Parte | Diseño | Implementación | Verificación |
|---|---|---|---|
| 1. Transmisor | Cerrado | En `tp_final.ipynb` §1–§5 | Simulación: OK |
| 2. SDR (config + emisión) | Cerrado | En `tp_final.ipynb` §6 | **Pendiente: hardware** |
| 3. Receptor | Cerrado (borrador) | En `tp_final.ipynb` §7–§9 | Simulación: OK |
| 4. PER | Cerrado | En `tp_final.ipynb` §10 | Simulación: OK; **hardware pendiente** |
| 5. Cierre | Cerrado | En `tp_final.ipynb` §11 | **Pendiente: hardware** |

**Deuda pendiente del entregable**:

- [x] Parámetros del PHY y dependencia de `SF` (configuración en minúsculas).
- [x] Transmisor: bits aleatorios deterministas, modulación y trama.
- [x] Gráficos del transmisor: último símbolo del payload y trama completa.
- [x] Receptor: sincronización por correlación, demodulación y buffer de 2 tramas.
- [x] SDR: configuración del HowTo, IP del grupo y RX de 2 tramas.
- [x] Cierre obligatorio en bloque aparte, al final del archivo.
- [x] Ganancia de RX moderada (30 dB) y documentada como nunca-máximo.
- [x] **Reordenar el `.ipynb`**: SDR antes del receptor; cierre al final, después del
  PER.
- [x] **Implementar la Parte 4 (PER)**: recepción repetida de la misma señal,
  comparación contra los símbolos transmitidos y contadores acumulados para 30 000
  paquetes.
- [ ] Reemplazar los 18 `def` por código lineal (restricción del entregable).
- [x] Sacar las pruebas sintéticas de la sección 7 (cadena completa con offset y
  ruido). La sección 9 se conserva porque modela el buffer de recepción de dos
  tramas.
- [ ] Verificar la cadena completa con las placas del laboratorio.
- [ ] Ajustar `TxAtten` en la medición real (el RX va con AGC).
- [ ] Medir el tiempo real de 30 000 capturas para el PER con el enlace abierto.
