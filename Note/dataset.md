# Struttura del dataset

Basandosi sui dati condivisi, la struttura ***ipogeo*** è composta da 3 lotti, ognuno dei quali è caratterizzato da un certo numero di celle. Per ogni cella troviamo le variabili:

- `TemperaturaCelle`: temperatura della cella, risoluzione di $0.1^\circ C$; feature della quale vogliamo fare la previsione temporale;
- `TemperaturaCellaMediata`: temperatura mediata della cella, ma la cui origine deve ancora essere chiarita;
- `TemperaturaMandataGlicole`: temperatura reale del glicole in entrata alla cella;
- `TemperaturaMandataGlicoleNominale`: temperatura teorica del glicole che dovrebbe arrivare in ingresso alla cella (setpoint del sistema);
- `TemperaturaRitornoGlicole`: temperatura reale del  glicole alla linea di ritorno;
- `TemperaturaMele`: temperatura misurata al decimo di grado al nocciolo di una mela mediante un'apposita sonda;
- `PercentualeAperturaValvolaMiscelatrice`: valore percentuale (range $[0,100]$) di apertura della valvola miscelatrice in ingresso al sistema;
- `TemperaturaRoccia1`, `TemperaturaRoccia2`, `TemperaturaRoccia3`, `TemperaturaRoccia4`: temperature misurate al decimo di grado;
- `UmiditaRelativa`: valore percentuale (range $[0,100]$);
- `Preventilazione`, `Postventilazione`: valore binario acceso (1) o spento (0);
- `PompaGlicoleMarcia`: valore binario acceso (1) o spento (0). Di fatto la pompa si accende solamente quando la valvola è aperta, altrimenti è spenta;
- `PercentualeVelocitaVentilatori`: assume solo due valori: acceso (100) o spento (0).

Altre variabili non documentate: `InterventoTermostatoAntigelo`, `PompaAvviamentiGiornalieri`, `Raffreddamento` (sembra coincidente con `PompaGlicoleMarcia`), `SbrinamentoAcqua`, `SbrinamentoAcquaAvviamenti`, `SbrinamentoAria`, `SelettoreFrigoManuale`, `SgocciolamentoDopoSbrinamentoAcqua`, `Umidificazione`, `UmidificazioneAvviamenti`, `ValvolaSbrinamentoAcquaAperta`, `VentilatoreAvviamentiGiornalieri`, `VentilatoreMarcia` (sembra coincidente con `PercentualeVelocitaVentilatori`), `VentilazioneAntistratificazionePortaAperta`, `VentilazioneForzata`.

# Controllo: lo stato attuale
L'apertura della valvola miscelatrice è controllata dalla temperatura del glicole sulal linea del ritorno (l'idea è che all'interno del circuito della cella la temperatura del fluido rientri in parametri prestabiliti).

