-- Deploy stupyd:02_sqlite_translator_schema to pg

BEGIN;

CREATE SCHEMA src;

CREATE TABLE src.corsi_di_studio (
       ID INTEGER,
       TIPO_CODICE TEXT,
       TIPO TEXT,
       CODICE TEXT,
       NOME TEXT,
       DIPARTIMENTO TEXT,
       DIPARTIMENTO_ACRONIMO TEXT,
       PRIMARY KEY (id)
);


CREATE TABLE src.studenti (
       ID INTEGER,
       --NOME TEXT,
       --COGNOME TEXT,
       --CODICE_FISCALE TEXT,
       SESSO TEXT,
       DATA_NASCITA TEXT,
       --EMAIL_PERSONALE TEXT,
       --TELEFONO_RESIDENZA TEXT,
       --TELEFONO_DOMICILIO TEXT,
       --CELLULARE TEXT,
       --EMAIL_ISTITUZIONALE TEXT,
       --VIA_RESIDENZA TEXT,
       --NUMERO_CIVICO_RESIDENZA TEXT,
       COMUNE_RESIDENZA TEXT,
       CAP_RESIDENZA TEXT,
       PRIMARY KEY (id)
);

CREATE TABLE src.matricole (
       ID INTEGER,
       STUDENTE_ID INTEGER,
       CORSO_DI_STUDIO_ID INTEGER,
       ANNO_REGOLAMENTO INTEGER,
       MATRICOLA TEXT,
       DATACHIUSURACARRIERA TEXT,
       ORDINAMENTO_ID TEXT,
       PERCORSO_ID TEXT,
       STATO_STUDENTE TEXT,
       STATO_MATRICOLA TEXT,
       MOTIVO_STATO_MATRICOLA TEXT,
       AA_INIZIO_CARRIERA INTEGER,
       ATTIVO INTEGER,
       ISCRITTO_PART_TIME INTEGER,
       PERIODO_PART_TIME INTEGER,
       ISCRITTO_RIPETENTE INTEGER,
       LAUREATO INTEGER,
       PREIMMATRICOLATO INTEGER,
       AAIMMSU INTEGER,
       ANNI_PART_TIME INTEGER,
       ANNO_CORSO INTEGER,
       PRIMARY KEY (id)
);


CREATE TABLE src.riga_libretto (
      ID INTEGER,
      MATRICOLA_ID INTEGER,
      ATTIVITA_DIDATTICA_ID INTEGER,
      DESCRIZIONE TEXT,
      CODICE TEXT,
      PESO float,
      VOTO float,
      LODE INTEGER,
      DATA TEXT,
      TIPO_GIUD_COD TEXT,
      STA_SCE_COD TEXT,
      ANNO_CORSO INTEGER,
      CDS_AD_ID INTEGER,
      AA_OFF_ID INTEGER,
      PRIMARY KEY (id)
);




CREATE VIEW src.vw_matricole AS
SELECT
    id,
    studente_id,
    corso_di_studio_id,
    anno_regolamento,
    matricola,

    CASE
        WHEN nullif(trim(datachiusuracarriera), '')::timestamp::date IS NULL
        THEN NULL
        ELSE datachiusuracarriera::timestamp::date
    END AS closure_date,

    ordinamento_id,
    percorso_id,
    stato_studente,
    stato_matricola,
    motivo_stato_matricola,

    aa_inizio_carriera,

    CASE
        WHEN aa_inizio_carriera BETWEEN 1901 AND 2099
        THEN make_date(aa_inizio_carriera, 9, 1)
    END AS enrollment_date,

    COALESCE(attivo, 0) <> 0 AS active,
    COALESCE(iscritto_part_time, 0) <> 0 AS part_time,
    periodo_part_time,
    COALESCE(iscritto_ripetente, 0) <> 0 AS repeating,
    COALESCE(laureato, 0) <> 0 AS graduated,
    COALESCE(preimmatricolato, 0) <> 0 AS pre_enrolled,

    aaimmsu,
    anni_part_time,
    anno_corso

FROM src.matricole;


COMMIT;
